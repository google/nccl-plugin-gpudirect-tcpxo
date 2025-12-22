/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpxo_prober/src/agent.h"

#include <grpcpp/client_context.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include <algorithm>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h"
#include "google/rpc/code.pb.h"
#include "tcpxo_prober/src/connection-manager.pb.h"
#include "tcpxo_prober/src/nic-manager.h"
#include "tcpxo_prober/src/prober.pb.h"

namespace dxs::prober {

namespace {
absl::Status CreateDirectory(const std::string& directory) {
  std::filesystem::path path = directory;
  std::error_code ec;
  if (std::filesystem::create_directories(path, ec)) {
    return absl::OkStatus();
  } else if (ec) {
    return absl::InternalError(absl::StrCat(
        "Error creating directory: ", directory, ": ", ec.message()));
  }
  return absl::OkStatus();
}
}  // namespace

AgentServiceImpl::AgentServiceImpl(std::shared_ptr<Agent> agent)
    : agent_(agent) {}

grpc::Status AgentServiceImpl::StartPings(
    grpc::ServerContext* context, const dxs::prober::StartPingsRequest* request,
    dxs::prober::StartPingsReply*) {
  agent_->ReportResults();
  agent_->TearDownAllPingConnections();
  agent_->SetProbeInterval(request->probe_rate_qps());

  LOG(INFO) << "StartPings() setting up new connections.";
  google::rpc::Status status =
      agent_->SetUpConnections(request, context->deadline());

  return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                      status.message(), status.SerializeAsString());
}

grpc::Status AgentServiceImpl::StopPings(grpc::ServerContext* context,
                                         const dxs::prober::StopPingsRequest*,
                                         dxs::prober::StopPingsReply*) {
  agent_->ReportResults();
  agent_->TearDownAllPingConnections();
  return grpc::Status::OK;
}

ConnectionManagerServiceImpl::ConnectionManagerServiceImpl(
    std::shared_ptr<Agent> agent)
    : agent_(agent) {}

grpc::Status ConnectionManagerServiceImpl::SetUpPongConnection(
    grpc::ServerContext* context, const dxs::prober::SetUpRequest* request,
    dxs::prober::SetUpReply* reply) {
  LOG(INFO) << "SetUpPongConnection() request received: "
            << request->DebugString();
  absl::StatusOr<NicManager*> conn_manager =
      agent_->GetConnectionManager(request->server_gpu_ip());
  if (!conn_manager.ok()) {
    LOG(ERROR) << "Failed to get connection manager for NIC IP: "
               << request->server_gpu_ip() << conn_manager.status();
    return grpc::Status::CANCELLED;
  }

  absl::StatusOr<int> port =
      (*conn_manager)->CreatingPongConnection(request, context->deadline());
  if (!port.ok()) {
    LOG(ERROR) << "Failed to create pong connection: " << port.status();
    return grpc::Status::CANCELLED;
  }
  reply->set_dxs_port(*port);
  return grpc::Status::OK;
}

grpc::Status ConnectionManagerServiceImpl::TearDownConnection(
    grpc::ServerContext* context, const dxs::prober::TearDownRequest* request,
    dxs::prober::TearDownReply*) {
  LOG(INFO) << "TearDownConnection() request received: "
            << request->DebugString();
  std::string_view local_nic_ip = request->ping_ip_is_servers_nic()
                                      ? request->ping_ip()
                                      : request->pong_ip();
  absl::StatusOr<NicManager*> conn_manager =
      agent_->GetConnectionManager(local_nic_ip);
  if (!conn_manager.ok()) {
    LOG(ERROR) << conn_manager.status();
    return grpc::Status::CANCELLED;
  }

  absl::Status status = (*conn_manager)->RemoveConnection(request);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to tear down connection: " << status;
    return grpc::Status::CANCELLED;
  }
  return grpc::Status::OK;
}

Agent::Agent(Agent::Options options) : options_(options) {
  results_file_counter_ = 0;
  probe_interval_ = absl::Seconds(1);
}

void Agent::SetProbeInterval(int qps) {
  if (options_.max_probe_qps > 0 && qps > options_.max_probe_qps) {
    LOG(WARNING) << "Probe QPS will be limited to " << options_.max_probe_qps
                 << " QPS.";
    qps = options_.max_probe_qps;
  }
  probe_interval_ = absl::Seconds(1.0 / qps);
}

absl::StatusOr<NicManager* absl_nonnull> Agent::GetConnectionManager(
    std::string_view nic_ip) {
  for (const std::unique_ptr<NicManager>& conn_manager : connection_managers_) {
    if (conn_manager->GetLocalNicIp() == nic_ip) {
      return conn_manager.get();
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No connection manager found with IP: ", nic_ip));
}

google::rpc::Status Agent::SetUpConnections(
    const dxs::prober::StartPingsRequest* request,
    const std::chrono::system_clock::time_point deadline) {
  LOG(INFO) << "Setting up new ping connections.";
  int num_threads = std::min(request->targets_size(), 10);
  boost::asio::thread_pool pool(num_threads);
  std::vector<ConnectionResults::Result> results(request->targets_size());
  bool verify_payload = request->verify_payload();
  for (int i = 0; i < request->targets_size(); ++i) {
    const dxs::prober::Target& target = request->targets(i);
    boost::asio::post(
        pool, [this, i, target, verify_payload, deadline, &results]() {
          results[i] = SetUpConnection(target, verify_payload, deadline);
        });
  }
  pool.join();

  ConnectionResults probe_connection_results;
  google::rpc::Status rpc_status;
  for (const ConnectionResults::Result& result : results) {
    probe_connection_results.add_results(result);
    if (result != ConnectionResults::RESULT_SUCCESS) {
      rpc_status.set_code(google::rpc::Code::INTERNAL);
      rpc_status.set_message("Failed to set up at least one connection.");
    }
  }
  rpc_status.add_details()->PackFrom(probe_connection_results);
  return rpc_status;
}

ConnectionResults::Result Agent::SetUpConnection(
    const dxs::prober::Target& target, bool verify_payload,
    std::chrono::system_clock::time_point deadline) {
  absl::StatusOr<NicManager*> conn_manager =
      GetConnectionManager(target.local_nic_ip_address());
  if (!conn_manager.ok()) {
    LOG(ERROR) << conn_manager.status();
    return ConnectionResults::RESULT_FAILURE;
  }

  return (*conn_manager)
      ->CreatePingConnection(target.peer_nic_ip_address(), verify_payload,
                             deadline);
}

void Agent::TearDownAllPingConnections() {
  LOG(INFO) << "Tearing down all ping connections from the "
               "previous StartPings RPC.";
  std::vector<std::thread> threads;
  threads.reserve(connection_managers_.size());
  for (const std::unique_ptr<NicManager>& cm : connection_managers_) {
    threads.push_back(
        std::thread([&cm]() { cm->TearDownAllPingConnections(); }));
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
}

absl::Status Agent::Run() {
  if (options_.gpu_nic_ips.size() != 8) {
    LOG(WARNING) << "Agent is expected to be run with 8 GPU NICs, but got "
                 << options_.gpu_nic_ips.size() << " NICs.";
  }
  if (!options_.dxs_ips.empty() || !options_.dxs_ports.empty()) {
    if (options_.dxs_ips.size() != options_.gpu_nic_ips.size() ||
        options_.dxs_ports.size() != options_.gpu_nic_ips.size()) {
      return absl::InvalidArgumentError(
          "If set dxs_ips and dxs_ports must be the same size as gpu_nic_ips.");
    }
  }

  for (int i = 0; i < options_.gpu_nic_ips.size(); ++i) {
    std::string dxs_ip =
        options_.dxs_ips.empty() ? kDefaultDxsAddr : options_.dxs_ips[i];
    std::string dxs_port =
        options_.dxs_ports.empty() ? kDefaultDxsPort : options_.dxs_ports[i];

    connection_managers_.push_back(
        std::make_unique<NicManager>(NicManager::Options{
            options_.gpu_nic_ips[i],
            dxs_ip,
            dxs_port,
            options_.server_port,
            options_.max_rpc_deadline,
            options_.payload_size,
            options_.max_connections_per_nic,
            options_.use_llcm,
            options_.llcm_device_directory,
        }));
    RETURN_IF_ERROR(connection_managers_.back()->SetUp());
  }

  RETURN_IF_ERROR(CreateDirectory(options_.results_directory));
  CreatePingAndResultsThread();
  CreatePongThread();
  return absl::OkStatus();
}

void Agent::CreatePingAndResultsThread() {
  ping_thread_ = std::make_unique<std::thread>([this] {
    absl::Time next_file_rotation_time =
        absl::Now() + options_.results_file_rotation_interval;
    UpdateResultsFileName();

    absl::Time next_ping_round_time = absl::Now();
    absl::Time next_report_results_time = absl::Now();

    while (!cancel_threads_flag_) {
      DoPingOps();
      if (absl::Now() > next_ping_round_time) {
        next_ping_round_time = absl::Now() + probe_interval_;
        PrepareNewPingRound();
      }

      if (absl::Now() > next_report_results_time) {
        next_report_results_time =
            absl::Now() + options_.results_file_output_interval;
        ReportResults();
      }

      if (options_.results_file_rotation_interval != absl::ZeroDuration() &&
          absl::Now() > next_file_rotation_time) {
        next_file_rotation_time =
            absl::Now() + options_.results_file_rotation_interval;
        UpdateResultsFileName();
      }
    }
  });
}

void Agent::DoPingOps() {
  for (const std::unique_ptr<NicManager>& cm : connection_managers_) {
    cm->DoPingOps();
  }
}

void Agent::PrepareNewPingRound() {
  for (const std::unique_ptr<NicManager>& cm : connection_managers_) {
    cm->PrepareNewPingRound();
  }
}

void Agent::CreatePongThread() {
  pong_thread_ = std::make_unique<std::thread>([this] {
    while (!cancel_threads_flag_) {
      for (const std::unique_ptr<NicManager>& cm : connection_managers_) {
        cm->DoPongOps();
      }
    }
  });
}

void Agent::UpdateResultsFileName() {
  absl::MutexLock lock(&mu_results_);
  std::string timestamp = absl::FormatTime(absl::Now(), absl::UTCTimeZone());
  results_file_name_ = absl::StrCat(
      options_.results_directory, "/", options_.results_file_prefix, "_",
      results_file_counter_, "_", timestamp, ".csv");
  LOG(INFO) << "Created new results file: " << results_file_name_;

  if (std::filesystem::exists(results_file_name_)) {
    LOG(WARNING) << "Results file already exists: " << results_file_name_;
  }
  results_file_counter_++;
}

void Agent::ReportResults() {
  absl::MutexLock lock(&mu_results_);
  std::ofstream results_file;
  results_file.open(results_file_name_, std::ios::app);

  std::vector<std::string> results;
  for (const std::unique_ptr<NicManager>& cm : connection_managers_) {
    std::vector<std::string> conn_results = cm->GetAndClearResults();
    for (const std::string& result : conn_results) {
      results.push_back(result);
    }
  }
  LOG(INFO) << "Collected " << results.size() << " results.";
  if (results.empty()) {
    return;
  }
  results_file << absl::StrJoin(results, "\n") << "\n";
  results_file.close();
}

absl::Status Agent::CleanUp() {
  LOG(INFO) << "Cleaning up agent.";
  // Deallocate GPU memory, tear down all active connections and notify remote
  // agents we are shutting down.
  std::vector<std::thread> threads;
  threads.reserve(connection_managers_.size());
  bool all_ok = true;
  for (const std::unique_ptr<NicManager>& cm : connection_managers_) {
    threads.push_back(std::thread([&cm, &all_ok]() {
      absl::Status status = cm->Cleanup();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to clean up connection manager: " << status;
        all_ok = false;
      }
    }));
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  ReportResults();
  cancel_threads_flag_ = true;
  if (ping_thread_ != nullptr) {
    ping_thread_->join();
  }
  if (pong_thread_ != nullptr) {
    pong_thread_->join();
  }

  return all_ok ? absl::OkStatus()
                : absl::InternalError("Failed to clean up Agent.");
}

}  // namespace dxs::prober

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/flags.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "tcpxo_prober/src/agent.h"

constexpr char gpu_nic_ips_env_var[] = "TCPXO_PROBER_GPU_NIC_IPS";
constexpr char server_port_env_var[] = "TCPXO_PROBER_SERVER_PORT";
constexpr char max_probe_qps_env_var[] = "TCPXO_PROBER_MAX_QPS_LIMIT";
constexpr char max_connections_per_nic_limit_env_var[] =
    "TCPXO_PROBER_MAX_CONNECTIONS_PER_NIC_LIMIT";
constexpr char fastrak_prober_payload_size_env_var[] =
    "TCPXO_PROBER_PAYLOAD_SIZE";
constexpr char enable_llcm_env_var[] = "TCPXO_PROBER_ENABLE_LLCM";
constexpr char output_file_prefix_env_var[] = "TCPXO_PROBER_OUTPUT_FILE_PREFIX";
constexpr char output_file_path_env_var[] = "TCPXO_PROBER_OUTPUT_FILES_PATH";
constexpr char output_file_rotation_interval_in_seconds_env_var[] =
    "TCPXO_PROBER_FILE_ROTATION_INTERVAL_IN_SECONDS";
constexpr char output_interval_in_seconds_env_var[] =
    "TCPXO_PROBER_OUTPUT_INTERVAL_IN_SECONDS";

// Non-owning pointer to the server so we can properly shut it down when
// receiving a SIGINT or SIGTERM.
grpc::Server* server = nullptr;
std::atomic<bool> shutdown_requested = false;

void signal_handler(int signal) {
  LOG(INFO) << "Shutting down. Received signal: " << signal;
  shutdown_requested = true;
  shutdown_requested.notify_all();
}

bool ParseBool(std::string str) {
  // Convert the string to lowercase for case-insensitive comparison
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (str == "true" || str == "t" || str == "1" || str == "yes" || str == "y") {
    return true;
  } else {
    return false;
  }
}

dxs::prober::Agent::Options GetAgentOptions() {
  dxs::prober::Agent::Options options;

  char* gpu_nic_ips = std::getenv(gpu_nic_ips_env_var);
  if (gpu_nic_ips == nullptr) {
    LOG(FATAL) << "Environment variable " << gpu_nic_ips_env_var
               << " is not set. Exiting";
  } else {
    std::vector<std::string> gpu_nics = absl::StrSplit(gpu_nic_ips, ',');
    if (gpu_nics.empty()) {
      LOG(FATAL) << "Environment variable " << gpu_nic_ips_env_var
                 << "must have 8 items, but has " << gpu_nics.size()
                 << " items. Exiting";
    }
    options.gpu_nic_ips = gpu_nics;
  }

  char* server_port = std::getenv(server_port_env_var);
  if (server_port == nullptr) {
    LOG(WARNING) << "Environment variable " << server_port_env_var
                 << " is not set. Using default value 8080.";
    options.server_port = 8080;
  } else {
    options.server_port = std::stoi(server_port);
  }

  char* max_probe_qps = std::getenv(max_probe_qps_env_var);
  if (max_probe_qps == nullptr) {
    LOG(WARNING) << "Environment variable " << max_probe_qps_env_var
                 << " is not set. The default value is 5 qps.";
    options.max_probe_qps = 5;
  } else {
    options.max_probe_qps = std::stoi(max_probe_qps);
  }

  char* max_connections_per_nic_limit =
      std::getenv(max_connections_per_nic_limit_env_var);
  if (max_connections_per_nic_limit == nullptr) {
    LOG(WARNING) << "Environment variable "
                 << max_connections_per_nic_limit_env_var
                 << " is not set. The default value disables the limit.";
    options.max_connections_per_nic = 0;
  } else {
    options.max_connections_per_nic = std::stoi(max_connections_per_nic_limit);
  }

  char* payload_size = std::getenv(fastrak_prober_payload_size_env_var);
  if (payload_size == nullptr) {
    LOG(WARNING) << "Environment variable "
                 << fastrak_prober_payload_size_env_var
                 << " is not set. Using default value 1.";
    options.payload_size = 1;
  } else {
    options.payload_size = std::stoi(payload_size);
  }

  char* enable_llcm = std::getenv(enable_llcm_env_var);
  if (enable_llcm == nullptr) {
    LOG(WARNING) << "Environment variable " << enable_llcm_env_var
                 << " is not set. Using default value true.";
    options.use_llcm = true;
  } else {
    options.use_llcm = ParseBool(enable_llcm);
  }

  char* output_file_prefix = std::getenv(output_file_prefix_env_var);
  if (output_file_prefix == nullptr) {
    LOG(WARNING) << "Environment variable " << output_file_prefix_env_var
                 << " is not set. Using default value tcpxo_probe_results.";
    options.results_file_prefix = "tcpxo_probe_results";
  } else {
    options.results_file_prefix = output_file_prefix;
  }

  char* output_file_path = std::getenv(output_file_path_env_var);
  if (output_file_path == nullptr) {
    LOG(WARNING) << "Environment variable " << output_file_path_env_var
                 << " is not set. Using default value /tmp.";
    options.results_directory = "/tmp";
  } else {
    options.results_directory = output_file_path;
  }

  char* output_file_rotation_interval_in_seconds =
      std::getenv(output_file_rotation_interval_in_seconds_env_var);
  if (output_file_rotation_interval_in_seconds == nullptr) {
    LOG(WARNING) << "Environment variable "
                 << output_file_rotation_interval_in_seconds_env_var
                 << " is not set. The default value is 3600 seconds.";
    options.results_file_rotation_interval = absl::Seconds(3600);
  } else {
    options.results_file_rotation_interval =
        absl::Seconds(std::stoi(output_file_rotation_interval_in_seconds));
  }

  char* output_interval_in_seconds =
      std::getenv(output_interval_in_seconds_env_var);
  if (output_interval_in_seconds == nullptr) {
    LOG(WARNING) << "Environment variable "
                 << output_interval_in_seconds_env_var
                 << " is not set. Using default value 5 seconds.";
    options.results_file_output_interval = absl::Seconds(5);
  } else {
    options.results_file_output_interval =
        absl::Seconds(std::stoi(output_interval_in_seconds));
  }

  return options;
}

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  dxs::prober::Agent::Options options = GetAgentOptions();
  std::shared_ptr<dxs::prober::Agent> agent =
      std::make_shared<dxs::prober::Agent>(options);
  QCHECK_OK(agent->Run());

  dxs::prober::AgentServiceImpl agent_service(agent);
  dxs::prober::ConnectionManagerServiceImpl connection_manager_service(agent);

  grpc::ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds =
      grpc::InsecureServerCredentials();  // NOLINT
  builder.AddListeningPort(absl::StrCat("[::]:", options.server_port), creds);
  builder.RegisterService(&agent_service);
  builder.RegisterService(&connection_manager_service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  std::thread shutdown_thread([&server] {
    shutdown_requested.wait(false);
    if (server != nullptr) {
      LOG(INFO) << "Shutting down the gRPC server.";
      server->Shutdown();
    }
  });

  server->Wait();
  shutdown_requested = true;
  shutdown_requested.notify_all();
  shutdown_thread.join();

  absl::Status status = agent->CleanUp();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to clean up agent: " << status;
    return 1;
  }
  return 0;
}

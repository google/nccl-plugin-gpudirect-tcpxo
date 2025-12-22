/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_TESTING_PROBER_AGENT_H_
#define DXS_TESTING_PROBER_AGENT_H_

#include <grpcpp/client_context.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "dxs/client/dxs-client.h"
#include "google/protobuf/empty.pb.h"
#include "tcpxo_prober/src/connection-manager.grpc.pb.h"
#include "tcpxo_prober/src/connection-manager.pb.h"
#include "tcpxo_prober/src/nic-manager.h"
#include "tcpxo_prober/src/prober.grpc.pb.h"
#include "tcpxo_prober/src/prober.pb.h"

namespace dxs {
namespace prober {

// Agent sets up probe connections, tears down probe connections, manages
// the ping-pong process and reports the result of probes for multiple
// local GPU NICs. One Agent should be run for each a3-mega VM.
class Agent {
 public:
  struct Options {
    // List of the local GPU NIC IPs the agent will manage.
    std::vector<std::string> gpu_nic_ips;
    // The port Agents use for gRPC. It must be the same for all Agents involved
    // in a probing session.
    int server_port;
    // The maximum time allowed for the SetUpPongConnection and
    // TearDownConnection RPCs.
    absl::Duration max_rpc_deadline = absl::Seconds(2);
    size_t payload_size;
    // The maximum probe rate for each NIC. If the Agent is given a probe rate
    // that is higher than this value, the Agent will use the max_probe_qps. 0
    // disables the limit.
    int max_probe_qps = 5;
    // The maximum number of connections per NIC. 0 disables the limit.
    int max_connections_per_nic = 0;

    std::string results_file_prefix;
    std::string results_directory;
    absl::Duration results_file_rotation_interval;
    absl::Duration results_file_output_interval;

    // dxs_ips and dxs_ports are optional. If specified, they must be the same
    // size and in the same order as gpu_nic_ips.
    std::vector<std::string> dxs_ips;
    std::vector<std::string> dxs_ports;

    bool use_llcm;
    // the search path for LLCM PCIe devices.
    std::string llcm_device_directory = std::string(dxs::kLlcmDeviceDirectory);
  };
  explicit Agent(Options options);

  // Run sets up the DXS connections and GPU buffers for each GPU NIC. It spawns
  // two background threads, one for pinging and result reporting and one for
  // ponging. This must be called before the Agent is passed to the gRPC
  // services.
  absl::Status Run();

  // CleanUp must be called before the Agent is destroyed. CleanUp tears down
  // all active connections, deallocates GPU memory and notifies remote agents
  // we are shutting down. It also stops the ping and pong threads and reports
  // the final results.
  absl::Status CleanUp();

  // SetUpConnections creates a bi-directional DXS connection with each
  // remote GPU NIC target specified in the request. This involves sending a
  // SetUpConnection RPC to the remote GPU NIC target's Agent.
  google::rpc::Status SetUpConnections(
      const dxs::prober::StartPingsRequest* request,
      std::chrono::system_clock::time_point deadline);

  // TearDownAllPingConnections tears down every connection the Agent set up
  // from the previous StartPings RPC. It sends a TearDownConnection RPC to
  // each remote Agent that owns the pong portion of the probe connection.
  void TearDownAllPingConnections();

  // ReportResults writes the most recent batch of results to the results file.
  void ReportResults();

  void SetProbeInterval(int qps);

  absl::StatusOr<NicManager* absl_nonnull> GetConnectionManager(
      std::string_view nic_ip);

 private:
  void CreatePingAndResultsThread();
  void CreatePongThread();
  void UpdateResultsFileName();

  // DoPingOps is a non-blocking function that does the ping send and receive
  // operations.
  void DoPingOps();

  // PrepareNewPingRound resets the ping operations for each ping connection
  // that finished the previous ping round.
  void PrepareNewPingRound();

  // SetUpConnection creates a bi-directional DXS connection with a local and
  // remote DXS Server.
  ConnectionResults::Result SetUpConnection(
      const dxs::prober::Target& target, bool verify_payload,
      std::chrono::system_clock::time_point deadline);

  const Options options_;

  std::vector<std::unique_ptr<NicManager>> connection_managers_;

  absl::Duration probe_interval_;

  std::atomic<bool> cancel_threads_flag_;
  std::unique_ptr<std::thread> ping_thread_;
  std::unique_ptr<std::thread> pong_thread_;

  absl::Mutex mu_results_;
  std::string results_file_name_ ABSL_GUARDED_BY(mu_results_);
  int results_file_counter_ ABSL_GUARDED_BY(mu_results_);
};

// AgentServiceImpl implements the Agent gRPC service. This will be used by the
// external parties Controller to coordinate probing jobs.
class AgentServiceImpl : public dxs::prober::AgentService::Service {
 public:
  explicit AgentServiceImpl(std::shared_ptr<Agent> agent);

  grpc::Status StartPings(grpc::ServerContext* context,
                          const dxs::prober::StartPingsRequest* request,
                          dxs::prober::StartPingsReply*) override;

  grpc::Status StopPings(grpc::ServerContext* context,
                         const dxs::prober::StopPingsRequest*,
                         dxs::prober::StopPingsReply*) override;

 private:
  std::shared_ptr<Agent> agent_;
};

// ConnectionManagerServiceImpl implements RPC methods to set up and tear down
// DXS connections between the local and remote GPU NICs. This RPC service
// should only be used internally.
class ConnectionManagerServiceImpl
    : public dxs::prober::ConnectionManager::Service {
 public:
  explicit ConnectionManagerServiceImpl(std::shared_ptr<Agent> agent);

  grpc::Status SetUpPongConnection(grpc::ServerContext* context,
                                   const dxs::prober::SetUpRequest* request,
                                   dxs::prober::SetUpReply* reply) override;

  grpc::Status TearDownConnection(grpc::ServerContext* context,
                                  const dxs::prober::TearDownRequest* request,
                                  dxs::prober::TearDownReply*) override;

 private:
  std::shared_ptr<Agent> agent_;
};
}  // namespace prober
}  // namespace dxs
#endif  // DXS_TESTING_PROBER_AGENT_H_

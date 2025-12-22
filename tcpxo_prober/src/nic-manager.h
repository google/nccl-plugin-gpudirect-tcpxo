/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPXO_PROBER_NIC_CONNECTION_MANAGER_H_
#define TCPXO_PROBER_NIC_CONNECTION_MANAGER_H_

#include <stdbool.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "cuda_helpers/gpu_mem_helper_interface.h"
#include "cuda_helpers/gpu_mem_validator_interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "tcpxo_prober/src/connection-manager.pb.h"
#include "tcpxo_prober/src/connection.h"
#include "tcpxo_prober/src/prober.pb.h"

namespace dxs {
namespace prober {

// NicManager manages GPU memory and set of Probing connections for a single
// local DXS server. It also sets up and tears down probing connections with
// with remote Prober Agents.
class NicManager {
 public:
  struct Options {
    std::string nic_ip;
    std::string dxs_ip;
    std::string dxs_port;
    int grpc_port;
    // The maximum time allowed for the SetUpPongConnection and
    // TearDownConnection RPCs.
    absl::Duration max_rpc_deadline;
    size_t payload_size;
    int max_connections_per_nic;
    bool use_llcm;
    std::string llcm_device_directory;
  };

  explicit NicManager(Options options);

  // Connects to a local DXS server and sets up a buffer manager client.
  absl::Status SetUp();

  // Notifies remote agents that local agent is involved with probing that it is
  // shutting down. Deallocates GPU memory and tears down all active
  // connections.
  absl::Status Cleanup();

  // Sets up a bi-directional DXS connection with the remote Server.
  // This method does the following:
  //
  // 1. If verify_payload is true, creates a new receive buffer that will be
  //    used to validate the payload received is the same as the payload sent.
  // 2. Listens on a new DXS port.
  // 3. Sends an RPC to the remote Agent containing the port.
  // 4. Waits for a RPC response that contains the port the remote DXS server
  //    is listening on.
  // 5. Connect to the remote DXS server using the port from the RPC response.
  // 6. Accept the connection request from the remote DXS Server.
  ConnectionResults::Result CreatePingConnection(
      const std::string& remote_nic_ip, bool verify_payload,
      const std::chrono::system_clock::time_point& rpc_deadline);

  // Creates a ponging connection with the remote DXS Server.
  // This method does the following:
  //
  // 1. Listens on a new DXS port.
  // 2. Spawns a detached thread that will complete the connection setup after
  // the RPC returns.
  // 3. Returns the port the local DXS server is listening on to the client
  // Agent.
  // 4. The detached thread then Accepts the connection from the remote DXS
  // client and Connects to the remote DXS server.
  absl::StatusOr<int> CreatingPongConnection(
      const dxs::prober::SetUpRequest* request,
      const std::chrono::system_clock::time_point& deadline);

  // Tears down all outgoing ping connections. Notifies the remote Agents that
  // own each connection to tear down the connection.
  void TearDownAllPingConnections();

  // Tears down all incoming ping connections. Notifies the remote Agents that
  // own each connection to tear down the connection.
  void TearDownAllPongConnections();

  // Removes a connection from the list of active connections.
  absl::Status RemoveConnection(const dxs::prober::TearDownRequest* request);

  // Non-blocking method that sends a ping to the remote DXS server and waits
  // for the response for each outgoing ping connection.
  void DoPingOps();

  // PrepareNewPingRound resets the ping operations for each ping connection
  // that finished the previous ping round.
  void PrepareNewPingRound();

  // Non-blocking method that polls for messages and responds.
  void DoPongOps();

  // Returns the results of all ping operations and clears the result store.
  std::vector<std::string> GetAndClearResults();

  // CreatePayloadVerificationContext allocates a new receive buffer and creates
  // a struct with buffer helpers used for payload verification.
  absl::StatusOr<std::unique_ptr<PayloadVerificationContext>>
  CreatePayloadVerificationContext();

  std::string GetLocalNicIp() { return options_.nic_ip; }

 private:
  // Send a SetUpPongConnection RPC request to a remote Agent.
  absl::StatusOr<int> SetUpClient(
      const std::string& server_gpu_ip, int recv_port,
      const std::chrono::system_clock::time_point& deadline);

  // Send a RPC request to a remote Agent to tear down a DXS connection.
  void TearDownClient(const std::string& client_gpu_ip,
                      const std::string& server_gpu_ip,
                      bool ping_ip_is_servers_nic);

  // Completes the setup of a ponging connection. This method accepts the
  // connection from the remote DXS server and connects to the remote DXS
  // server.
  absl::Status CompletePongConnectionSetup(
      const std::string& remote_nic_ip, int connect_port, int recv_port,
      ListenSocketInterface* dxs_listen_sock,
      const std::chrono::system_clock::time_point& deadline);

  struct ListenResult {
    int recv_port;
    std::unique_ptr<ListenSocketInterface> dxs_listen_sock;
  };
  // Command DXS Server to start listening for new connections.
  absl::StatusOr<ListenResult> Listen(
      const std::chrono::system_clock::time_point& deadline);

  // Command DXS Server to connect to a remote DXS server.
  absl::StatusOr<std::unique_ptr<SendSocketInterface>> Connect(
      const std::string& remote_nic_ip, int port,
      const std::chrono::system_clock::time_point& deadline);

  // Command DXS server to accept an incoming DXS client connection.
  absl::StatusOr<std::unique_ptr<RecvSocketInterface>> Accept(
      ListenSocketInterface* listen_sock,
      const std::chrono::system_clock::time_point& deadline);

  const Options options_;

  std::unique_ptr<DxsClientInterface> dxs_client_;

  std::unique_ptr<::cuda_helpers::GpuMemHelperInterface> mem_helper_;
  std::unique_ptr<::cuda_helpers::GpuMemValidatorInterface> mem_validator_;

  std::unique_ptr<BufferManagerInterface> buffer_manager_;
  std::unique_ptr<tcpdirect::BufferManagerClientInterface>
      buffer_manager_client_;

  uint64_t send_buffer_id_;
  uint64_t recv_buffer_id_;
  Reg send_buffer_handle_;
  Reg recv_buffer_handle_;

  absl::Mutex mu_ping_connections_;
  std::vector<std::unique_ptr<PingConnection>> ping_connections_
      ABSL_GUARDED_BY(mu_ping_connections_);

  absl::Mutex mu_pong_connections_;
  std::vector<std::unique_ptr<PongConnection>> pong_connections_
      ABSL_GUARDED_BY(mu_pong_connections_);
};

}  // namespace prober
}  // namespace dxs

#endif  // TCPXO_PROBER_NIC_CONNECTION_MANAGER_H_

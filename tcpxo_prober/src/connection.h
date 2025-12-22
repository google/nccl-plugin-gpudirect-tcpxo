/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPXO_PROBER_CONNECTION_H_
#define TCPXO_PROBER_CONNECTION_H_

#include <stdbool.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "cuda_helpers/gpu_mem_helper_interface.h"
#include "cuda_helpers/gpu_mem_validator_interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"

namespace dxs {
namespace prober {

// Connection is a DXS connection between a local DXS server and a remote DXS
// server and can send and receive data.
class Connection {
 public:
  Connection(std::string local_ip, std::string peer_ip, size_t payload_size,
             Reg send_buffer_handle, Reg recv_buffer_handle,
             std::unique_ptr<SendSocketInterface> dxs_send_sock,
             std::unique_ptr<RecvSocketInterface> dxs_recv_sock);

  // Creates a new data socket for a DXS send operation.
  absl::Status CreateSendOp();

  // Creates a new data socket for a DXS recv operation.
  absl::Status CreateRecvOp();

  // Checks once if the send operation has completed. Returns true if the
  // operation completed and false if the operation is still in progress.
  absl::StatusOr<bool> TestSend();

  // Checks once if the recv operation has completed. Returns true if the
  // operation completed and false if the operation is still in progress.
  absl::StatusOr<uint64_t> TestRecv();

  std::string GetLocalIp() { return local_ip_; }

  std::string GetPeerIp() { return peer_ip_; }

  size_t GetPayloadSize() { return payload_size_; }

 private:
  const std::string local_ip_;
  const std::string peer_ip_;

  size_t payload_size_;

  Reg send_buffer_handle_;
  Reg recv_buffer_handle_;

  std::unique_ptr<SendSocketInterface> send_sock_;
  std::unique_ptr<RecvSocketInterface> recv_sock_;

  std::unique_ptr<SendOpInterface> send_op_;
  std::unique_ptr<LinearizedRecvOpInterface> recv_op_;
};

// PayloadVerificationContext contains memory buffers and helpers used to
// verify the contents of a received payload. The receive buffer is owned by
// this class.
struct PayloadVerificationContext {
  PayloadVerificationContext(
      uint64_t send_id, uint64_t recv_id, Reg recv_handle,
      cuda_helpers::GpuMemHelperInterface* mem_helper,
      cuda_helpers::GpuMemValidatorInterface* mem_validator,
      tcpdirect::BufferManagerClientInterface* manager_client);

  PayloadVerificationContext(PayloadVerificationContext&& other) = default;

  ~PayloadVerificationContext();

  // send_id is a the id of a buffer shared between all connections.
  uint64_t send_id;
  // recv_id and recv_handle are owned by this class.
  uint64_t recv_id;
  Reg recv_handle;
  cuda_helpers::GpuMemHelperInterface* mem_helper;
  cuda_helpers::GpuMemValidatorInterface* mem_validator;
  tcpdirect::BufferManagerClientInterface* manager_client;
};

// PingConnection is a connection that sends a payload to a remote DXS server
// waits for a response and then records the RTT result of the request-response.
class PingConnection {
 public:
  PingConnection(
      Connection conn,
      std::unique_ptr<PayloadVerificationContext> payload_verification);

  // If the previous probe round has finished, sets the state to start a new
  // probe round, otherwise does nothing.
  void SetStateToStartNewProbeRound();

  // SendPingAndRecordResult sends a ping to the remote DXS server, waits for
  // a response and then records the RTT result. If the probe fails, it will
  // record the failure and the connection will continue to send probes.
  void SendPingAndRecordResult();

  // Returns the result of previous ping operations and empties the results
  // vector.
  std::vector<std::string> GetAndClearResults();

  // If the previous probe round has finished, sets the state to start a new
  // probe round, otherwise does nothing.
  void PrepareNewPingRound();

  std::string GetLocalIp() { return conn_.GetLocalIp(); }

  std::string GetPeerIp() { return conn_.GetPeerIp(); }

 private:
  // Checks if the send operation has completed until the deadline. Returns true
  // if the send operation completed successfully before the deadline and
  // false if the send operation is still in progress.
  absl::StatusOr<bool> TestSendBlocking(absl::Time deadline);

  // Checks if the recv operation has completed until the deadline and validates
  // the received payload if verification is enabled. Returns true if the recv
  // operation completed successfully before the deadline and false if the recv
  // operation is still in progress.
  absl::StatusOr<bool> DoRecvBlocking(absl::Time deadline);

  // Verifies the received payload equals the sent payload and resets the
  // receive buffer to 0 to avoid stale buffer content from matching the next
  // send payload.
  absl::StatusOr<bool> VerifyAndResetRecvBuffer(uint64_t recv_size);

  // Write a 0 payload to the recv buffer.
  void ResetRecvBuffer();

  void AddResult();

  void AddFailedResult(std::string error);

  Connection conn_;

  std::unique_ptr<PayloadVerificationContext> payload_verification_;

  enum class State {
    WAIT_FOR_NEXT_PROBE_ROUND = 1,
    START_NEW_PROBE_ROUND = 2,
    START_SEND = 3,
    WAIT_FOR_SEND_RESPONSE = 4,
    START_RECV = 5,
    WAIT_FOR_RECV_RESPONSE = 6,
  };
  State state_;
  absl::Time op_start_;

  std::vector<std::string> results_;
};

// PongConnection is a connection that waits for a message from a remote DXS
// server and then sends a response. The Pong operation is non-blocking.
class PongConnection {
 public:
  explicit PongConnection(Connection conn);

  // DoPongOp is a non-blocking operations that checks if there is a message
  // from the remote DXS server and then sends a response. It is non-blocking
  // and will return immediately if there is no incoming ping.
  void DoPongOp();

  std::string GetLocalIp() { return conn_.GetLocalIp(); }

  std::string GetPeerIp() { return conn_.GetPeerIp(); }

 private:
  Connection conn_;

  enum class State {
    START_RECV = 1,
    WAIT_FOR_RECV_RESPONSE = 2,
    START_SEND = 3,
    WAIT_FOR_SEND_RESPONSE = 4,
  };
  State state_;
};

}  // namespace prober
}  // namespace dxs

#endif  // TCPXO_PROBER_CONNECTION_H_

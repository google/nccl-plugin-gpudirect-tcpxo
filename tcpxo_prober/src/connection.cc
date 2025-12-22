/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpxo_prober/src/connection.h"

#include <sys/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "cuda_helpers/gpu_mem_helper_interface.h"
#include "cuda_helpers/gpu_mem_validator_interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h"

namespace dxs::prober {

Connection::Connection(const std::string local_ip, const std::string peer_ip,
                       size_t payload_size, Reg send_buffer_handle,
                       Reg recv_buffer_handle,
                       std::unique_ptr<SendSocketInterface> dxs_send_sock,
                       std::unique_ptr<RecvSocketInterface> dxs_recv_sock)
    : local_ip_(local_ip),
      peer_ip_(peer_ip),
      payload_size_(payload_size),
      send_buffer_handle_(send_buffer_handle),
      recv_buffer_handle_(recv_buffer_handle),
      send_sock_(std::move(dxs_send_sock)),
      recv_sock_(std::move(dxs_recv_sock)) {};

absl::Status Connection::CreateSendOp() {
  ASSIGN_OR_RETURN(send_op_,
                   send_sock_->Send(0, payload_size_, send_buffer_handle_));
  return absl::OkStatus();
}

absl::Status Connection::CreateRecvOp() {
  ASSIGN_OR_RETURN(recv_op_, recv_sock_->RecvLinearized(0, payload_size_,
                                                        recv_buffer_handle_));
  return absl::OkStatus();
}

absl::StatusOr<bool> Connection::TestSend() {
  std::optional<absl::Status> status = send_op_->Test();
  if (!status.has_value()) {
    return false;
  }

  RETURN_IF_ERROR(status.value());
  return true;
}

absl::StatusOr<uint64_t> Connection::TestRecv() {
  std::optional<absl::StatusOr<uint64_t>> size = recv_op_->Test();
  if (size.has_value()) {
    RETURN_IF_ERROR(size->status());
    return size->value();
  }
  return 0;
}

PayloadVerificationContext::PayloadVerificationContext(
    uint64_t send_id, uint64_t recv_id, Reg recv_handle,
    cuda_helpers::GpuMemHelperInterface* mem_helper,
    cuda_helpers::GpuMemValidatorInterface* mem_validator,
    tcpdirect::BufferManagerClientInterface* manager_client)
    : send_id(send_id),
      recv_id(recv_id),
      recv_handle(recv_handle),
      mem_helper(mem_helper),
      mem_validator(mem_validator),
      manager_client(manager_client) {}

PayloadVerificationContext::~PayloadVerificationContext() {
  absl::Status status = manager_client->DeregBuf(recv_handle);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to deregister recv buffer: " << status;
  }
  mem_helper->FreeBuffer(recv_id);
}

PingConnection::PingConnection(
    Connection conn,
    std::unique_ptr<PayloadVerificationContext> payload_verification)
    : conn_(std::move(conn)),
      payload_verification_(std::move(payload_verification)),
      state_(State::WAIT_FOR_NEXT_PROBE_ROUND) {};

absl::StatusOr<bool> PingConnection::TestSendBlocking(absl::Time deadline) {
  while (absl::Now() < deadline) {
    ASSIGN_OR_RETURN(bool send_complete, conn_.TestSend());
    if (send_complete) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<bool> PingConnection::DoRecvBlocking(absl::Time deadline) {
  while (absl::Now() < deadline) {
    ASSIGN_OR_RETURN(uint64_t recv_size, conn_.TestRecv());
    if (recv_size == 0) {
      continue;
    }
    if (payload_verification_ != nullptr) {
      return VerifyAndResetRecvBuffer(recv_size);
    }
    return true;
  }
  return false;
}

void PingConnection::AddResult() {
  auto rtt = absl::ToInt64Nanoseconds(absl::Now() - op_start_);
  std::string timestamp = absl::FormatTime(op_start_, absl::UTCTimeZone());
  results_.push_back(
      absl::StrCat(timestamp, ",", GetLocalIp(), ",", GetPeerIp(), ",", rtt));
}

void PingConnection::AddFailedResult(std::string error) {
  // Remove commas from the error message, because the results will be written
  // to a CSV file.
  error.erase(std::remove(error.begin(), error.end(), ','), error.end());
  std::string timestamp = absl::FormatTime(op_start_, absl::UTCTimeZone());
  results_.push_back(
      absl::StrCat(timestamp, ",", GetLocalIp(), ",", GetPeerIp(), ",", error));
}

std::vector<std::string> PingConnection::GetAndClearResults() {
  return std::exchange(results_, {});
}

absl::StatusOr<bool> PingConnection::VerifyAndResetRecvBuffer(
    uint64_t recv_size) {
  ASSIGN_OR_RETURN(void* expected_buf,
                   payload_verification_->mem_helper->GetMem(
                       payload_verification_->send_id));
  ASSIGN_OR_RETURN(void* received_buf,
                   payload_verification_->mem_helper->GetMem(
                       payload_verification_->recv_id));
  ASSIGN_OR_RETURN(bool match,
                   payload_verification_->mem_validator->MemCmp(
                       received_buf, expected_buf, conn_.GetPayloadSize()));
  if (!match) {
    return absl::InternalError("Recv payload does not match expected payload.");
  }
  ResetRecvBuffer();
  return true;
}

void PingConnection::ResetRecvBuffer() {
  auto payload = std::make_unique<char[]>(conn_.GetPayloadSize());
  absl::Status status = payload_verification_->mem_helper->WriteBuffer(
      payload_verification_->recv_id, static_cast<const void*>(payload.get()),
      0, conn_.GetPayloadSize());
  if (!status.ok()) {
    LOG(WARNING)
        << "Failed to reset payload payload verification receive buffer: "
        << status;
  }
}

constexpr absl::Duration kPingBlockingMs = absl::Milliseconds(10);

void PingConnection::PrepareNewPingRound() {
  if (state_ == State::WAIT_FOR_NEXT_PROBE_ROUND) {
    state_ = State::START_NEW_PROBE_ROUND;
  }
}

void PingConnection::SendPingAndRecordResult() {
  absl::Time deadline = absl::Now() + kPingBlockingMs;
  switch (state_) {
    case State::WAIT_FOR_NEXT_PROBE_ROUND: {
      return;
    }
    case State::START_NEW_PROBE_ROUND: {
      state_ = State::START_SEND;
      op_start_ = absl::Now();
      ABSL_FALLTHROUGH_INTENDED;
    }
    case State::START_SEND: {
      absl::Status status = conn_.CreateSendOp();
      if (!status.ok()) {
        LOG(ERROR) << "PingConnection failed to create send op: " << status;
        AddFailedResult(status.ToString());
        state_ = State::WAIT_FOR_NEXT_PROBE_ROUND;
        return;
      }
      state_ = State::WAIT_FOR_SEND_RESPONSE;
      ABSL_FALLTHROUGH_INTENDED;
    }
    case State::WAIT_FOR_SEND_RESPONSE: {
      absl::StatusOr<bool> status = TestSendBlocking(deadline);
      if (!status.ok()) {
        AddFailedResult(status.status().ToString());
        state_ = State::WAIT_FOR_NEXT_PROBE_ROUND;
        return;
      }
      if (!status.value()) {
        return;
      }
      state_ = State::START_RECV;
      ABSL_FALLTHROUGH_INTENDED;
    }
    case State::START_RECV: {
      absl::Status status = conn_.CreateRecvOp();
      if (!status.ok()) {
        LOG(ERROR) << "PingConnection failed to create send op: " << status;
        AddFailedResult(status.ToString());
        state_ = State::WAIT_FOR_NEXT_PROBE_ROUND;
        return;
      }
      state_ = State::WAIT_FOR_RECV_RESPONSE;
      ABSL_FALLTHROUGH_INTENDED;
    }
    case State::WAIT_FOR_RECV_RESPONSE: {
      absl::StatusOr<bool> status = DoRecvBlocking(deadline);
      if (!status.ok()) {
        AddFailedResult(status.status().ToString());
        state_ = State::WAIT_FOR_NEXT_PROBE_ROUND;
        return;
      }
      if (!status.value()) {
        return;
      }
      AddResult();
      state_ = State::WAIT_FOR_NEXT_PROBE_ROUND;
      return;
    }
  }
}

PongConnection::PongConnection(Connection conn) : conn_(std::move(conn)) {
  state_ = State::START_RECV;
};

// DoPongOp is a non-blocking operations that checks if there is a message
// from the remote DXS server and then sends a response.
void PongConnection::DoPongOp() {
  switch (state_) {
    case State::START_RECV: {
      absl::Status status = conn_.CreateRecvOp();
      if (!status.ok()) {
        LOG(ERROR) << "PongConnection failed to create send op: " << status;
        return;
      }
      state_ = State::WAIT_FOR_RECV_RESPONSE;
      ABSL_FALLTHROUGH_INTENDED;
    }
    case State::WAIT_FOR_RECV_RESPONSE: {
      absl::StatusOr<bool> status = conn_.TestRecv();
      if (!status.ok()) {
        state_ = State::START_RECV;
        return;
      }
      if (!status.value()) {
        return;
      }
      state_ = State::START_SEND;
      ABSL_FALLTHROUGH_INTENDED;
    }
    case State::START_SEND: {
      absl::Status status = conn_.CreateSendOp();
      if (!status.ok()) {
        LOG(ERROR) << "PongConnection failed to create send op: " << status;
        state_ = State::START_RECV;
        return;
      }
      state_ = State::WAIT_FOR_SEND_RESPONSE;
      ABSL_FALLTHROUGH_INTENDED;
    }
    case State::WAIT_FOR_SEND_RESPONSE: {
      absl::StatusOr<bool> status = conn_.TestSend();
      if (!status.ok()) {
        state_ = State::START_RECV;
        return;
      }
      if (!status.value()) {
        return;
      }
      state_ = State::START_RECV;
      return;
    }
  }
}

}  // namespace dxs::prober

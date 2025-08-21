/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/sctp-handler.h"

#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "api/array_view.h"
#include "api/task_queue/task_queue_base.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/guest-clock.h"
#include "dxs/client/thread-shim.h"
#include "dxs/sctp-timeout-queue.h"
#include "net/dcsctp/public/dcsctp_message.h"
#include "net/dcsctp/public/dcsctp_options.h"
#include "net/dcsctp/public/dcsctp_socket.h"
#include "net/dcsctp/public/dcsctp_socket_factory.h"
#include "net/dcsctp/public/timeout.h"
#include "net/dcsctp/public/types.h"

ABSL_FLAG(int, dxs_sctp_max_retransmissions, 60,
          "Max retransmissions for SCTP connection. Each one adds 1s to the "
          "timeout.");

ABSL_FLAG(int, dxs_socket_mtu_testonly, 0,
          "If > 0, use this MTU for the socket.");

namespace dxs {
constexpr size_t kMaxMessageSize = 10 * 1024;

SctpHandler::SctpHandler(
    ControlMessageReceiverInterface& receiver, std::string_view id,
    int socket_fd, uint16_t local_port, uint16_t remote_port, Mode mode,
    std::unique_ptr<dcsctp::DcSctpSocketFactory> socket_factory)
    : receiver_do_not_access_(receiver),
      id_(id),
      socket_fd_(socket_fd),
      local_port_(local_port),
      remote_port_(remote_port),
      mode_(mode),
      socket_factory_(std::move(socket_factory)) {}

bool SctpHandler::Init() {
  absl::MutexLock lock(&mu_);
  int recv_buffer_size = 5 * 1024 * 1024;
  if (setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &recv_buffer_size,
                 sizeof(recv_buffer_size)) < 0) {
    PLOG(ERROR) << "Could not set receive buffer size.";
    return false;
  }

  mtu_ = absl::GetFlag(FLAGS_dxs_socket_mtu_testonly);
  if (mtu_ == 0) {
    socklen_t mtu_len = sizeof(mtu_);
    if (getsockopt(socket_fd_, IPPROTO_IP, IP_MTU, &mtu_, &mtu_len) < 0) {
      PLOG(ERROR) << "Could not get socket MTU.";
      return false;
    }
  }

  recv_buffer_.resize(mtu_);

  dcsctp::DcSctpOptions options;
  options.local_port = local_port_;
  options.remote_port = remote_port_;
  options.max_send_buffer_size = std::numeric_limits<size_t>::max();
  options.max_message_size = kMaxMessageSize;

  options.delayed_ack_max_timeout = dcsctp::DurationMs(1);
  options.rto_min = dcsctp::DurationMs(2);
  options.rto_max = dcsctp::DurationMs(1000);
  options.rto_initial = dcsctp::DurationMs(2);
  options.min_rtt_variance = dcsctp::DurationMs(20);
  // Sctp connection will close if no response from DXS for max_retransmissions
  // * 1s.
  options.max_retransmissions =
      absl::GetFlag(FLAGS_dxs_sctp_max_retransmissions);
  options.heartbeat_interval = dcsctp::DurationMs(1000);  // 1s

  options.max_burst = std::numeric_limits<int>::max();
  // Disable, but don't overflow on multiplication with options.mtus
  options.avoid_fragmentation_cwnd_mtus = 1 << 16;
  options.mtu = mtu_ - sizeof(ip6_hdr) - sizeof(udphdr);

  timeout_queue_ =
      std::make_unique<SctpTimeoutQueue>(*this, GlobalGuestClock::GetClock());
  sctp_socket_ = socket_factory_->Create(id_, *this, nullptr, options);
  if (mode_ == Mode::kClient) {
    sctp_socket_->Connect();
  }

  // Start the handler thread.
  running_ = true;
  handler_thread_ = NewThreadShim([this] { RunSctpHandler(); }, "SctpHandler");
  return true;
}

absl::Status SctpHandler::Shutdown(absl::Duration timeout) {
  absl::MutexLock lock(&mu_);
  if (running_) {
    shutting_down_ = true;
    auto disconnected = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return !connected_;
    };
    if (!mu_.AwaitWithTimeout(absl::Condition(&disconnected), timeout)) {
      running_ = false;
      return absl::DeadlineExceededError(
          "SCTP handler failed to shut down within timeout.");
    }
  }

  return absl::OkStatus();
}

SctpHandler::~SctpHandler() {
  if (absl::Status shutdown_result = Shutdown(/*timeout=*/absl::Seconds(1));
      !shutdown_result.ok()) {
    LOG(ERROR) << shutdown_result.message();
  }
  handler_thread_.reset();
  if (close(socket_fd_) != 0) {
    PLOG(ERROR) << absl::StrFormat(
        "close failed while destructing SctpHandler");
  }
}

void SctpHandler::RunSctpHandler() {
  bool clean_shutdown = false;
  auto cleanup = absl::MakeCleanup([&] {
    if (!clean_shutdown) {
      Receiver().OnControlChannelFailure();
    }
  });

  while (true) {
    // Granularity of timers is 1ms, just sleep for 1ms to avoid unnecessarily
    // pulling data off the socket.
    absl::SleepFor(absl::Milliseconds(1));
    absl::MutexLock lock(&mu_);

    // First check if we should exit.
    if (!running_) {
      return;
    }

    if (shutting_down_) {
      // Consume event
      shutting_down_ = false;
      clean_shutdown = true;
      sctp_socket_->Shutdown();
      LOG(INFO) << "SCTP handler shutdown signal received";
    }

    // Socket needs to be read.
    ReadSocket();

    if (!connected_) {
      LOG(INFO) << "Socket closed, handler exiting.";
      return;
    }

    timeout_queue_->Run();
  }
}

/*** Client <-> Queue Methods ***/

// Sends the specified message. Returns 0 on success, negative on failure.  This
// method is thread-safe.
absl::Status SctpHandler::SendControlMessage(absl::Span<const uint8_t> buffer) {
  if (buffer.size() > kMaxMessageSize) {
    return absl::InvalidArgumentError("Message too large.");
  }

  if (buffer.empty()) {
    return absl::InvalidArgumentError("Zero-byte payloads are not allowed.");
  }

  absl::MutexLock lock(&mu_);
  if (!connected_) {
    return absl::FailedPreconditionError("SCTP handler is not connected.");
  }
  dcsctp::SendStatus status = sctp_socket_->Send(
      dcsctp::DcSctpMessage(dcsctp::StreamID(0), dcsctp::PPID(0),
                            {buffer.begin(), buffer.end()}),
      {});
  if (status != dcsctp::SendStatus::kSuccess) {
    LOG(ERROR) << absl::StrFormat("SCTP send invalid %d.", status);
    sctp_socket_->Close();
    connected_ = false;
    return absl::InternalError(
        absl::StrCat("SCTP send invalid: ", ToString(status)));
  }
  return absl::OkStatus();
}

void SctpHandler::OnMessageReceived(dcsctp::DcSctpMessage message) {
  mu_.AssertHeld();
  receive_buffer_.push_back(std::move(message));
}

/*** UDP Socket Methods ***/

// Reads available data from the UDP socket and pushes it to sctp_socket_.
void SctpHandler::ReadSocket() {
  ::iovec iov{.iov_base = recv_buffer_.data(), .iov_len = recv_buffer_.size()};
  msghdr msg_hdr{.msg_name = nullptr,
                 .msg_namelen = 0,
                 .msg_iov = &iov,
                 .msg_iovlen = 1,
                 .msg_control = nullptr,
                 .msg_controllen = 0,
                 .msg_flags = 0};
  while (true) {
    msg_hdr.msg_flags = 0;
    int result = recvmsg(socket_fd_, &msg_hdr, MSG_DONTWAIT);

    if (result < 0) {
      if (errno == EAGAIN || errno == EINTR) {
        // Nothing left to read.
        break;
      }
      PLOG(ERROR) << "Error reading from socket";
      break;
    }

    if (result == 0) {
      break;
    }

    if ((msg_hdr.msg_flags & MSG_TRUNC) != 0) {
      LOG(ERROR) << absl::StrFormat(
          "Received control message larger than internal buffer. fd: %d",
          socket_fd_);
      break;
    }
    sctp_socket_->ReceivePacket(
        {recv_buffer_.data(), static_cast<size_t>(result)});
  }
  if (active_receive_) return;  // Existing receiver will send message
  active_receive_ = true;
  while (!receive_buffer_.empty()) {
    auto received = std::exchange(receive_buffer_, {});
    mu_.Unlock();
    for (const auto& message : received) {
      Receiver().ReceiveControlMessage(message.payload());
    }
    mu_.Lock();
  }
  active_receive_ = false;
}

// DcSctpCallback - Sends the data over the UDP socket.
dcsctp::SendPacketStatus SctpHandler::SendPacketWithStatus(
    webrtc::ArrayView<const uint8_t> data) {
  mu_.AssertHeld();
  if (data.size() > static_cast<size_t>(mtu_)) {
    LOG(ERROR) << absl::StrFormat(
        "tried to send UDP packet with %d bytes, which is larger than the "
        "socket MTU %d",
        data.size(), mtu_);
    return dcsctp::SendPacketStatus::kError;
  }
  const auto result = send(socket_fd_, data.data(), data.size(), 0);
  if (std::cmp_not_equal(result, data.size())) {
    PLOG(ERROR) << "send failed";
    return dcsctp::SendPacketStatus::kError;
  }
  return dcsctp::SendPacketStatus::kSuccess;
}

/*** Timeout Methods ***/

std::unique_ptr<dcsctp::Timeout> SctpHandler::CreateTimeout(
    webrtc::TaskQueueBase::DelayPrecision precision) {
  mu_.AssertHeld();
  return timeout_queue_->CreateTimeout();
}

}  // namespace dxs

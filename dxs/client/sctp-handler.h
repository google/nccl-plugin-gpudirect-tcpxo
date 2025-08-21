/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_SCTP_HANDLER_H_
#define DXS_CLIENT_SCTP_HANDLER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "api/array_view.h"
#include "api/task_queue/task_queue_base.h"
#include "api/units/timestamp.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/thread-shim.h"
#include "dxs/sctp-timeout-queue.h"
#include "net/dcsctp/public/dcsctp_message.h"
#include "net/dcsctp/public/dcsctp_socket.h"
#include "net/dcsctp/public/dcsctp_socket_factory.h"
#include "net/dcsctp/public/timeout.h"
#include "net/dcsctp/public/types.h"

namespace dxs {
// A class for communicating via SCTP over UDP.  This provides a reliable
// transport over an unreliable one.
class SctpHandler : public ControlMessageHandlerInterface,
                    public SctpTimeoutHandlerInterface,
                    private dcsctp::DcSctpSocketCallbacks {
 public:
  enum class Mode {
    kClient,
    kServer,
  };

  // Note SctpHandler closes socket_fd upon destruction.
  SctpHandler(
      ABSL_ATTRIBUTE_LIFETIME_BOUND ControlMessageReceiverInterface& receiver,
      absl::string_view id, int socket_fd, uint16_t local_port,
      uint16_t remote_port, Mode mode = Mode::kClient,
      std::unique_ptr<dcsctp::DcSctpSocketFactory> socket_factory =
          std::make_unique<dcsctp::DcSctpSocketFactory>());

  ~SctpHandler() override;

  // Initializes the handler - returns true on success.
  bool Init();

  // Shutdown SCTP handler gracefully.
  // A error indicates server may not received the shutdown message.
  absl::Status Shutdown(absl::Duration timeout)
      ABSL_LOCKS_EXCLUDED(mu_) override;

  // See ControlMessageHandlerInterface.
  absl::Status SendControlMessage(absl::Span<const uint8_t> buffer) override;
  void RxPoll() override {
    absl::MutexLock lock(&mu_);
    ReadSocket();
  };

  void HandleTimeout(dcsctp::TimeoutID timeout_id) override {
    mu_.AssertHeld();
    sctp_socket_->HandleTimeout(timeout_id);
  }

  bool HealthCheck() const override {
    absl::MutexLock lock(&mu_);
    return connected_;
  }

 private:
  void RunSctpHandler();

  void ReadSocket() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // DcSctpSocketCallbacks
  dcsctp::SendPacketStatus SendPacketWithStatus(
      webrtc::ArrayView<const uint8_t> data) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::unique_ptr<dcsctp::Timeout> CreateTimeout(
      webrtc::TaskQueueBase::DelayPrecision precision) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  webrtc::Timestamp Now() override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    mu_.AssertHeld();
    return timeout_queue_->GetTime();
  }

  uint32_t GetRandomInt(uint32_t low, uint32_t high) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    mu_.AssertHeld();
    return absl::Uniform(bitgen_, low, high);
  };

  void OnMessageReceived(dcsctp::DcSctpMessage message) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void OnError(dcsctp::ErrorKind error, absl::string_view message) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    LOG(WARNING) << absl::StrFormat("SCTP Non-Fatal Error: %s, %s",
                                    std::string(ToString(error)).c_str(),
                                    std::string(message).c_str());
  };
  void OnAborted(dcsctp::ErrorKind error, absl::string_view message) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    LOG(WARNING) << absl::StrFormat("SCTP Aborted: %s, %s",
                                    std::string(ToString(error)).c_str(),
                                    std::string(message).c_str());
    connected_ = false;
  };
  void OnConnected() override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    LOG(INFO) << "SCTP Connected";
  };
  void OnClosed() override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    LOG(INFO) << "SCTP Closed";
    connected_ = false;
  };
  void OnConnectionRestarted() override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    mu_.AssertHeld();
    LOG(INFO) << "SCTP Connection Restarted";
    // There could have been packet loss as a result of restarting the
    // association.  Close the socket to be safe.
    sctp_socket_->Close();
    connected_ = false;
  };
  void OnStreamsResetFailed(
      webrtc::ArrayView<const dcsctp::StreamID> outgoing_streams,
      absl::string_view reason) override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    LOG(WARNING) << absl::StrFormat("SCTP Streams Reset Failed: %s",
                                    std::string(reason).c_str());
  };
  void OnStreamsResetPerformed(
      webrtc::ArrayView<const dcsctp::StreamID> outgoing_streams) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    LOG(INFO) << "SCTP Streams Reset Performed";
  };
  void OnIncomingStreamsReset(
      webrtc::ArrayView<const dcsctp::StreamID> incoming_streams) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    LOG(INFO) << "SCTP Incoming Streams Reset";
  };
  ControlMessageReceiverInterface& Receiver() ABSL_LOCKS_EXCLUDED(mu_) {
    return receiver_do_not_access_;
  }
  ControlMessageReceiverInterface& receiver_do_not_access_;
  std::string id_;
  absl::BitGen bitgen_ ABSL_GUARDED_BY(mu_);

  mutable absl::Mutex mu_;

  bool running_ ABSL_GUARDED_BY(mu_) = false;
  bool shutting_down_ ABSL_GUARDED_BY(mu_) = false;
  bool connected_ ABSL_GUARDED_BY(mu_) = true;
  std::unique_ptr<ThreadShim> handler_thread_;
  std::vector<uint8_t> recv_buffer_ ABSL_GUARDED_BY(mu_);
  int socket_fd_ ABSL_GUARDED_BY(mu_);
  int mtu_;

  // Whether there is a thread currently calling Receive.
  // ControlMessageReceiverInterface implementations must be thread safe, but we
  // need to ensure order is maintained.
  bool active_receive_ ABSL_GUARDED_BY(mu_) = false;
  // Buffer of messages from racing threads. Must be drained before setting
  // active_receive_ = false.
  std::vector<dcsctp::DcSctpMessage> receive_buffer_ ABSL_GUARDED_BY(mu_);

  uint16_t local_port_;
  uint16_t remote_port_;
  Mode mode_;
  std::unique_ptr<dcsctp::DcSctpSocketFactory> socket_factory_;
  std::unique_ptr<SctpTimeoutQueue> timeout_queue_ ABSL_GUARDED_BY(mu_);
  // Must be after timeout_queue_, as it hold references to timeouts.
  std::unique_ptr<dcsctp::DcSctpSocketInterface> sctp_socket_
      ABSL_GUARDED_BY(mu_);

  friend class SctpHandlerTest;
};

}  // namespace dxs

#endif  // DXS_CLIENT_SCTP_HANDLER_H_

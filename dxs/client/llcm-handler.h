/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_LLCM_HANDLER_H_
#define DXS_CLIENT_LLCM_HANDLER_H_

#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/spsc_queue/spsc_messaging_queue_pair.h"

namespace dxs {

class LlcmHandler : public ControlMessageHandlerInterface {
 public:
  // Have the constructor be same as SctpHandler and SocketHandler
  explicit LlcmHandler(
      ABSL_ATTRIBUTE_LIFETIME_BOUND ControlMessageReceiverInterface& receiver,
      std::unique_ptr<ControlMessageHandlerInterface> slow_handler)
      : receiver_(receiver), slow_handler_(std::move(slow_handler)) {}

  ~LlcmHandler() override = default;

  absl::Status Init(absl::Span<volatile uint8_t> local_memory,
                    absl::Span<volatile uint8_t> remote_memory);

  absl::Status Shutdown(absl::Duration timeout)
      ABSL_LOCKS_EXCLUDED(mutex_) override {
    absl::MutexLock lock(&mutex_);
    return slow_handler_->Shutdown(timeout);
  }

  absl::Status SendControlMessage(absl::Span<const uint8_t> buffer) override;
  void RxPoll() override;

  // Must be called regularly. it will call receiver_.ReceiveControlMessage() or
  // OnControlChannelFailure() internally. Returns true if receiver_ is called,
  // otherwise returns false.
  //
  // Exposed for test only.
  bool LlcmRecvTest();

 private:
  absl::Status LlcmSend(absl::Span<const uint8_t> buffer)
      ABSL_LOCKS_EXCLUDED(mutex_);
  bool LlcmRecv() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  absl::Status LlcmSendLockHeld(absl::Span<const uint8_t> buffer)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  absl::StatusCode TrySendLlcmOverflow() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  absl::Mutex mutex_;
  ControlMessageReceiverInterface& receiver_;
  std::unique_ptr<ControlMessageHandlerInterface> slow_handler_;
  std::unique_ptr<SpscMessagingQueuePair> llcmq_;
  std::queue<std::vector<uint8_t>> overflow_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_LLCM_HANDLER_H_

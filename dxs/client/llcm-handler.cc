/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/llcm-handler.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "dxs/client/control-command.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/oss/status_macros.h"
#include "dxs/client/spsc_queue/spsc_messaging_queue_pair.h"

namespace dxs {

absl::Status LlcmHandler::Init(absl::Span<volatile uint8_t> local_memory,
                               absl::Span<volatile uint8_t> remote_memory) {
  ASSIGN_OR_RETURN(llcmq_,
                   SpscMessagingQueuePair::Create(local_memory, remote_memory));
  return absl::OkStatus();
}

absl::Status LlcmHandler::SendControlMessage(absl::Span<const uint8_t> buffer) {
  // If slow_handler_ is not set, just send everything over LLCM.
  if (!slow_handler_) {
    return LlcmSend(buffer);
  }

  ControlCommand command = static_cast<ControlCommand>(buffer[0]);
  switch (command) {
    case ControlCommand::kSend:
    case ControlCommand::kSendAck:
    case ControlCommand::kRecv:
    case ControlCommand::kRecvAck:
    case ControlCommand::kRelease:
    case ControlCommand::kPing:
    case ControlCommand::kPong:
      return LlcmSend(buffer);
    default:
      return slow_handler_->SendControlMessage(buffer);
  }
}

void LlcmHandler::RxPoll() {
  absl::MutexLock lock(&mutex_);
  absl::StatusCode code = TrySendLlcmOverflow();
  if (code != absl::StatusCode::kOk &&
      code != absl::StatusCode::kResourceExhausted) {
    LOG(ERROR) << "Unknown LLCM Send error: " << absl::StatusCodeToString(code);
    receiver_.OnControlChannelFailure();
    return;
  }

  // TCP sends several IOVEC messages per Recv(). Consider changing this
  // back to 32 when and if we implement the coalescing.
  constexpr int kBatchSize = 256;
  for (int i = 0; i < kBatchSize; ++i) {
    if (!LlcmRecv()) break;
  }
}

bool LlcmHandler::LlcmRecvTest() {
  absl::MutexLock lock(&mutex_);
  return LlcmRecv();
}

bool LlcmHandler::LlcmRecv() {
  DCHECK(llcmq_ != nullptr);
  absl::StatusOr<std::string> recv = llcmq_->Receive();
  if (recv.status().code() == absl::StatusCode::kUnavailable) {
    return false;
  }
  if (recv.ok()) {
    auto buffer = absl::MakeSpan(
        reinterpret_cast<uint8_t*>(recv.value().data()), recv.value().size());

    // Check for kPing and directly send kPong here
    if (buffer.size() == sizeof(PingMessage) &&
        static_cast<ControlCommand>(buffer[0]) == ControlCommand::kPing) {
      auto* message = ValidateAndGetMessage<PingMessage>(buffer);
      if (message == nullptr) return true;

      PongMessage pong;
      pong.seq = message->seq;
      absl::Status result = LlcmSendLockHeld(absl::MakeSpan(
          reinterpret_cast<uint8_t*>(&pong), sizeof(PongMessage)));
      if (!result.ok()) {
        LOG(WARNING) << absl::StrFormat(
            "Failed to send pong msg for ping: %d, %s", pong.seq,
            result.ToString());
      }
    } else {
      receiver_.ReceiveControlMessage(buffer);
    }
  } else {
    receiver_.OnControlChannelFailure();
  }
  return true;
}

absl::Status LlcmHandler::LlcmSend(absl::Span<const uint8_t> buffer) {
  absl::MutexLock lock(&mutex_);
  DCHECK(llcmq_ != nullptr);

  return LlcmSendLockHeld(buffer);
}

absl::Status LlcmHandler::LlcmSendLockHeld(absl::Span<const uint8_t> buffer) {
  // Send overflow first.
  absl::StatusCode code = TrySendLlcmOverflow();
  if (code == absl::StatusCode::kResourceExhausted) {
    overflow_.emplace(buffer.begin(), buffer.end());
    return absl::OkStatus();
  } else if (code != absl::StatusCode::kOk) {
    return absl::Status(code, "");
  }

  // Send message.
  code = llcmq_->Send(buffer);
  if (code == absl::StatusCode::kResourceExhausted) {
    overflow_.emplace(buffer.begin(), buffer.end());
    return absl::OkStatus();
  } else if (code != absl::StatusCode::kOk) {
    return absl::Status(code, "");
  }

  return absl::OkStatus();
}

absl::StatusCode LlcmHandler::TrySendLlcmOverflow() {
  while (!overflow_.empty()) {
    const auto& message = overflow_.front();
    absl::StatusCode code = llcmq_->Send(message);
    if (code != absl::StatusCode::kOk) return code;
    overflow_.pop();
  }
  return absl::StatusCode::kOk;
}

}  // namespace dxs

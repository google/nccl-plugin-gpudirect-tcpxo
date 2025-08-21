/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_SEND_OP_H_
#define DXS_CLIENT_SEND_OP_H_

#include <atomic>
#include <optional>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "dxs/client/control-command.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/monotonic-timestamp.h"

namespace dxs {

typedef struct {
  OpStatus status;
  SendAckMessage::Status req_status;
} SendOpResult;

static_assert(std::is_trivially_copyable<SendOpResult>::value,
              "SendOpResult must be trivially copyable.");

class SendOp : public SendOpInterface {
 public:
  // Shared state between SendOp and its owner.
  struct SharedState {
    // The current op state. RecvOp can only access the completion time
    // once status has been set to kComplete using a release store.
    std::atomic<SendOpResult> result{
        {OpStatus::kPending, SendAckMessage::Status::kOk}};
    MonotonicTs completion_time{absl::InfinitePast()};
  };

  explicit SendOp(
      ABSL_ATTRIBUTE_LIFETIME_BOUND ControlMessageHandlerInterface& client,
      ABSL_ATTRIBUTE_LIFETIME_BOUND SharedState& state, dxs::OpId op_id)
      : client_(client), state_(state), op_id_(op_id) {}
  ~SendOp() override = default;

  std::optional<absl::Status> Test() override {
    // Check for and handle completions.
    client_.RxPoll();

    // Protects completion_time; user can only rely on
    // completion time if Test==kComplete
    SendOpResult result = state_.result.load(std::memory_order_acquire);
    switch (result.status) {
      case OpStatus::kPending:
        return std::nullopt;
      case OpStatus::kComplete:
        return absl::OkStatus();
      case OpStatus::kError:
        if (result.req_status == SendAckMessage::Status::kOk) {
          return absl::InternalError("SendOp failed.");
        }

        return absl::InternalError(absl::StrCat(
            "SendOp failed: ", GetDetailedStatus(result.req_status)));
    }
    LOG(ERROR) << "Unexpected SendOp status: "
               << static_cast<int>(result.status);
    return absl::InternalError(
        absl::StrCat("Unexpected SendOp status: ", result.status));
  }

  OpId GetOpId() const override { return op_id_; }

  MonotonicTs GetCompletionTime() const override {
    return state_.completion_time;
  }

 private:
  ControlMessageHandlerInterface& client_;
  SharedState& state_;
  const OpId op_id_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_SEND_OP_H_

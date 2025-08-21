/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_LINEARIZED_RECV_OP_H_
#define DXS_CLIENT_LINEARIZED_RECV_OP_H_

#include <atomic>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
  RecvAckMessage::Status req_status;
} RecvOpResult;

static_assert(std::is_trivially_copyable<RecvOpResult>::value,
              "RecvOpResult must be trivially copyable");

class LinearizedRecvOp : public LinearizedRecvOpInterface {
 public:
  // Shared state between RecvOp and its owner.
  struct SharedState {
    // The current op status. RecvOp can only access struct members once status
    // has been set to kComplete using a release store.
    std::atomic<RecvOpResult> result{
        {OpStatus::kPending, RecvAckMessage::Status::kOk}};
    uint64_t size{0};
    MonotonicTs completion_time{absl::InfinitePast()};
  };

  LinearizedRecvOp(
      ABSL_ATTRIBUTE_LIFETIME_BOUND ControlMessageHandlerInterface& client,
      ABSL_ATTRIBUTE_LIFETIME_BOUND SharedState& state,
      DataSocketHandle data_sock, OpId op_id)
      : client_(client), state_(state), data_sock_(data_sock), op_id_(op_id) {}
  std::optional<absl::StatusOr<uint64_t>> Test() override {
    // Check for and handle completions.
    client_.RxPoll();

    // Use memory_order_acquire so that changes to state_ are visible to this
    // thread in the completed case.
    RecvOpResult result = state_.result.load(std::memory_order_acquire);
    switch (result.status) {
      case OpStatus::kPending:
        return std::nullopt;
      case OpStatus::kComplete:
        return state_.size;
      case OpStatus::kError:
        if (result.req_status == RecvAckMessage::Status::kOk) {
          return absl::InternalError("LinearizedRecvOp failed.");
        }

        return absl::InternalError(absl::StrCat(
            "LinearizedRecvOp failed: ", GetDetailedStatus(result.req_status)));
    }
    return absl::InternalError(
        absl::StrCat("Unexpected LinearizedRecvOp status: ", result.status));
  }

  OpId GetOpId() const override { return op_id_; }

  absl::Time GetCompletionTime() const override {
    return state_.completion_time;
  }

 private:
  ControlMessageHandlerInterface& client_;
  SharedState& state_;
  const DataSocketHandle data_sock_;
  const OpId op_id_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_LINEARIZED_RECV_OP_H_

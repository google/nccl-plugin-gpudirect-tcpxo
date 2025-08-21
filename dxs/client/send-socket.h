/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_SEND_SOCKET_H_
#define DXS_CLIENT_SEND_SOCKET_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "dxs/client/control-command.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/data-sock.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h"
#include "dxs/client/relaxed-atomic.h"
#include "dxs/client/sequence-number.h"

namespace dxs {

class SendSocket : public SendSocketInterface {
 public:
  SendSocket(
      ABSL_ATTRIBUTE_LIFETIME_BOUND ControlMessageHandlerInterface& handler,
      ABSL_ATTRIBUTE_LIFETIME_BOUND SocketRegistryInterface& registry,
      ABSL_ATTRIBUTE_LIFETIME_BOUND SequenceNumber& next_op_id,
      ABSL_ATTRIBUTE_LIFETIME_BOUND RelaxedAtomic<DataSockStatus>& status,
      DataSocketHandle handle, std::string nic_addr, WireSocketAddr peer,
      bool close_on_teardown)
      : handler_(handler),
        registry_(registry),
        next_op_id_(next_op_id),
        status_(status),
        handle_(handle),
        nic_addr_(std::move(nic_addr)),
        peer_(std::move(peer)),
        close_on_teardown_(close_on_teardown) {}
  ~SendSocket() override {
    if (close_on_teardown_) {
      // Tell DXS to close the socket.
      CloseDataSockMessage message{.sock = handle_};
      absl::Status result = handler_.SendMessage(&message);
      if (!result.ok()) {
        LOG(ERROR) << absl::StrFormat("Failed to close send socket: %s",
                                      result.ToString().c_str());
        assert(false);
      }
    }
  }

  std::optional<absl::Status> SocketReady() override {
    handler_.RxPoll();
    DataSockStatus status = status_.Load();
    switch (status) {
      case DataSockStatus::kPendingConnect:
        return std::nullopt;
      case DataSockStatus::kConnected:
        return absl::OkStatus();
      default:
        return absl::InternalError(
            absl::StrCat("SendSocket connect failed: ", ToString(status)));
    }
  }

  absl::StatusOr<std::unique_ptr<SendOpInterface>> Send(
      uint64_t offset, uint64_t size, Reg reg_handle) override {
    // In debug mode check 'data_sock' is valid.
    assert(status_.Load() == DataSockStatus::kConnected);

    OpId op_id{next_op_id_.Next()};
    SendMessage message{.sock = handle_,
                        .reg_handle = reg_handle,
                        .op_id = op_id,
                        .offset = offset,
                        .size = size};

    auto new_op = registry_.RegisterSendOp(op_id);
    RETURN_IF_ERROR(handler_.SendMessage(&message));
    return new_op;
  }
  std::string Address() const override { return nic_addr_; }

  WireSocketAddr Peer() const override { return peer_; }

 private:
  ControlMessageHandlerInterface& handler_;
  SocketRegistryInterface& registry_;
  SequenceNumber& next_op_id_;
  RelaxedAtomic<DataSockStatus>& status_;
  const DataSocketHandle handle_;
  const std::string nic_addr_;
  const WireSocketAddr peer_;
  const bool close_on_teardown_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_SEND_SOCKET_H_

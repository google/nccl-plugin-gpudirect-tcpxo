/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_LISTEN_SOCKET_H_
#define DXS_CLIENT_LISTEN_SOCKET_H_

#include <arpa/inet.h>
#include <sys/socket.h>

#include <cassert>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "dxs/client/control-command.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h"
#include "dxs/client/relaxed-atomic.h"
#include "dxs/client/sequence-number.h"

namespace dxs {

class ListenSocket : public ListenSocketInterface {
 public:
  class SharedState {
   public:
    RelaxedAtomic<int> port_{0};

    void AddPendingConnection(WireSocketAddr peer_saddr) {
      absl::MutexLock l(&mutex_);
      pending_connections_.push_back(WireSocketAddr(peer_saddr));
    }

    std::optional<WireSocketAddr> PopPendingConnection() {
      absl::MutexLock l(&mutex_);
      if (pending_connections_.empty()) {
        return std::nullopt;
      }

      auto result = pending_connections_.front();
      pending_connections_.pop_front();
      return result;
    }

   private:
    std::deque<WireSocketAddr> pending_connections_ ABSL_GUARDED_BY(mutex_);
    absl::Mutex mutex_;
  };

  ListenSocket(
      ABSL_ATTRIBUTE_LIFETIME_BOUND ControlMessageHandlerInterface& handler,
      ABSL_ATTRIBUTE_LIFETIME_BOUND SocketRegistryInterface& registry,
      ABSL_ATTRIBUTE_LIFETIME_BOUND SequenceNumber& next_data_socket_handle,
      ABSL_ATTRIBUTE_LIFETIME_BOUND SharedState& state,
      ListenSocketHandle handle, std::string nic_addr)
      : handler_(handler),
        registry_(registry),
        next_data_socket_handle_(next_data_socket_handle),
        state_(state),
        handle_(handle),
        nic_addr_(std::move(nic_addr)) {}

  ~ListenSocket() override {
    // Tell DXS to close the socket.
    CloseListenSockMessage message{.sock = handle_};
    absl::Status result = handler_.SendMessage(&message);
    if (!result.ok()) {
      LOG(ERROR) << absl::StrFormat("Failed to close listen socket: %s",
                                    result.ToString().c_str());
      assert(false);
    }
  }

  std::optional<absl::Status> SocketReady() override {
    handler_.RxPoll();
    switch (state_.port_.Load()) {
      case -1:
        return absl::InternalError("ListenSocket failed.");
      case 0:
        return std::nullopt;
      default:
        return absl::OkStatus();
    }
  }

  absl::StatusOr<absl_nullable std::unique_ptr<RecvSocketInterface>> Accept()
      override {
    if (state_.port_.Load() <= 0) {
      return absl::FailedPreconditionError(
          "Accept() called on socket that isn't listening.");
    }

    auto peer = state_.PopPendingConnection();
    if (!peer.has_value()) {
      return nullptr;
    }

    // There's a pending connection. Create a data socket for it.
    DataSocketHandle handle{next_data_socket_handle_.Next()};
    AcceptMessage message;
    message.listen_sock = handle_;
    message.data_sock = handle;

    auto sock = registry_.RegisterRecvSocket(handle, peer.value());
    RETURN_IF_ERROR(handler_.SendMessage(&message));
    return sock;
  }

  int Port() const override {
    int port = state_.port_.Load();
    assert(port > 0);
    return port;
  }

  std::string Address() const override { return nic_addr_; }

 private:
  ControlMessageHandlerInterface& handler_;
  SocketRegistryInterface& registry_;
  SequenceNumber& next_data_socket_handle_;
  SharedState& state_;
  const ListenSocketHandle handle_;
  const std::string nic_addr_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_LISTEN_SOCKET_H_

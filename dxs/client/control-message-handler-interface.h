/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_CONTROL_MESSAGE_HANDLER_INTERFACE_H_
#define DXS_CLIENT_CONTROL_MESSAGE_HANDLER_INTERFACE_H_

#include <netinet/ip6.h>
#include <netinet/udp.h>

#include <concepts>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/message.h"

class ControlMessageHandlerInterface {
 public:
  virtual ~ControlMessageHandlerInterface() = default;

  // Close connection with the server.
  virtual absl::Status Shutdown(absl::Duration timeout) = 0;

  // Sends the specified message.
  virtual absl::Status SendControlMessage(absl::Span<const uint8_t> buffer) = 0;

  // Performs a synchronous check for new messages from DXS. The handler
  // is still free process incoming messages asynchronously without calls
  // to RxPoll, but with potentially higher latency.
  virtual void RxPoll() = 0;

  // Sends the typed message.
  template <typename MessageT>
  absl::Status SendMessage(MessageT* message) {
    return SendControlMessage(
        absl::MakeSpan(reinterpret_cast<uint8_t*>(message), sizeof(MessageT)));
  }

  // Sends a typed message with a trailer.
  template <class MessageT,
            std::derived_from<google::protobuf::Message> PayloadT>
  absl::Status SendMessageWithPayload(const MessageT& message,
                                      const PayloadT& payload) {
    size_t payload_size = payload.ByteSizeLong();
    absl::FixedArray<uint8_t> msg_space(sizeof(MessageT) + payload_size);
    auto* msg = new (msg_space.data()) MessageT(message);
    msg->payload_size = static_cast<uint16_t>(payload_size);
    payload.SerializeToArray(msg->payload, payload_size);
    return SendControlMessage(msg_space);
  }

  virtual absl::StatusOr<std::string> GetPeerHostname() {
    return absl::UnimplementedError("GetPeerHostname not implemented");
  }

  virtual bool HealthCheck() const { return true; }
};

class ControlMessageReceiverInterface {
 public:
  virtual ~ControlMessageReceiverInterface() = default;
  // Receives the specified message. At most one outstanding call to this method
  // will be in flight, and the order of the calls will be preserved.
  virtual void ReceiveControlMessage(absl::Span<const uint8_t> buffer) = 0;
  // Notify the receiver that reading has failed and will not resume.
  virtual void OnControlChannelFailure() = 0;
};

#endif  // DXS_CLIENT_CONTROL_MESSAGE_HANDLER_INTERFACE_H_

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/control-command.h"

#include <arpa/inet.h>
#include <stdint.h>

#include <string>
#include <string_view>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "dxs/client/control-command.pb.h"
#include "dxs/client/dxs-client-types.h"

namespace dxs {

namespace {

template <typename ProtoT, typename MessageT>
std::string DumpPayloadProto(const MessageT& message) {
  absl::StatusOr<ProtoT> proto = GetPayloadProto<ProtoT>(message);
  if (!proto.ok()) return proto.status().ToString();
  return absl::StrCat(*proto);
}

template <typename MessageT>
std::string DumpStatusProto(const MessageT& msg) {
  return GetPayloadStatus(msg).ToString();
}

}  // namespace

std::string ToString(ControlCommand cmd) {
  switch (cmd) {
    case ControlCommand::kInvalid:
      return "kInvalid";
    case ControlCommand::kInit:
      return "kInit";
    case ControlCommand::kInitAck:
      return "kInitAck";
    case ControlCommand::kListen:
      return "kListen";
    case ControlCommand::kListenAck:
      return "kListenAck";
    case ControlCommand::kIncomingConnection:
      return "kIncomingConnection";
    case ControlCommand::kAccept:
      return "kAccept";
    case ControlCommand::kAcceptAck:
      return "kAcceptAck";
    case ControlCommand::kCloseListenSock:
      return "kCloseListenSock";
    case ControlCommand::kConnect:
      return "kConnect";
    case ControlCommand::kConnectAck:
      return "kConnectAck";
    case ControlCommand::kSend:
      return "kSend";
    case ControlCommand::kSendAck:
      return "kSendAck";
    case ControlCommand::kRecv:
      return "kRecv";
    case ControlCommand::kRecvAck:
      return "kRecvAck";
    case ControlCommand::kRelease:
      return "kRelease";
    case ControlCommand::kCloseDataSock:
      return "kCloseDataSock";
    case ControlCommand::kRegBuffer:
      return "kRegBuffer";
    case ControlCommand::kRegBufferAck:
      return "kRegBufferAck";
    case ControlCommand::kDeregBuffer:
      return "kDeregBuffer";
    case ControlCommand::kDeregBufferAck:
      return "kDeregBufferAck";
    case ControlCommand::kPing:
      return "kPing";
    case ControlCommand::kPong:
      return "kPong";
    case ControlCommand::kVersionedInit:
      return "kVersionedInit";
    case ControlCommand::kVersionedInitAck:
      return "kVersionedInitAck";
    case ControlCommand::kInitLlcm:
      return "kInitLlcm";
    case ControlCommand::kInitLlcmAck:
      return "kInitLlcmAck";
    case ControlCommand::kSubscribeToPeriodicStats:
      return "kSubscribeToPeriodicStats";
    case ControlCommand::kPeriodicStatsUpdate:
      return "kPeriodicStatsUpdate";
    default:
      return std::string("Unknown(" +
                         std::to_string(static_cast<uint8_t>(cmd)) + ")");
  }
}
std::string ToString(ControlCommand cmd, std::string_view delim) {
  return ToString(cmd) + std::string(delim);
}

std::string ToString(InitMessage::ClientType client_type) {
  switch (client_type) {
    case InitMessage::ClientType::kBufferManager:
      return "kBufferManager";
    case InitMessage::ClientType::kSnapClient:
      return "kSnapClient";
    default:
      return "Unknown(" + std::to_string(static_cast<uint8_t>(client_type)) +
             ")";
  }
}

std::string ToString(InitAckMessage::Status status) {
  switch (status) {
    case InitAckMessage::Status::kOk:
      return "kOk";
    case InitAckMessage::Status::kExists:
      return "kExists";
    case InitAckMessage::Status::kSecureInitRequired:
      return "kSecureInitRequired";
    case InitAckMessage::Status::kRejected:
      return "kRejected";
    case InitAckMessage::Status::kInvalidRequest:
      return "kInvalidRequest";
    case InitAckMessage::Status::kUnsupportedClientVersion:
      return "kUnsupportedClientVersion";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string ToString(InitLlcmAckMessage::Status status) {
  switch (status) {
    case InitLlcmAckMessage::Status::kOk:
      return "kOk";
    case InitLlcmAckMessage::Status::kFailed:
      return "kFailed";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string ToString(AcceptAckMessage::Status status) {
  switch (status) {
    case AcceptAckMessage::Status::kOk:
      return "kOk";
    case AcceptAckMessage::Status::kInvalidRequest:
      return "kInvalidRequest";
    case AcceptAckMessage::Status::kInvalidListenSocket:
      return "kInvalidListenSocket";
    case AcceptAckMessage::Status::kInternalError:
      return "kInternalError";
    case AcceptAckMessage::Status::kNoPendingConnection:
      return "kNoPendingConnection";
    case AcceptAckMessage::Status::kInvalidClientType:
      return "kInvalidClientType";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string ToString(RegBufferAckMessage::Status status) {
  switch (status) {
    case RegBufferAckMessage::Status::kOk:
      return "kOk";
    case RegBufferAckMessage::Status::kInvalidRegion:
      return "kInvalidRegion";
    case RegBufferAckMessage::Status::kOverlappingRegions:
      return "kOverlappingRegions";
    case RegBufferAckMessage::Status::kAlreadyRegistered:
      return "kAlreadyRegistered";
    case RegBufferAckMessage::Status::kMismatchedDirection:
      return "kMismatchedDirection";
    case RegBufferAckMessage::Status::kRegistrationFailed:
      return "kRegistrationFailed";
    case RegBufferAckMessage::Status::kNotAllowed:
      return "kNotAllowed";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string ToString(DeregBufferAckMessage::Status status) {
  switch (status) {
    case DeregBufferAckMessage::Status::kOk:
      return "kOk";
    case DeregBufferAckMessage::Status::kNotFound:
      return "kNotFound";
    case DeregBufferAckMessage::Status::kNotAllowed:
      return "kNotAllowed";
    case DeregBufferAckMessage::Status::kInternal:
      return "kInternal";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string ToString(OpStatus status) {
  switch (status) {
    case OpStatus::kComplete:
      return "kComplete";
    case OpStatus::kPending:
      return "kPending";
    case OpStatus::kError:
      return "kError";
    default:
      return "Unknown(" + std::to_string(static_cast<uint8_t>(status)) + ")";
  }
}

std::string ToString(absl::StatusCode code) {
  return absl::StatusCodeToString(code);
}

std::string ToString(ListenSocketHandle handle) {
  return std::to_string(
      static_cast<std::underlying_type_t<ListenSocketHandle>>(handle));
}

std::string ToString(DataSocketHandle handle) {
  return std::to_string(
      static_cast<std::underlying_type_t<DataSocketHandle>>(handle));
}

// ****** Format Helper Functions ******

std::string FormatCommand(ControlCommand cmd, std::string_view delim = "") {
  return "Command: " + ToString(cmd) + std::string(delim);
}

template <typename T>
std::string FormatFieldStd(std::string_view field_name, T value,
                           std::string_view delim = "") {
  return FormatFieldStd(field_name, std::to_string(value), delim);
}

template <>
std::string FormatFieldStd<std::string_view>(std::string_view field_name,
                                             std::string_view value,
                                             std::string_view delim) {
  return absl::StrCat(field_name, ": ", value, delim);
}

template <>
std::string FormatFieldStd<std::string>(std::string_view field_name,
                                        std::string value,
                                        std::string_view delim) {
  return absl::StrCat(field_name, ": ", value, delim);
}

template <>
std::string FormatFieldStd<bool>(std::string_view field_name, bool value,
                                 std::string_view delim) {
  return FormatFieldStd(field_name, std::string_view(value ? "True" : "False"),
                        delim);
}

template <typename T>
std::string FormatFieldVal(std::string_view field_name, T value,
                           std::string_view delim = "") {
  return FormatFieldStd(field_name, ToString(value), delim);
}

// ****** Message ToString() Functions ******

std::string ToString(const InitMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldVal("client_type", msg.client_type);
}

std::string ToString(const InitAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldVal("status", msg.status);
}

std::string ToString(const VersionedInitMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldStd("client_build_id",
                        std::string_view(msg.client_build_id), delim) +  //
         FormatFieldStd("client_version", msg.client_version);
}

std::string ToString(const VersionedInitAckMessage& msg,
                     std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldStd("server_build_id",
                        std::string_view(msg.server_build_id), delim) +  //
         FormatFieldStd("server_version", msg.server_version);
}

std::string ToString(const InitLlcmMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command);
}

std::string ToString(const InitLlcmAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +
         FormatFieldVal("status", msg.status, delim) +
         FormatFieldStd("a_queue_offset", msg.llcm_queue_offset, delim) +
         FormatFieldStd("a_queue_size", msg.llcm_queue_size, delim) +
         FormatFieldStd("ra_queue_offset", msg.reverse_llcm_queue_offset,
                        delim) +
         FormatFieldStd("ra_queue_size", msg.reverse_llcm_queue_size);
}

std::string ToString(const ListenMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldVal("sock", msg.sock);
}

std::string ToString(const ListenAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +        //
         FormatFieldVal("sock", msg.sock, delim) +  //
         FormatFieldStd("port", msg.port);
}

std::string ToString(const IncomingConnectionMessage& msg,
                     std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldVal("sock", msg.sock);
}

std::string ToString(const AcceptMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +                      //
         FormatFieldVal("listen_sock", msg.listen_sock, delim) +  //
         FormatFieldVal("data_sock", msg.data_sock);
}

std::string ToString(const AcceptAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +        //
         FormatFieldVal("sock", msg.sock, delim) +  //
         FormatFieldVal("status", msg.status);
}

std::string ToString(const CloseListenSockMessage& msg,
                     std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldVal("sock", msg.sock);
}

std::string ToString(const ConnectMessage& msg, std::string_view delim) {
  char buff6[INET6_ADDRSTRLEN];
  char buff4[INET_ADDRSTRLEN];
  const char* addr;
  if (msg.ipv6) {
    addr = inet_ntop(AF_INET6, msg.addr, buff6, sizeof(buff6));
  } else {
    addr = inet_ntop(AF_INET, msg.addr, buff4, sizeof(buff4));
  }

  return FormatCommand(msg.command, delim) +  //
         "addr: " +
         ((addr == nullptr) ? "[]" : ("[" + std::string(addr) + "]")) +
         std::string(delim) +                       //
         FormatFieldVal("sock", msg.sock, delim) +  //
         FormatFieldStd("port", msg.port, delim) +  //
         FormatFieldStd("ipv6", msg.ipv6);
}

std::string ToString(const ConnectAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +        //
         FormatFieldVal("sock", msg.sock, delim) +  //
         FormatFieldVal("status", msg.status);
}

std::string ToString(const SendMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +                    //
         FormatFieldVal("sock", msg.sock, delim) +              //
         FormatFieldStd("reg_handle", msg.reg_handle, delim) +  //
         FormatFieldStd("op_id", msg.op_id, delim) +            //
         FormatFieldStd("offset)", msg.offset, delim) +         //
         FormatFieldStd("size", msg.size);
}

std::string ToString(const SendAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +          //
         FormatFieldStd("op_id", msg.op_id, delim) +  //
         FormatFieldVal("status", msg.status);
}

std::string ToString(const RecvMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +                    //
         FormatFieldVal("sock", msg.sock, delim) +              //
         FormatFieldStd("offset", msg.offset, delim) +          //
         FormatFieldStd("size", msg.size, delim) +              //
         FormatFieldStd("reg_handle", msg.reg_handle, delim) +  //
         FormatFieldStd("op_id", msg.op_id);
}

std::string ToString(const RecvAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +            //
         FormatFieldStd("op_id", msg.op_id, delim) +    //
         FormatFieldVal("status", msg.status, delim) +  //
         FormatFieldStd("size", msg.size);
}

std::string ToString(const ReleaseMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +        //
         FormatFieldVal("sock", msg.sock, delim) +  //
         FormatFieldStd("op_id", msg.op_id);
}

std::string ToString(const CloseDataSockMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldVal("sock", msg.sock);
}

std::string ToString(const RegBufferMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +        //
         FormatFieldStd("more", msg.more, delim) +  //
         FormatFieldStd("num_gpas", msg.num_gpas);
}

std::string ToString(const RegBufferAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +                    //
         FormatFieldStd("reg_handle", msg.reg_handle, delim) +  //
         FormatFieldStd("num_gpas", msg.num_gpas, delim) +      //
         FormatFieldVal("status", msg.status);
}

std::string ToString(const DeregBufferMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +                    //
         FormatFieldStd("reg_handle", msg.reg_handle, delim) +  //
         FormatFieldStd("is_bounce_buffer", msg.is_bounce_buffer);
}

std::string ToString(const DeregBufferAckMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldVal("status", msg.status);
}

std::string ToString(const PingMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldStd("seq", msg.seq);
}

std::string ToString(const PongMessage& msg, std::string_view delim) {
  return FormatCommand(msg.command, delim) +  //
         FormatFieldStd("seq", msg.seq);
}

std::string ToString(const SubscribeToPeriodicStatsMessage& msg,
                     std::string_view delim) {
  return FormatCommand(msg.command, delim);
}

std::string ToString(const PeriodicStatsUpdateMessage& msg,
                     std::string_view delim) {
  return FormatCommand(msg.command, delim) +
         FormatFieldStd("payload", DumpPayloadProto<PeriodicStatsUpdate>(msg));
}

std::string GetDetailedStatus(InitAckMessage::Status status) {
  switch (status) {
    case InitAckMessage::Status::kOk:
      return "Everything is OK. DXS client should now be initialized.";
    case InitAckMessage::Status::kExists:
      return "The client is already initialized.";
    case InitAckMessage::Status::kSecureInitRequired:
      return "kSecureInitRequired";
    case InitAckMessage::Status::kRejected:
      // Question: Should we split this status in to two?
      return "Init request rejected (Possible reasons: (1) Client's source "
             "port doesn't match the buffer manager port that DXS is using or "
             "(2) Buffer manager client already exists).";
    case InitAckMessage::Status::kInvalidRequest:
      return "Init request is invalid (or not provided)";
    case InitAckMessage::Status::kUnsupportedClientVersion:
      return "Client version is no longer supported.";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string GetDetailedStatus(RegBufferAckMessage::Status status) {
  switch (status) {
    case RegBufferAckMessage::Status::kOk:
      return "Everything is OK. Buffer should now be registered.";
    case RegBufferAckMessage::Status::kInvalidRegion:
      return "Truncated packet or malicious client.";
    case RegBufferAckMessage::Status::kOverlappingRegions:
      return "Iovec overlaps existing pending buffer registration.";
    case RegBufferAckMessage::Status::kAlreadyRegistered:
      return "Buffer is already registered.";
    case RegBufferAckMessage::Status::kMismatchedDirection:
      return "Direction of buffer requested, isn't what DXS is expecting.";
    case RegBufferAckMessage::Status::kRegistrationFailed:
      return "Registration failed. Possibly, malformed (or empty) RegBuffer "
             "request.";
    case RegBufferAckMessage::Status::kNotAllowed:
      return "The client associated with this request isn't allowed to "
             "register the buffer. Please use this method with a buffer "
             "manager client.";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string GetDetailedStatus(DeregBufferAckMessage::Status status) {
  switch (status) {
    case DeregBufferAckMessage::Status::kOk:
      return "Everything is OK. Buffer should now be deregistered.";
    case DeregBufferAckMessage::Status::kNotFound:
      return "Dereg Buffer request is either malformed (or empty) or receive "
             "buffer reg handle not found.";
    case DeregBufferAckMessage::Status::kNotAllowed:
      return "The client associated with this request isn't allowed to "
             "deregister the buffer. Please use this method with a buffer "
             "manager client.";
    case DeregBufferAckMessage::Status::kInternal:
      return "Failed to uninstall LEM rules.";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string GetDetailedStatus(ListenAckMessage::Status status) {
  switch (status) {
    case ListenAckMessage::Status::kOk:
      return "Everything is OK.";
    case ListenAckMessage::Status::kInvalidRequest:
      return "Listen request is invalid (or not provided)";
    case ListenAckMessage::Status::kOutOfFreePorts:
      return "Free ports unavailable.";
    case ListenAckMessage::Status::kHandlerAlreadyInUse:
      return "Listen handler is already in use.";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

std::string GetDetailedStatus(AcceptAckMessage::Status status) {
  switch (status) {
    case AcceptAckMessage::Status::kOk:
      return "Everything is OK.";
    case AcceptAckMessage::Status::kInvalidRequest:
      return "Accept request is invalid (or not provided)";
    case AcceptAckMessage::Status::kInvalidListenSocket:
      return "Accept request had an invalid listen socket handle";
    case AcceptAckMessage::Status::kInternalError:
      return "Server error processing the accept request";
    case AcceptAckMessage::Status::kNoPendingConnection:
      return "Accept called when no connection was pending on the listening "
             "socket";
    case AcceptAckMessage::Status::kInvalidClientType:
      return "Accept request had an unknown client type";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(status)) + ")";
  }
}

/* Get a public error code and message for the given RecvAckMessage::Status. */
std::string GetDetailedStatus(SendAckMessage::Status req_status) {
  // All of the req_status values are negative, so we use 1 as the default
  // error code for catch alls.
  constexpr int kUnknownErrorCode = 1;
  int public_error_code = static_cast<int>(req_status);
  std::string message;

  switch (req_status) {
    case SendAckMessage::Status::kOk:
      message = "Everything is OK.";
      break;
    case SendAckMessage::Status::kInvalidRegHandle:
      message = "Invalid send buffer reg handle.";
      break;
    case SendAckMessage::Status::kInvalidRegisteredBuffer:
      message = "Invalid registered buffer.";
      break;
    case SendAckMessage::Status::kInvalidSocket:
      message = "Invalid socket.";
      break;
    case SendAckMessage::Status::kInvalidPipeline:
      message = "Invalid pipeline.";
      break;
    case SendAckMessage::Status::kAckError:
      message = "Error sent in the ack; completion failed.";
      break;
    case SendAckMessage::Status::kBadConnection:
      message =
          "Bad connection (Possible early connection teardown from "
          "remote side).";
      break;
    case SendAckMessage::Status::kSendRecvSizeMismatch:
      message = "Send and Receive sides have mismatched size.";
      break;
    case SendAckMessage::Status::kControlChannelFailure:
      message = "SCTP or LLCM control channel failed.";
      break;
    default:
      public_error_code = kUnknownErrorCode;  // An overwrite.
      message = "Unknown failure in Send Ack.";
      break;
  }
  return absl::StrCat("Code (", public_error_code, "): ", message);
}

/* Get a public error code and message for the given RecvAckMessage::Status. */
std::string GetDetailedStatus(RecvAckMessage::Status req_status) {
  // All of the req_status values are negative, so we use 1 as the default
  // error code for catch alls.
  constexpr int kUnknownErrorCode = 1;
  int public_error_code = static_cast<int>(req_status);
  std::string message;

  switch (req_status) {
    case RecvAckMessage::Status::kOk:
      message = "Everything is OK.";
      break;
    case RecvAckMessage::Status::kNoRegisteredReceiveBuffer:
      message = "Receive buffer is not registered.";
      break;
    case RecvAckMessage::Status::kInvalidRegHandle:
      message = "Invalid receive buffer reg handle.";
      break;
    case RecvAckMessage::Status::kInvalidRegisteredBuffer:
      message = "Invalid registered buffer.";
      break;
    case RecvAckMessage::Status::kInvalidSocket:
      message = "Invalid socket.";
      break;
    case RecvAckMessage::Status::kInvalidPipeline:
      message = "Invalid pipeline.";
      break;
    case RecvAckMessage::Status::kAckError:
      message = "Error sent in the ack; completion failed.";
      break;
    case RecvAckMessage::Status::kWrongElement:
      message = "Wrong element.";
      break;
    case RecvAckMessage::Status::kBadConnection:
      message =
          "Bad connection (Possible early connection teardown from "
          "remote side).";
      break;
    case RecvAckMessage::Status::kSendRecvSizeMismatch:
      message = "Send and Receive sides have mismatched size.";
      break;
    case RecvAckMessage::Status::kControlChannelFailure:
      message = "SCTP or LLCM control channel failed.";
      break;
    default:
      public_error_code = kUnknownErrorCode;  // An overwrite.
      message = "Unknown failure in Recv Ack.";
      break;
  }
  return absl::StrCat("Code (", public_error_code, "): ", message);
}

std::string GetDetailedStatus(ConnectAckMessage::Status req_status) {
  switch (req_status) {
    case ConnectAckMessage::Status::kOk:
      return "Everything is OK. Connection successful.";
    case ConnectAckMessage::Status::kInvalid:
      return "Invalid (or empty) connection request.";
    case ConnectAckMessage::Status::kNotServingVM:
      return "Got connection request but not serving any VM.";
    case ConnectAckMessage::Status::kVIPNotFound:
      return "VIP not found.";
    case ConnectAckMessage::Status::kAddSocketToEngineError:
      return "Unable to add socket to engine.";
    case ConnectAckMessage::Status::kClientDisconnected:
      return "Error opening to connection to DXS data channel.";
    case ConnectAckMessage::Status::kConnectionFailed:
      return "Connection failed.";
    case ConnectAckMessage::Status::kSnapNotEnabled:
      return "Snap not enabled.";
    default:
      return "Unknown(" + std::to_string(static_cast<int8_t>(req_status)) + ")";
  }
}

}  // namespace dxs

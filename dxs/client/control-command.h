/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_CONTROL_COMMAND_H_
#define DXS_CLIENT_CONTROL_COMMAND_H_

#include <stdint.h>
#include <sys/stat.h>

#include <ostream>
#include <string>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "dxs/client/control-command.pb.h"
#include "dxs/client/data-sock.h"
#include "dxs/client/dxs-client-types.h"
#include "google/protobuf/any.pb.h"
#include "google/rpc/status.pb.h"

namespace dxs {

enum class ControlCommand : uint8_t {
  kInvalid = 0,
  kInit = 1,
  kInitAck = 2,
  kListen = 3,
  kListenAck = 4,
  kIncomingConnection = 5,
  kAccept = 6,
  kCloseListenSock = 7,
  kConnect = 8,
  kConnectAck = 9,
  kSend = 10,
  kSendAck = 11,
  kRecv = 12,
  kRecvAck = 13,
  kRelease = 14,
  kCloseDataSock = 15,
  kRegBuffer = 16,
  kRegBufferAck = 17,
  kDeregBuffer = 18,
  kDeregBufferAck = 19,
  kPing = 22,
  kPong = 23,
  kVersionedInit = 34,
  kVersionedInitAck = 35,
  kInitLlcm = 36,
  kInitLlcmAck = 37,
  kAcceptAck = 48,  // Out of order since it was added after kAccept.
  kSubscribeToPeriodicStats = 52,
  kPeriodicStatsUpdate = 53,
  kIncomingConnectionV2 = 55,
  kNumControlCommands = 60,  // Not an actual command.
};
// CHANGES TO THIS ENUM ARE NOT BACKWARDS COMPATIBLE.
//
// If you are adding a new command, ensure to have your CL reviewed by the DXS
// team (such as dpcollins@) as well.

enum class OpStatus : uint8_t {
  kPending = 0,
  kComplete = 1,
  kError = 2,
};

struct __attribute__((__packed__)) WireSocketAddr {
  bool operator==(const WireSocketAddr& other) const = default;

  bool is_ipv6;
  uint8_t addr[16];  // An IP address packed with inet_pton
  uint16_t port;
};

struct __attribute__((__packed__)) InitMessage {
  static ControlCommand GetCommand() { return ControlCommand::kInit; }
  ControlCommand command = GetCommand();
  enum class ClientType : int8_t {
    kBufferManager = 0,
    // DEPRECATED: 1, do not use again.
    kSnapClient = 2,
  } client_type;
};

struct __attribute__((__packed__)) InitAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kInitAck; }
  ControlCommand command = GetCommand();
  enum class Status : int8_t {
    kOk = 0,
    kExists = -1,
    kSecureInitRequired = -2,
    kRejected = -3,
    kInvalidRequest = -4,
    kUnsupportedClientVersion = -5,
  } status;
};

struct __attribute__((__packed__)) VersionedInitMessage {
  static ControlCommand GetCommand() { return ControlCommand::kVersionedInit; }
  ControlCommand command = GetCommand();
  InitMessage::ClientType client_type;

  // Null-terminated.
  char client_build_id[256];
  uint64_t client_version;
};

struct __attribute__((__packed__)) VersionedInitAckMessage {
  static ControlCommand GetCommand() {
    return ControlCommand::kVersionedInitAck;
  }
  ControlCommand command = GetCommand();
  InitAckMessage::Status status;

  // Null-terminated.
  char server_build_id[256];
  uint64_t server_version;
};

struct __attribute__((__packed__)) InitLlcmMessage {
  static ControlCommand GetCommand() { return ControlCommand::kInitLlcm; }
  ControlCommand command = GetCommand();
};

struct __attribute__((__packed__)) InitLlcmAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kInitLlcmAck; }
  ControlCommand command = GetCommand();

  enum class Status : int8_t {
    kOk = 0,
    kFailed = -1,
  } status;

  uint64_t llcm_queue_offset;
  uint64_t llcm_queue_size;
  uint64_t reverse_llcm_queue_offset;
  uint64_t reverse_llcm_queue_size;
};

struct __attribute__((__packed__)) ListenMessage {
  static ControlCommand GetCommand() { return ControlCommand::kListen; }
  ControlCommand command = GetCommand();
  ListenSocketHandle sock;
};

struct __attribute__((__packed__)) ListenAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kListenAck; }
  ControlCommand command = GetCommand();
  ListenSocketHandle sock;
  uint16_t port;
  enum class Status : int8_t {
    kOk = 0,
    kInvalidRequest = -1,
    kOutOfFreePorts = -2,
    kHandlerAlreadyInUse = -3,
  } status;
};

struct __attribute__((__packed__)) IncomingConnectionMessage {
  static ControlCommand GetCommand() {
    return ControlCommand::kIncomingConnection;
  }
  ControlCommand command = GetCommand();
  ListenSocketHandle sock;
};

struct __attribute__((__packed__)) IncomingConnectionMessageV2 {
  static ControlCommand GetCommand() {
    return ControlCommand::kIncomingConnectionV2;
  }
  ControlCommand command = GetCommand();
  ListenSocketHandle sock;
  WireSocketAddr peer_saddr;
};

struct __attribute__((__packed__)) AcceptMessage {
  static ControlCommand GetCommand() { return ControlCommand::kAccept; }
  ControlCommand command = GetCommand();
  ListenSocketHandle listen_sock;
  DataSocketHandle data_sock;
};

struct __attribute__((__packed__)) AcceptAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kAcceptAck; }
  ControlCommand command = GetCommand();
  DataSocketHandle sock;
  enum class Status : int8_t {
    kOk = 0,
    kInvalidRequest = -1,
    kInvalidListenSocket = -2,
    kInternalError = -3,
    kNoPendingConnection = -4,
    kInvalidClientType = -5,
  } status;
};

struct __attribute__((__packed__)) CloseListenSockMessage {
  static ControlCommand GetCommand() {
    return ControlCommand::kCloseListenSock;
  }
  ControlCommand command = GetCommand();
  ListenSocketHandle sock;
};

struct __attribute__((__packed__)) ConnectMessage {
  static ControlCommand GetCommand() { return ControlCommand::kConnect; }
  ControlCommand command = GetCommand();
  uint8_t addr[16];
  DataSocketHandle sock;
  uint16_t port;
  bool ipv6;
};

struct __attribute__((__packed__)) ConnectAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kConnectAck; }
  ControlCommand command = GetCommand();
  DataSocketHandle sock;
  DataSockStatus status;
  enum class Status : int8_t {
    kOk = 0,
    kInvalid = -1,
    kNotServingVM = -2,
    kVIPNotFound = -3,
    kAddSocketToEngineError = -4,
    kClientDisconnected = -5,
    kTimeout = -6,
    kConnectionFailed = -7,
    kSnapNotEnabled = -8,
  } req_status;
};

struct __attribute__((__packed__)) SendMessage {
  static ControlCommand GetCommand() { return ControlCommand::kSend; }
  ControlCommand command = GetCommand();
  DataSocketHandle sock;
  Reg reg_handle;
  OpId op_id;
  uint64_t offset;
  uint64_t size;
};

struct __attribute__((__packed__)) SendAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kSendAck; }
  ControlCommand command = GetCommand();
  OpId op_id;
  OpStatus status;
  enum class Status : int8_t {
    kOk = 0,
    kInvalidRegHandle = -1,
    kInvalidRegisteredBuffer = -2,
    kInvalidSocket = -3,
    kInvalidPipeline = -4,
    kAckError = -5,
    kWrongElement = -6,
    kBadConnection = -7,
    kSendRecvSizeMismatch = -8,
    kControlChannelFailure = -9,
  } req_status;
};

struct __attribute__((__packed__)) RecvMessage {
  static ControlCommand GetCommand() { return ControlCommand::kRecv; }
  ControlCommand command = GetCommand();
  DataSocketHandle sock;
  uint64_t offset;
  uint64_t size;
  Reg reg_handle;
  OpId op_id;
};

struct __attribute__((__packed__)) RecvAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kRecvAck; }
  ControlCommand command = GetCommand();

  OpId op_id;
  // The status of the operation.  If kPending, one or more RecvAckMessages will
  // follow for this op_id.  If kComplete or kError, this is the last
  // RecvAckMessage for this op.
  OpStatus status;

  // For client-linearized requests: the number of iovecs which follow this
  // message.
  //
  // For server-linearized requests: the size of the received message.
  uint64_t size;

  enum class Status : int8_t {
    kOk = 0,
    kNoRegisteredReceiveBuffer = -1,
    kInvalidRegHandle = -2,
    kInvalidRegisteredBuffer = -3,
    kInvalidSocket = -4,
    kInvalidPipeline = -5,
    kAckError = -6,
    kWrongElement = -7,
    kBadConnection = -8,
    kSendRecvSizeMismatch = -9,
    kControlChannelFailure = -10,
  } req_status;

  // Immediately following this struct in memory is an array of iovec entries.
  // They are not explicitly included in this structure as it causes problems
  // for logging this structure.
  // iovec iovecs[];
};

struct __attribute__((__packed__)) ReleaseMessage {
  static ControlCommand GetCommand() { return ControlCommand::kRelease; }
  ControlCommand command = GetCommand();
  // Tells DxsServer to release all of the buffers associated with this Recv op.
  DataSocketHandle sock;
  OpId op_id;
};

struct __attribute__((__packed__)) CloseDataSockMessage {
  static ControlCommand GetCommand() { return ControlCommand::kCloseDataSock; }
  ControlCommand command = GetCommand();
  DataSocketHandle sock;
};

struct __attribute__((__packed__)) RegBufferMessage {
  static ControlCommand GetCommand() { return ControlCommand::kRegBuffer; }
  ControlCommand command = GetCommand();

  // Whether or not another RegBufferMessage will follow for the same
  // registration.  When `more` == false, a registration handle for the current
  // message and all immediately previous messages with `more` == true will be
  // returned.
  bool more;

  // Legacy "is_bounce_buffer" field.
  bool is_bounce_buffer = false;

  // The length of the `gpas` array.
  //
  // The maximum number of GPAs per RegBufferMessage is limited by the
  // message size, which is limited by the MTU.
  //    1 byte  sizeof(ControlCommand)
  //    2 bytes sizeof(RegBufferMessage)
  //   16 bytes for each iovec
  //
  //  Current message size is capped at 1480 bytes
  //  (1500 byte MTU, 20 bytes for IPv4 header).
  //  1480-1-2 = 1477.
  //  1477 \ 16 = 92.
  uint8_t num_gpas;

  // Immediately following this struct in memory is an array of iovec entries.
  // They are not explicitly included in this structure as it causes problems
  // for logging this structure.
  // iovec gpas[];
};

struct __attribute__((__packed__)) RegBufferAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kRegBufferAck; }
  ControlCommand command = GetCommand();

  // The registration handle.  kInvalidRegistration will be sent for each
  // RegBufferMessage with `more` == true.
  Reg reg_handle;

  // The total number of guest physical addresses this message is acknowledging.
  // For each RegBufferMessage with `more` == true, the server will
  // accumulate GPAs in a group until `more` == false.
  //
  // Example with 3 registrations of sizes [10, 20, 12] in a group:
  // client: RegBufferMessage    (more=true, num_gpas=10)
  // server: RegBufferAckMessage (reg_handle=invalid, num_gpas=10)
  // client: RegBufferMessage    (more=true, num_gpas=20)
  // server: RegBufferAckMessage (reg_handle=invalid, num_gpas=30)
  // client: RegBufferMessage    (more=false, num_gpas=12)
  // server: RegBufferAckMessage (reg_handle=0xaabbccdd, num_gpas=42)
  uint32_t num_gpas;

  enum class Status : int8_t {
    kOk = 0,
    kOverlappingRegions = -1,
    kInvalidRegion = -2,
    kAlreadyRegistered = -3,
    kMismatchedDirection = -4,
    kRegistrationFailed = -5,
    kNotAllowed = -6,
  } status;
};

struct __attribute__((__packed__)) DeregBufferMessage {
  static ControlCommand GetCommand() { return ControlCommand::kDeregBuffer; }
  ControlCommand command = GetCommand();

  // The registration handle.  Must match the reg_handle returned in a previous
  // RegBufferAckMessage.
  Reg reg_handle;
  // Legacy "is_bounce_buffer" field.
  bool is_bounce_buffer = false;
};

struct __attribute__((__packed__)) DeregBufferAckMessage {
  static ControlCommand GetCommand() { return ControlCommand::kDeregBufferAck; }
  ControlCommand command = GetCommand();

  enum class Status : int8_t {
    kOk = 0,
    kNotFound = -1,
    kNotAllowed = -2,
    kInternal = -3,
  } status;
};

struct __attribute__((__packed__)) PingMessage {
  static ControlCommand GetCommand() { return ControlCommand::kPing; }
  ControlCommand command = GetCommand();

  int seq;
};

struct __attribute__((__packed__)) PongMessage {
  static ControlCommand GetCommand() { return ControlCommand::kPong; }
  ControlCommand command = GetCommand();

  int seq;
};

struct __attribute__((__packed__)) SubscribeToPeriodicStatsMessage {
  static ControlCommand GetCommand() {
    return ControlCommand::kSubscribeToPeriodicStats;
  }
  ControlCommand command = GetCommand();

  uint16_t payload_size = 0;
  // Immediately following this struct in memory is the serialized
  // SubscribeToPeriodicStats proto.
  uint8_t payload[0];
};

struct __attribute__((__packed__)) PeriodicStatsUpdateMessage {
  static ControlCommand GetCommand() {
    return ControlCommand::kPeriodicStatsUpdate;
  }
  ControlCommand command = GetCommand();

  uint16_t payload_size = 0;
  // Immediately following this struct in memory is the serialized
  // PeriodicStatsUpdate proto.
  uint8_t payload[0];
};

std::string ToString(ControlCommand cmd);
std::string ToString(InitAckMessage::Status status);
std::string ToString(InitLlcmAckMessage::Status status);
std::string ToString(AcceptAckMessage::Status status);
std::string ToString(RegBufferAckMessage::Status status);
std::string ToString(DeregBufferAckMessage::Status status);
std::string ToString(OpStatus status);
std::string ToString(ListenSocketHandle handle);
std::string ToString(DataSocketHandle handle);

std::string ToString(const InitMessage& msg, std::string_view delim = "\n");
std::string ToString(const InitAckMessage& msg, std::string_view delim = "\n");
std::string ToString(const VersionedInitMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const VersionedInitAckMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const InitLlcmMessage& msg, std::string_view delim = "\n");
std::string ToString(const InitLlcmAckMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const ListenMessage& msg, std::string_view delim = "\n");
std::string ToString(const ListenAckMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const IncomingConnectionMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const AcceptMessage& msg, std::string_view delim = "\n");
std::string ToString(const AcceptAckMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const CloseListenSockMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const ConnectMessage& msg, std::string_view delim = "\n");
std::string ToString(const ConnectAckMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const SendMessage& msg, std::string_view delim = "\n");
std::string ToString(const SendAckMessage& msg, std::string_view delim = "\n");
std::string ToString(const RecvMessage& msg, std::string_view delim = "\n");
std::string ToString(const RecvAckMessage& msg, std::string_view delim = "\n");
std::string ToString(const ReleaseMessage& msg, std::string_view delim = "\n");
std::string ToString(const CloseDataSockMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const RegBufferMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const DeregBufferMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const DeregBufferAckMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const PingMessage& msg, std::string_view delim = "\n");
std::string ToString(const PongMessage& msg, std::string_view delim = "\n");
std::string ToString(const SubscribeToPeriodicStatsMessage& msg,
                     std::string_view delim = "\n");
std::string ToString(const PeriodicStatsUpdateMessage& msg,
                     std::string_view delim = "\n");

std::string GetDetailedStatus(InitAckMessage::Status status);
std::string GetDetailedStatus(RegBufferAckMessage::Status status);
std::string GetDetailedStatus(DeregBufferAckMessage::Status status);
std::string GetDetailedStatus(ListenAckMessage::Status status);
std::string GetDetailedStatus(SendAckMessage::Status req_status);
std::string GetDetailedStatus(RecvAckMessage::Status req_status);
std::string GetDetailedStatus(ConnectAckMessage::Status req_status);

template <typename Sink>
void AbslStringify(Sink& sink, ControlCommand cmd) {
  sink.Append(ToString(cmd));
}

inline std::ostream& operator<<(std::ostream& os, ListenSocketHandle handle) {
  return os << static_cast<uint64_t>(handle);
}

inline std::ostream& operator<<(std::ostream& os, DataSocketHandle handle) {
  return os << static_cast<uint64_t>(handle);
}

template <typename ProtoT, typename MessageT>
absl::StatusOr<ProtoT> GetPayloadProto(const MessageT& msg) {
  ProtoT payload;
  if (!payload.ParseFromString(
          {reinterpret_cast<const char*>(msg.payload), msg.payload_size})) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse ", ProtoT::descriptor()->name(), " proto"));
  }
  return payload;
}

template <class MessageT>
absl::Status GetPayloadStatus(const MessageT& msg) {
  google::rpc::Status proto;
  if (!proto.ParseFromString(
          {reinterpret_cast<const char*>(msg.payload), msg.payload_size})) {
    return absl::InvalidArgumentError("Failed to parse Status proto");
  }
  if (proto.code() == 0) return absl::OkStatus();
  absl::Status ret(static_cast<absl::StatusCode>(proto.code()),
                   proto.message());
  for (const google::protobuf::Any& detail : proto.details()) {
    ret.SetPayload(detail.type_url(), absl::Cord(detail.value()));
  }
  return ret;
}

template <typename MessageT>
const MessageT* absl_nullable ValidateAndGetMessage(
    absl::Span<const uint8_t> packet) {
  constexpr uint64_t kExpectedLength = sizeof(MessageT);
  if (packet.size() != kExpectedLength) {
    LOG(DFATAL) << absl::StrFormat(
        "Incorrect size for message. Expected: %d Actual: %d", kExpectedLength,
        packet.size());
    return nullptr;
  }
  return reinterpret_cast<const MessageT*>(packet.data());
}

}  // namespace dxs

#endif  // DXS_CLIENT_CONTROL_COMMAND_H_

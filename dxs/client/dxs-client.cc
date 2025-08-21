/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/dxs-client.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <linux/sctp.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <netinet/udp.h>
#include <sched.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dxs/client/atomic-future.h"
#include "dxs/client/control-command.h"
#include "dxs/client/control-command.pb.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/data-sock.h"
#include "dxs/client/derive_dxs_address.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/guest-llcm-device.h"
#include "dxs/client/linearized-recv-op.h"
#include "dxs/client/listen-socket.h"
#include "dxs/client/llcm-handler.h"
#include "dxs/client/llcm-memory-interface.h"
#include "dxs/client/make-unique-with-cleanup.h"
#include "dxs/client/monotonic-timestamp.h"
#include "dxs/client/mtu_utils.h"
#include "dxs/client/oss/status_macros.h"
#include "dxs/client/recv-socket.h"
#include "dxs/client/relaxed-atomic.h"
#include "dxs/client/sctp-handler.h"
#include "dxs/client/send-op.h"
#include "dxs/client/send-socket.h"
#include "dxs/client/wire-version.h"

ABSL_FLAG(absl::Duration, TEST_ONLY_dxs_client_default_periodic_stats_interval,
          absl::Seconds(10),
          "Default interval for the client to update periodic stats. Used in "
          "test to wait for stats propagation.");

ABSL_FLAG(absl::Duration, buffer_manager_init_timeout, absl::InfiniteDuration(),
          "Duration to wait for the buffer manager to initialize.");
ABSL_FLAG(absl::Duration, dxs_client_llcm_response_timeout, absl::Seconds(10),
          "Duration to wait for a response from DXS when using Llcm.");
ABSL_FLAG(absl::Duration, dxs_client_storage_read_timeout, absl::Minutes(5),
          "Duration to wait for client to read from storage.");
ABSL_FLAG(absl::Duration, dxs_client_storage_write_timeout, absl::Minutes(5),
          "Duration to wait for client to write from storage.");
ABSL_FLAG(absl::Duration, dxs_client_buffer_register_timeout, absl::Minutes(5),
          "Duration to wait for client to register buffers.");
ABSL_FLAG(absl::Duration, dxs_client_buffer_deregister_timeout,
          absl::Minutes(5),
          "Duration to wait for client to deregister buffers.");

namespace dxs {

namespace {
constexpr char kAnyPort[] = "0";
constexpr size_t kMinMtu = 1460;
constexpr size_t kBufferSize = kMinMtu;
const size_t kAvailableBufferSize = GetUsableSizeForMtu(kMinMtu);
}  // namespace

absl::StatusOr<std::unique_ptr<DxsClientInterface>> DxsClient::Create(
    std::string nic_addr, bool enable_llcm, std::string llcm_device_directory,
    bool send_close_on_teardown) {
  std::string dxs_addr = kDefaultDxsAddr;

  return Create(nic_addr, dxs_addr, kDefaultDxsPort, kAnyPort, enable_llcm,
                llcm_device_directory, send_close_on_teardown);
}

absl::StatusOr<std::unique_ptr<DxsClientInterface>>
DxsClient::CreateWithLlcmMemory(
    std::string nic_addr, std::string dxs_addr, std::string dxs_port,
    std::string source_port, std::unique_ptr<LlcmMemoryInterface> llcm_memory,
    bool send_close_on_teardown) {
  return CreateInternal(nic_addr, dxs_addr, dxs_port, source_port,
                        std::move(llcm_memory), send_close_on_teardown);
}

absl::StatusOr<std::unique_ptr<DxsClientInterface>> DxsClient::Create(
    std::string nic_addr, std::string dxs_addr, std::string dxs_port,
    std::string source_port, bool enable_llcm,
    std::string llcm_device_directory, bool send_close_on_teardown) {
  std::unique_ptr<GuestLlcmDevice> llcm_device;
  if (enable_llcm) {
    ASSIGN_OR_RETURN(WireSocketAddr local_address, PackIpAddress(nic_addr, 0));
    llcm_device = std::make_unique<GuestLlcmDevice>();
    RETURN_IF_ERROR(llcm_device->Init(local_address, llcm_device_directory));
  }
  return CreateWithLlcmMemory(nic_addr, dxs_addr, dxs_port, source_port,
                              std::move(llcm_device), send_close_on_teardown);
}

absl::StatusOr<std::unique_ptr<DxsClientInterface>> DxsClient::CreateInternal(
    std::string nic_addr, std::string dxs_addr, std::string dxs_port,
    std::string source_port, std::unique_ptr<LlcmMemoryInterface> llcm_memory,
    bool send_close_on_teardown) {
  auto client = absl::WrapUnique(new DxsClient(
      std::move(nic_addr), std::move(llcm_memory), send_close_on_teardown));
  RETURN_IF_ERROR(client->Init(dxs_addr, dxs_port, source_port));
  return client;
}

namespace {

uint16_t GetPortFromAddrInfo(struct addrinfo* ai) {
  struct sockaddr* sa = ai->ai_addr;
  if (sa->sa_family == AF_INET)
    return ntohs((((struct sockaddr_in*)sa)->sin_port));

  return ntohs((((struct sockaddr_in6*)sa)->sin6_port));
}

}  // namespace

absl::Status DxsClient::Init(std::string dxs_addr, std::string dxs_port,
                             std::string source_port) {
  const bool enable_llcm = (llcm_memory_ != nullptr);
  LOG(INFO) << absl::StrFormat(
      "Initializing DxsClient (src_addr=%s:%s, dxs_addr=%s:%s, llcm=%s)",
      nic_addr_, source_port, dxs_addr, dxs_port,
      enable_llcm ? "true" : "false");

  ASSIGN_OR_RETURN(message_handler_do_not_access_directly_,
                   CreateControlMessageHandler(
                       nic_addr_, dxs_addr, dxs_port, source_port,
                       /*is_buffer_manager =*/false, enable_llcm, *this));
  // Wait for DXS initialization.
  auto done = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    bool awaiting_llcm = enable_llcm && !llcm_info_.has_value();
    return failed_ || (initialized_ && !awaiting_llcm);
  };
  absl::MutexLock l(&mu_);
  mu_.Await(absl::Condition(&done));
  if (failed_) {
    return absl::InternalError("Failed to initialize DXS.");
  }

  // Finish setting up Llcm.
  if (enable_llcm) {
    DCHECK(llcm_info_.has_value());
    auto llcm_handler = std::make_unique<LlcmHandler>(
        *this, std::move(message_handler_do_not_access_directly_));
    RETURN_IF_ERROR(llcm_handler->Init(
        /* local_memory = */ llcm_memory_->GetLocalMemory(
            llcm_info_->reverse_llcm_queue_offset,
            llcm_info_->reverse_llcm_queue_size),
        /* remote_memory = */ llcm_memory_->GetRemoteMemory(
            llcm_info_->llcm_queue_offset, llcm_info_->llcm_queue_size)));
    message_handler_do_not_access_directly_ = std::move(llcm_handler);
  }

  LOG(INFO) << "DxsClient initialized.";
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<BufferManagerInterface>> BufferManager::Create(
    BufferManagerOptions options) {
  in_addr ipv4_addr;
  in6_addr ipv6_addr;
  int source_addr_family = AF_UNSPEC;
  if (!inet_pton(AF_INET, options.source_addr.c_str(), &ipv4_addr)) {
    if (inet_pton(AF_INET6, options.source_addr.c_str(), &ipv6_addr)) {
      source_addr_family = AF_INET6;
    }
  } else {
    source_addr_family = AF_INET;
  }
  std::string dest_addr;
  if (options.dest_addr_override) {
    dest_addr = *options.dest_addr_override;
  } else {
    switch (source_addr_family) {
      case AF_INET: {
        dest_addr = kDefaultDxsAddr;
        break;
      }
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported address family: ", source_addr_family));
    }
  }

  LOG(INFO) << absl::StrFormat(
      "Initializing BufferManager (src_addr=%s:%s, dest_addr=%s:%s)",
      options.source_addr, options.source_port, dest_addr, options.dest_port);

  auto manager = absl::WrapUnique(new BufferManager);
  RETURN_IF_ERROR(manager->Init(options.source_addr, options.source_port,
                                dest_addr, options.dest_port,
                                options.periodic_stats_options));
  return manager;
}

absl::StatusOr<std::unique_ptr<BufferManagerInterface>> BufferManager::Create(
    std::string nic_addr) {
  return Create(BufferManagerOptions{
      .source_addr = nic_addr,
  });
}

absl::StatusOr<std::unique_ptr<BufferManagerInterface>> BufferManager::Create(
    std::string nic_addr, std::string dxs_addr, std::string dxs_port,
    std::string source_port) {
  return Create(BufferManagerOptions{
      .source_addr = nic_addr,
      .source_port = source_port,
      .dest_addr_override = std::make_optional(dxs_addr),
      .dest_port = dxs_port,
  });
}

absl::Status BufferManager::Init(
    std::string source_addr, std::string source_port, std::string dest_addr,
    std::string dest_port,
    std::optional<PeriodicStatsOptions> periodic_stats_options) {
  ASSIGN_OR_RETURN(message_handler_do_not_access_directly_,
                   CreateControlMessageHandler(source_addr, dest_addr,
                                               dest_port, source_port,
                                               /*is_buffer_manager=*/true,
                                               /*enable_llcm=*/false, *this));
  // Wait for the initial ack from DXS.
  auto done = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return failed_ || initialized_;
  };
  absl::MutexLock l(&mu_);
  if (!mu_.AwaitWithTimeout(absl::Condition(&done),
                            absl::GetFlag(FLAGS_buffer_manager_init_timeout))) {
    return absl::DeadlineExceededError("Initialize BufferManager timed out.");
  }
  if (failed_) {
    return absl::InternalError("Failed to initialize BufferManager.");
  }

  const bool kSubscribeToPeriodicStatsSupported =
      server_version_ >= kSubscribeToPeriodicStatsVersion;
  if (periodic_stats_options) {
    if (!kSubscribeToPeriodicStatsSupported) {
      LOG(WARNING) << "Periodic stats unsupported by current server version";
    } else {
      periodic_stats_handler_ = std::move(periodic_stats_options->handler);

      ProtoInit::SubscribeToPeriodicStats subscribe;
      subscribe.set_period_ms(absl::ToInt64Milliseconds(
          periodic_stats_options->periodic_stats_interval));
      RETURN_IF_ERROR(
          message_handler_do_not_access_directly_->SendMessageWithPayload(
              SubscribeToPeriodicStatsMessage{}, subscribe));
    }
  }

  LOG(INFO) << "BufferManager client initialized.";
  return absl::OkStatus();
}

DxsClient::~DxsClient() {
  absl::MutexLock l(&mu_);
  if (!outstanding_listen_sockets_.empty() ||
      !outstanding_data_sockets_.empty() ||
      !outstanding_linearized_recv_ops_.empty() ||
      !outstanding_send_ops_.empty()) {
    LOG(FATAL) << absl::StrFormat(  // Crash OK
        "DxsClient destroyed with outstanding objects, aborting to avoid "
        "likely segfault. Listen Sockets: %d, Data Sockets: %d, Recv Ops: %d, "
        "Send Ops: %d",
        outstanding_listen_sockets_.size(), outstanding_data_sockets_.size(),
        outstanding_linearized_recv_ops_.size(), outstanding_send_ops_.size());
  }
}

namespace {

// Returns true iff 'a' and 'b' represent the same address. Only works for
// IPv4/IPv6 addresses.
bool IsSockaddrEqual(const sockaddr* a, const sockaddr* b) {
  if (a->sa_family != b->sa_family) return false;
  if (a->sa_family == AF_INET) {
    auto* sa = reinterpret_cast<const sockaddr_in*>(a);
    auto* sb = reinterpret_cast<const sockaddr_in*>(b);
    return sa->sin_addr.s_addr == sb->sin_addr.s_addr;
  }
  if (a->sa_family == AF_INET6) {
    auto* sa = reinterpret_cast<const sockaddr_in6*>(a);
    auto* sb = reinterpret_cast<const sockaddr_in6*>(b);
    return memcmp(sa->sin6_addr.s6_addr, sb->sin6_addr.s6_addr,
                  sizeof(in6_addr::s6_addr)) == 0;
  }
  LOG(ERROR) << absl::StrFormat(
      "IsSockaddrEqual with invalid address family: %d", a->sa_family);
  return false;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ControlMessageHandlerInterface>>
CreateControlMessageHandler(std::string nic_addr, std::string dxs_addr,
                            std::string dxs_port, std::string source_port,
                            bool is_buffer_manager, bool enable_llcm,
                            ControlMessageReceiverInterface& receiver) {
  std::string client_id = source_port + "->" + dxs_addr + ":" + dxs_port;
  // Setup dest address.
  struct addrinfo hints = {};
  hints.ai_flags = AI_NUMERICHOST;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_protocol = IPPROTO_UDP;

  struct addrinfo* dest_addr;
  int result =
      getaddrinfo(dxs_addr.c_str(), dxs_port.c_str(), &hints, &dest_addr);
  if (result == EAI_SYSTEM) {
    return absl::InternalError(
        absl::StrCat("getaddrinfo failed: ", strerror(errno)));
  }
  if (result != 0) {
    return absl::InternalError(
        absl::StrCat("getaddrinfo failed: ", gai_strerror(result)));
  }
  absl::Cleanup dest_addr_cleanup = [dest_addr] { freeaddrinfo(dest_addr); };

  // Setup source address. Port 0 allows for an ephemeral port to be chosen
  // automatically.
  struct addrinfo* src_addr;
  result =
      getaddrinfo(nic_addr.c_str(), source_port.c_str(), &hints, &src_addr);
  if (result == EAI_SYSTEM) {
    return absl::ErrnoToStatus(errno, "getaddrinfo failed");
  }
  if (result != 0) {
    return absl::InternalError(
        absl::StrCat("getaddrinfo failed: ", gai_strerror(result)));
  }
  absl::Cleanup src_addr_cleanup = [src_addr] { freeaddrinfo(src_addr); };

  // Open, bind, and connect a socket.
  int socket_fd =
      socket(dest_addr->ai_family, hints.ai_socktype, hints.ai_protocol);
  if (socket_fd < 0) {
    return absl::ErrnoToStatus(errno, "socket(af, sock_type, protocol) failed");
  }
  result = bind(socket_fd, src_addr->ai_addr, src_addr->ai_addrlen);
  if (result != 0) {
    return absl::ErrnoToStatus(errno, "bind failed");
  }

  // Find device for source IP.
  ifaddrs* ifap;
  getifaddrs(&ifap);
  std::string device_name;
  absl::Cleanup ifaddrs_cleanup([ifap] { freeifaddrs(ifap); });
  for (ifaddrs* ifa = ifap; ifa; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr && IsSockaddrEqual(ifa->ifa_addr, src_addr->ai_addr)) {
      LOG(INFO) << absl::StrFormat("Binding to interface: %s", ifa->ifa_name);
      device_name = ifa->ifa_name;
      break;
    }
  }
  if (device_name.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("No interface found for source IP: ", nic_addr));
  }

  char device[IFNAMSIZ];
  memset(device, '\0', IFNAMSIZ);
  if (device_name.size() > IFNAMSIZ) {
    return absl::InternalError(
        absl::StrCat("Invalid interface name: ", device_name));
  }
  memcpy(device, device_name.c_str(), device_name.size());
  if (setsockopt(socket_fd, SOL_SOCKET, SO_BINDTODEVICE, device,
                 sizeof(device))) {
    return absl::ErrnoToStatus(errno, "bind failed");
  }

  result = connect(socket_fd, dest_addr->ai_addr, dest_addr->ai_addrlen);
  if (result != 0) {
    return absl::ErrnoToStatus(errno, "connect failed");
  }

  // Grab the assigned source port. Maybe be different than the one we specified
  // (which might be 0).
  sockaddr_in sin;
  socklen_t addr_len = sizeof(sin);
  result = getsockname(socket_fd, (struct sockaddr*)&sin, &addr_len);
  if (result != 0) {
    return absl::ErrnoToStatus(errno, "getsockname failed");
  }
  uint16_t actual_source_port = ntohs(sin.sin_port);
  LOG(INFO) << absl::StrFormat(
      "DxsClient connected (src_addr=%s:%hu, socket_fd=%d)", nic_addr,
      actual_source_port, socket_fd);

  ControlMessageReceiverInterface* message_receiver = &receiver;

  // Create Control Message Handler.
  std::unique_ptr<ControlMessageHandlerInterface> message_handler;
  auto sctp_handler = std::make_unique<SctpHandler>(
      *message_receiver, client_id, socket_fd, actual_source_port,
      GetPortFromAddrInfo(dest_addr));
  if (!sctp_handler->Init()) {
    return absl::InternalError("Failed to initialize SctpHandler");
  }
  message_handler = std::move(sctp_handler);

  InitMessage::ClientType client_type =
      is_buffer_manager ? InitMessage::ClientType::kBufferManager
                        : InitMessage::ClientType::kSnapClient;

  // Send an initial message to DXS.
  VersionedInitMessage init_message = {};

  std::string build_label =
      absl::StrFormat("dxs_client_cloud_%s",
                      absl::FormatTime("%E4Y%m%d_%H:%M:%S%Ez", absl::Now(),
                                       absl::UTCTimeZone()));
  const auto build_id_len = sizeof(init_message.client_build_id) - 1;
  strncpy(init_message.client_build_id, build_label.c_str(), build_id_len - 1);
  init_message.client_build_id[build_id_len - 1] = '\0';

  init_message.client_version = kWireVersion;
  init_message.client_type = client_type;
  RETURN_IF_ERROR(message_handler->SendMessage(&init_message));

  if (enable_llcm) {
    InitLlcmMessage init_llcm_message;
    RETURN_IF_ERROR(message_handler->SendMessage(&init_llcm_message));
  }

  return message_handler;
}

absl::StatusOr<std::unique_ptr<ListenSocketInterface>> DxsClient::Listen() {
  // Prepare a ListenMessage to send to DXS.
  ListenSocketHandle handle = NextListenSocketHandle();
  ListenMessage message;
  message.sock = handle;

  auto sock = RegisterListenSocket(handle);

  RETURN_IF_ERROR(message_handler().SendMessage(&message));
  return sock;
}

absl::StatusOr<std::unique_ptr<SendSocketInterface>> DxsClient::Connect(
    std::string addr, uint16_t port) {
  ASSIGN_OR_RETURN(WireSocketAddr wire_addr, PackIpAddress(addr, port));
  // Prepare a ConnectMessage to send to DXS.
  ConnectMessage message{.port = wire_addr.port, .ipv6 = wire_addr.is_ipv6};
  static_assert(sizeof(message.addr) == sizeof(wire_addr.addr));
  std::memcpy(message.addr, wire_addr.addr, sizeof(message.addr));

  // Create a socket handle to identify this connection.
  DataSocketHandle handle = NextDataSocketHandle();
  message.sock = handle;

  auto sock = RegisterSendSocket(handle, wire_addr);

  RETURN_IF_ERROR(message_handler().SendMessage(&message));
  return sock;
}

absl::StatusOr<absl::Duration> DxsClient::Ping() {
  uint32_t seq;
  {
    absl::MutexLock l(&mu_);
    seq = ++ping_seq_;
    outstanding_pings_[seq] = absl::Now();
  }

  PingMessage message;
  message.seq = seq;
  RETURN_IF_ERROR(message_handler().SendMessage(&message));

  while (true) {
    message_handler().RxPoll();
    absl::MutexLock l(&mu_);
    if (failed_ || !outstanding_pings_.contains(seq)) {
      auto result = ping_results_.find(seq);
      if (result == ping_results_.end()) {
        return absl::NotFoundError("Ping not found");
      }
      absl::Duration ret = result->second;
      ping_results_.erase(seq);
      return ret;
    }
  }
}

std::unique_ptr<SendOpInterface> DxsClient::RegisterSendOp(OpId id) {
  auto state = new SendOp::SharedState();
  {
    absl::MutexLock l(&mu_);
    outstanding_send_ops_[id] = absl::WrapUnique(state);
  }
  return MakeUniqueWithCleanup<SendOp>(
      [this, id] {
        absl::MutexLock l(&mu_);
        outstanding_send_ops_.erase(id);
      },
      message_handler(), *state, id);
}

std::unique_ptr<LinearizedRecvOpInterface> DxsClient::RegisterLinearizedRecvOp(
    DataSocketHandle handle, OpId id, uint64_t size) {
  auto state = new LinearizedRecvOp::SharedState();
  {
    absl::MutexLock l(&mu_);
    outstanding_linearized_recv_ops_[id] = absl::WrapUnique(state);
  }
  return MakeUniqueWithCleanup<LinearizedRecvOp>(
      [this, id] {
        absl::MutexLock l(&mu_);
        outstanding_linearized_recv_ops_.erase(id);
      },
      message_handler(), *state, handle, id);
}

std::unique_ptr<RecvSocketInterface> DxsClient::RegisterRecvSocket(
    DataSocketHandle handle, WireSocketAddr peer) {
  // Currently a new receive socket starts in the connected state because DXS
  // does not yet send an AcceptAck.
  auto status =
      new RelaxedAtomic<DataSockStatus>(DataSockStatus::kPendingConnect);
  {
    absl::MutexLock l(&mu_);
    const bool kAcceptAckSupported = server_version_ >= kAcceptAckVersion;
    if (!kAcceptAckSupported) {
      // If DXS is old and doesn't send an AcceptAck message we directly move
      // the socket to the connected state instead of waiting for an AcceptAck.
      status->Store(DataSockStatus::kConnected);
    }
    outstanding_data_sockets_[handle] = absl::WrapUnique(status);
  }
  return MakeUniqueWithCleanup<RecvSocket>(
      [this, handle] {
        absl::MutexLock l(&mu_);
        outstanding_data_sockets_.erase(handle);
      },
      message_handler(), *this, next_op_id_, *status, handle, peer);
}

std::unique_ptr<SendSocketInterface> DxsClient::RegisterSendSocket(
    DataSocketHandle handle, WireSocketAddr peer) {
  auto status =
      new RelaxedAtomic<DataSockStatus>(DataSockStatus::kPendingConnect);
  {
    absl::MutexLock l(&mu_);
    outstanding_data_sockets_[handle] = absl::WrapUnique(status);
  }
  return MakeUniqueWithCleanup<SendSocket>(
      [this, handle] {
        absl::MutexLock l(&mu_);
        outstanding_data_sockets_.erase(handle);
      },
      message_handler(), *this, next_op_id_, *status, handle, nic_addr_, peer,
      send_close_on_teardown_);
}

std::unique_ptr<ListenSocketInterface> DxsClient::RegisterListenSocket(
    ListenSocketHandle handle) {
  auto state = new ListenSocket::SharedState();
  {
    absl::MutexLock l(&mu_);
    outstanding_listen_sockets_[handle] = absl::WrapUnique(state);
  }
  return MakeUniqueWithCleanup<ListenSocket>(
      [this, handle] {
        absl::MutexLock l(&mu_);
        outstanding_listen_sockets_.erase(handle);
      },
      message_handler(), *this, next_data_socket_handle_, *state, handle,
      nic_addr_);
}

absl::string_view DxsClient::GetServerBuildId() {
  absl::MutexLock l(&mu_);
  return server_build_id_;
}

uint64_t DxsClient::GetServerVersion() {
  absl::MutexLock l(&mu_);
  return server_version_;
}

void DxsClient::ReceiveControlMessage(absl::Span<const uint8_t> buffer) {
  if (buffer.size() < sizeof(ControlCommand)) {
    LOG(ERROR) << "recv failed.";
    return;
  }
  ControlCommand command = static_cast<ControlCommand>(buffer[0]);
  absl::MutexLock l(&mu_);
  switch (command) {
    case ControlCommand::kVersionedInitAck:
      return HandleVersionedInitAck(buffer);
    case ControlCommand::kInitLlcmAck:
      return HandleInitLlcmAck(buffer);
    case ControlCommand::kListenAck:
      return HandleListenAck(buffer);
    case ControlCommand::kIncomingConnection:
      return HandleIncomingConnection(buffer);
    case ControlCommand::kIncomingConnectionV2:
      return HandleIncomingConnectionV2(buffer);
    case ControlCommand::kConnectAck:
      return HandleConnectAck(buffer);
    case ControlCommand::kAcceptAck:
      return HandleAcceptAck(buffer);
    case ControlCommand::kSendAck:
      return HandleSendAck(buffer);
    case ControlCommand::kRecvAck:
      return HandleRecvAck(buffer);
    case ControlCommand::kPong:
      return HandlePong(buffer);
    default:
      LOG(DFATAL) << absl::StrFormat("Unexpected control command from DXS: %d",
                                     command);
  }
}

void DxsClient::OnControlChannelFailure() {
  absl::MutexLock l(&mu_);
  failed_ = true;
  for (auto& [key, op] : outstanding_send_ops_) {
    op->result.store(
        {OpStatus::kError, SendAckMessage::Status::kControlChannelFailure},
        std::memory_order_relaxed);
  }
  for (auto& [key, op] : outstanding_linearized_recv_ops_) {
    op->result.store(
        {OpStatus::kError, RecvAckMessage::Status::kControlChannelFailure},
        std::memory_order_relaxed);
  }
  for (auto& [key, sock] : outstanding_data_sockets_) {
    sock->Store(DataSockStatus::kInternalError);
  }
  for (auto& [key, sock] : outstanding_listen_sockets_) {
    sock->port_.Store(-1);
  }
}

void BufferManager::ReceiveControlMessage(absl::Span<const uint8_t> buffer) {
  if (buffer.size() < sizeof(ControlCommand)) {
    LOG(DFATAL) << "recv failed.";
    return;
  }
  ControlCommand command = static_cast<ControlCommand>(buffer[0]);
  absl::MutexLock l(&mu_);
  switch (command) {
    case ControlCommand::kVersionedInitAck:
      return HandleVersionedInitAck(buffer);
    case ControlCommand::kRegBufferAck:
      return HandleRegBufferAck(buffer);
    case ControlCommand::kDeregBufferAck:
      return HandleDeregBufferAck(buffer);
    case ControlCommand::kPeriodicStatsUpdate:
      return HandlePeriodicStatsUpdate(buffer);
    default:
      LOG(DFATAL) << absl::StrFormat("Unexpected control command from DXS: %d",
                                     command);
  }
}

void BufferManager::OnControlChannelFailure() {
  absl::MutexLock l(&mu_);
  failed_ = true;
}

bool BufferManager::HealthCheck() const {
  return message_handler().HealthCheck();
}

absl::string_view BufferManager::GetServerBuildId() {
  absl::MutexLock l(&mu_);
  return server_build_id_;
}

uint64_t BufferManager::GetServerVersion() {
  absl::MutexLock l(&mu_);
  return server_version_;
}

namespace {

template <typename MessageT>
const MessageT* ValidateAndGetMessageWithTail(
    absl::Span<const uint8_t> packet) {
  constexpr uint64_t kExpectedLength = sizeof(MessageT);
  if (packet.size() < kExpectedLength) {
    LOG(DFATAL) << absl::StrFormat(
        "Incorrect size for message. Expected at least: %d Actual: %d",
        kExpectedLength, packet.size());
    return nullptr;
  }
  return reinterpret_cast<const MessageT*>(packet.data());
}

}  // namespace

void DxsClient::HandleVersionedInitAck(absl::Span<const uint8_t> packet) {
  if (initialized_) {
    LOG(ERROR) << absl::StrFormat(
        "Received kVersionedInitAck after initialization");
    return;
  }
  auto* message = ValidateAndGetMessage<VersionedInitAckMessage>(packet);
  if (message == nullptr) return;

  if (message->status != InitAckMessage::Status::kOk) {
    LOG(ERROR) << absl::StrFormat("Failed to initialize DXS: %s",
                                  ToString(message->status));
    LOG(ERROR) << absl::StrFormat("Detailed Status : %s",
                                  GetDetailedStatus(message->status));
    failed_ = true;
    return;
  }

  server_build_id_.assign(message->server_build_id,
                          sizeof(message->server_build_id));
  server_build_id_.erase(
      std::find(server_build_id_.begin(), server_build_id_.end(), '\0'),
      server_build_id_.end());

  server_version_ = message->server_version;
  if (server_version_ < kMinimumWireVersion) {
    LOG(ERROR) << "Server version is unsupported";
    failed_ = true;
    return;
  }

  initialized_ = true;
}

void DxsClient::HandleInitLlcmAck(absl::Span<const uint8_t> packet) {
  if (llcm_info_.has_value()) {
    LOG(ERROR) << "Received redundant kInitLlcmAck";
    return;
  }

  auto* message = ValidateAndGetMessage<InitLlcmAckMessage>(packet);
  if (message == nullptr) return;

  if (message->status != InitLlcmAckMessage::Status::kOk) {
    LOG(ERROR) << absl::StrFormat("Failed to initialize LLCM: %s",
                                  ToString(message->status));
    failed_ = true;
    return;
  }

  llcm_info_.emplace(LlcmInfo{
      .llcm_queue_offset = message->llcm_queue_offset,
      .llcm_queue_size = message->llcm_queue_size,
      .reverse_llcm_queue_offset = message->reverse_llcm_queue_offset,
      .reverse_llcm_queue_size = message->reverse_llcm_queue_size,
  });
}

void BufferManager::HandleVersionedInitAck(absl::Span<const uint8_t> packet) {
  if (initialized_) {
    LOG(ERROR) << absl::StrFormat(
        "Received kVersionedInitAck after initialization");
    return;
  }
  auto* message = ValidateAndGetMessage<VersionedInitAckMessage>(packet);
  if (message == nullptr) return;

  if (message->status != InitAckMessage::Status::kOk) {
    LOG(ERROR) << absl::StrFormat("Failed to initialize DXS: %s",
                                  ToString(message->status));
    LOG(ERROR) << absl::StrFormat("Detailed Status : %s",
                                  GetDetailedStatus(message->status));
    failed_ = true;
    return;
  }
  initialized_ = true;
  server_build_id_ = message->server_build_id;
  server_version_ = message->server_version;
}

void DxsClient::HandleListenAck(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<ListenAckMessage>(packet);
  if (message == nullptr) return;
  auto it = outstanding_listen_sockets_.find(message->sock);
  if (it == outstanding_listen_sockets_.end()) {
    LOG(INFO) << absl::StrFormat(
        "Received ack for nonexistent listen socket: %d", message->sock);
    return;
  }
  it->second->port_.Store(message->port);
}

void DxsClient::HandleIncomingConnection(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<IncomingConnectionMessage>(packet);
  if (message == nullptr) return;
  auto it = outstanding_listen_sockets_.find(message->sock);
  if (it == outstanding_listen_sockets_.end()) {
    LOG(INFO) << absl::StrFormat(
        "Received connection for nonexistent listen socket: %d", message->sock);
    return;
  }
  ListenSocket::SharedState& state = *it->second;
  if (state.port_.Load() <= 0) return;
  state.AddPendingConnection(WireSocketAddr{});
}

void DxsClient::HandleIncomingConnectionV2(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<IncomingConnectionMessageV2>(packet);
  if (message == nullptr) return;
  auto it = outstanding_listen_sockets_.find(message->sock);
  if (it == outstanding_listen_sockets_.end()) {
    LOG(INFO) << absl::StrFormat(
        "Received connection for nonexistent listen socket: %d", message->sock);
    return;
  }
  ListenSocket::SharedState& state = *it->second;
  if (state.port_.Load() <= 0) return;
  state.AddPendingConnection(message->peer_saddr);
}

void DxsClient::HandleConnectAck(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<ConnectAckMessage>(packet);
  if (message == nullptr) return;
  auto it = outstanding_data_sockets_.find(message->sock);
  if (it == outstanding_data_sockets_.end()) {
    LOG(INFO) << absl::StrFormat("Received ack for nonexistent data socket: %d",
                                 message->sock);
    return;
  }
  it->second->Store(message->status);
}

void DxsClient::HandleAcceptAck(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<AcceptAckMessage>(packet);
  if (message == nullptr) return;
  auto it = outstanding_data_sockets_.find(message->sock);
  if (it == outstanding_data_sockets_.end()) {
    LOG(INFO) << absl::StrFormat("Received ack for nonexistent data socket: %d",
                                 message->sock);
    return;
  }
  DataSockStatus sock_status;
  if (message->status == AcceptAckMessage::Status::kOk) {
    sock_status = DataSockStatus::kConnected;
  } else {
    LOG(ERROR) << "Accept failure: " << ToString(message->status);
    sock_status = DataSockStatus::kInvalid;
  }
  it->second->Store(sock_status);
}

void DxsClient::HandleSendAck(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<SendAckMessage>(packet);
  if (message == nullptr) return;
  auto outstanding_op = outstanding_send_ops_.find(message->op_id);
  if (outstanding_op == outstanding_send_ops_.end()) {
    LOG(ERROR) << absl::StrFormat("Received completion for unknown send op: %d",
                                  message->op_id);
    return;
  }

  SendOp::SharedState& send_op_info = *outstanding_op->second;

  if (message->status == dxs::OpStatus::kComplete) {
    send_op_info.completion_time = GetMonotonicTs();
  }
  send_op_info.result.store({message->status, message->req_status},
                            std::memory_order_release);
}

void DxsClient::HandleRecvAck(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<RecvAckMessage>(packet);
  if (message == nullptr) return;
  auto outstanding_op = outstanding_linearized_recv_ops_.find(message->op_id);
  if (outstanding_op == outstanding_linearized_recv_ops_.end()) {
    LOG(ERROR) << absl::StrFormat("Received completion for unknown recv op: %d",
                                  message->op_id);
    return;
  }

  LinearizedRecvOp::SharedState& recv_op_info = *outstanding_op->second;
  if (message->status == dxs::OpStatus::kComplete) {
    recv_op_info.size = message->size;
    recv_op_info.completion_time = GetMonotonicTs();
  }

  // Use memory_order_release so that changes to struct members are visible
  // to the other thread.
  recv_op_info.result.store({message->status, message->req_status},
                            std::memory_order_release);
}

void DxsClient::HandlePong(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<PongMessage>(packet);
  if (message == nullptr) return;

  auto outstanding_op = outstanding_pings_.find(message->seq);
  if (outstanding_op == outstanding_pings_.end()) {
    LOG(ERROR) << absl::StrFormat("Received pong for unknown ping: %d",
                                  message->seq);
    return;
  }
  ping_results_[message->seq] = absl::Now() - outstanding_op->second;
  outstanding_pings_.erase(message->seq);
}

/*** Buffer Manager API ***/

absl::StatusOr<Reg> BufferManager::RegBuffer(absl::Span<const iovec> gpas) {
  return RegBufferWithDxsServer(gpas);
}

absl::StatusOr<Reg> BufferManager::RegBufferWithDxsServer(
    absl::Span<const iovec> gpas) {
  const uint64_t kMaxGpaPerMessage =
      (kAvailableBufferSize - sizeof(RegBufferMessage)) / sizeof(iovec);
  {
    absl::MutexLock l(&mu_);
    if (outstanding_registrations_ != 0) {
      return absl::FailedPreconditionError(
          "Registrations were left outstanding from a previous failure, cannot "
          "continue.");
    }
  }

  auto buffer = std::make_unique<uint8_t[]>(kBufferSize);
  for (auto i = 0u; i < gpas.size(); i += kMaxGpaPerMessage) {
    RegBufferMessage* message = new (buffer.get()) RegBufferMessage{
        .more = i + kMaxGpaPerMessage < gpas.size(),
        .num_gpas =
            static_cast<uint8_t>(std::min(kMaxGpaPerMessage, gpas.size() - i)),
    };

    iovec* gpa = reinterpret_cast<iovec*>(message + 1);
    for (int j = 0; j < message->num_gpas; ++j) {
      *gpa = gpas[i + j];
      ++gpa;
    }

    {
      absl::MutexLock l(&mu_);
      ++outstanding_registrations_;
    }
    RETURN_IF_ERROR(message_handler().SendControlMessage(absl::MakeSpan(
                        buffer.get(), sizeof(RegBufferMessage) +
                                          message->num_gpas * sizeof(iovec))))
            .LogError()
        << "outstanding_registrations_ may be > 1, "
           "this error unrecoverable; caller should "
           "delete and recreate the DxsClient.";
  }

  // Now wait for outstanding_registrations_ to complete before returning.
  auto done = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return failed_ || outstanding_registrations_ == 0;
  };
  absl::MutexLock l(&mu_);
  if (!mu_.AwaitWithTimeout(
          absl::Condition(&done),
          absl::GetFlag(FLAGS_dxs_client_buffer_register_timeout))) {
    return absl::DeadlineExceededError(absl::StrCat(
        "Buffer registration timed out. Client borked: ",
        failed_ ? "true" : "false", ", num outstanding buffer registrations: ",
        outstanding_registrations_));
  }

  if (failed_) {
    return absl::InternalError(
        "outstanding_registrations_ may be > 0, this error is unrecoverable; "
        "caller should delete and recreate the DxsClient.");
  }

  if (last_reg_buffer_ack_.status != RegBufferAckMessage::Status::kOk) {
    // Either client/server are out of sync or server rejected the
    // registration.
    return absl::InternalError(absl::StrFormat(
        "Buffer ack failed with status:%d reg_handle:%d last ack num_gpas: %d",
        last_reg_buffer_ack_.status, last_reg_buffer_ack_.reg_handle,
        last_reg_buffer_ack_.num_gpas));
  }

  uint64_t reg_handle = last_reg_buffer_ack_.reg_handle;

  return reg_handle;
}

void BufferManager::HandleRegBufferAck(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<RegBufferAckMessage>(packet);
  if (message == nullptr) return;
  if (outstanding_registrations_ <= 0) {
    LOG(ERROR) << absl::StrFormat(
        "Recv unexpected RegBufferAck: num_gpas: %d, reg_handle: %d",
        message->num_gpas, message->reg_handle);
    return;
  }
  if (message->status != RegBufferAckMessage::Status::kOk) {
    LOG(ERROR) << absl::StrFormat("Recv error RegBufferAck: %s",
                                  ToString(message->status));
    LOG(ERROR) << absl::StrFormat("Detailed Status : %s",
                                  GetDetailedStatus(message->status));
  }
  --outstanding_registrations_;
  last_reg_buffer_ack_ = *message;
}

absl::Status BufferManager::DeregBuffer(Reg reg_handle) {
  return DeregBufferWithDxsServer(reg_handle);
}

absl::Status BufferManager::DeregBufferWithDxsServer(Reg reg_handle) {
  {
    absl::MutexLock l(&mu_);
    if (awaiting_unreg_buffer_) {
      return absl::FailedPreconditionError(
          "DeregBuffer is already in progress.");
    }

    // Sync messages are a little cumbersome, requiring a couple member
    // variables.
    awaiting_unreg_buffer_ = true;
  }

  // Prepare a message to send to DXS.
  DeregBufferMessage message;
  message.reg_handle = reg_handle;

  RETURN_IF_ERROR(message_handler().SendMessage(&message));

  auto done = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return failed_ || !awaiting_unreg_buffer_;
  };
  absl::MutexLock l(&mu_);
  if (!mu_.AwaitWithTimeout(
          absl::Condition(&done),
          absl::GetFlag(FLAGS_dxs_client_buffer_deregister_timeout))) {
    return absl::DeadlineExceededError(absl::StrCat(
        "DeregBuffer timed out. Client borked: ", failed_ ? "true" : "false",
        ", buffer unregistration in progress: ",
        awaiting_unreg_buffer_ ? "true" : "false"));
  };

  if (failed_) return absl::InternalError("BufferManager failed.");
  if (last_dereg_buffer_ack_.status != DeregBufferAckMessage::Status::kOk) {
    return absl::InternalError("DeregBuffer failed: " +
                               ToString(last_dereg_buffer_ack_.status));
  }
  return absl::OkStatus();
}

void BufferManager::HandleDeregBufferAck(absl::Span<const uint8_t> packet) {
  auto* message = ValidateAndGetMessage<DeregBufferAckMessage>(packet);
  if (message == nullptr) return;
  if (!awaiting_unreg_buffer_) {
    LOG(ERROR) << absl::StrFormat("Recv unexpected DeregBufferAck: %s",
                                  ToString(message->status));
    LOG(ERROR) << absl::StrFormat("Detailed Status : %s",
                                  GetDetailedStatus(message->status));
    return;
  }
  last_dereg_buffer_ack_ = *message;
  awaiting_unreg_buffer_ = false;
}

void BufferManager::HandlePeriodicStatsUpdate(
    absl::Span<const uint8_t> packet) {
  auto* message =
      ValidateAndGetMessageWithTail<PeriodicStatsUpdateMessage>(packet);
  if (message == nullptr) {
    LOG(WARNING) << "Failed to parse PeriodicStatsUpdateMessage";
    return;
  }

  auto stats_or = GetPayloadProto<PeriodicStatsUpdate>(*message);
  if (!stats_or.ok()) {
    LOG(WARNING) << "Failed to parse PeriodicStatsUpdateMessage";
    return;
  }

  // Synchronously invoke user's callback.
  periodic_stats_handler_(*stats_or);
}

}  // namespace dxs

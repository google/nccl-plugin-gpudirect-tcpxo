/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_DXS_CLIENT_H_
#define DXS_CLIENT_DXS_CLIENT_H_

#include <stdint.h>
#include <sys/socket.h>
#include <sys/uio.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dxs/client/atomic-future.h"
#include "dxs/client/control-command.h"
#include "dxs/client/control-command.pb.h"
#include "dxs/client/control-message-handler-interface.h"
#include "dxs/client/data-sock.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/linearized-recv-op.h"
#include "dxs/client/listen-socket.h"
#include "dxs/client/llcm-memory-interface.h"
#include "dxs/client/relaxed-atomic.h"
#include "dxs/client/send-op.h"
#include "dxs/client/sequence-number.h"

ABSL_DECLARE_FLAG(absl::Duration,
                  TEST_ONLY_dxs_client_default_periodic_stats_interval);
ABSL_DECLARE_FLAG(absl::Duration, buffer_manager_init_timeout);
ABSL_DECLARE_FLAG(absl::Duration, dxs_client_init_timeout);
ABSL_DECLARE_FLAG(absl::Duration, dxs_client_storage_read_timeout);
ABSL_DECLARE_FLAG(absl::Duration, dxs_client_storage_write_timeout);
ABSL_DECLARE_FLAG(absl::Duration, dxs_client_buffer_register_timeout);
ABSL_DECLARE_FLAG(absl::Duration, dxs_client_buffer_deregister_timeout);

namespace dxs {

inline constexpr absl::string_view kLlcmDeviceDirectory =
    "/sys/bus/pci/devices";

class DxsClient : public DxsClientInterface,
                  public ControlMessageReceiverInterface,
                  public SocketRegistryInterface {
 public:
  ~DxsClient() override;

  // Create a new DxsClient.
  //
  // nic_addr is the address of the NIC this client sends data for.
  static absl::StatusOr<std::unique_ptr<DxsClientInterface>> Create(
      std::string nic_addr, bool enable_llcm = false,
      std::string llcm_device_directory = std::string(kLlcmDeviceDirectory),
      bool send_close_on_teardown = true);

  // nic_addr is the address of the NIC this client sends data for.
  // dxs_addr is the address of the dxs server to connect to.
  // dxs_port is the port of the dxs server to connect to.
  // source_port is the local port on nic_addr to use for outbound sockets.
  static absl::StatusOr<std::unique_ptr<DxsClientInterface>> Create(
      std::string nic_addr, std::string dxs_addr, std::string dxs_port,
      std::string source_port, bool enable_llcm = false,
      std::string llcm_device_directory = std::string(kLlcmDeviceDirectory),
      bool send_close_on_teardown = true);

  // nic_addr is the address of the NIC this client sends data for.
  // dxs_addr is the address of the dxs server to connect to.
  // dxs_port is the port of the dxs server to connect to.
  // source_port is the local port on nic_addr to use for outbound sockets.
  static absl::StatusOr<std::unique_ptr<DxsClientInterface>>
  CreateWithLlcmMemory(
      std::string nic_addr, std::string dxs_addr, std::string dxs_port,
      std::string source_port,
      std::unique_ptr<LlcmMemoryInterface> llcm_memory = nullptr,
      bool send_close_on_teardown = true);

  absl::Status Shutdown(absl::Duration timeout) override {
    return message_handler().Shutdown(timeout);
  }

  absl::StatusOr<std::unique_ptr<ListenSocketInterface>> Listen() override;

  absl::StatusOr<std::unique_ptr<SendSocketInterface>> Connect(
      std::string addr, uint16_t port) override;

  absl::StatusOr<absl::Duration> Ping() override;

  // Returns the server's build ID.
  absl::string_view GetServerBuildId() override ABSL_LOCKS_EXCLUDED(mu_);

  // Returns the server version, or zero iff the client is not connected to DXS.
  uint64_t GetServerVersion() override ABSL_LOCKS_EXCLUDED(mu_);

  // ControlMessageReceiverInterface implementation
  void ReceiveControlMessage(absl::Span<const uint8_t> buffer) override;
  void OnControlChannelFailure() override;

  // SocketRegistryInterface implementation
  std::unique_ptr<SendOpInterface> RegisterSendOp(OpId id) override
      ABSL_LOCKS_EXCLUDED(mu_);
  std::unique_ptr<LinearizedRecvOpInterface> RegisterLinearizedRecvOp(
      DataSocketHandle handle, OpId id, uint64_t size) override
      ABSL_LOCKS_EXCLUDED(mu_);
  std::unique_ptr<RecvSocketInterface> RegisterRecvSocket(
      DataSocketHandle handle, WireSocketAddr peer) override
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  friend class DxsClientTestPeer;

  struct LlcmInfo {
    uint64_t llcm_queue_offset;
    uint64_t llcm_queue_size;
    uint64_t reverse_llcm_queue_offset;
    uint64_t reverse_llcm_queue_size;
  };

  explicit DxsClient(std::string nic_addr,
                     std::unique_ptr<LlcmMemoryInterface> llcm_memory,
                     bool send_close_on_teardown)
      : nic_addr_(std::move(nic_addr)),
        llcm_memory_(std::move(llcm_memory)),
        send_close_on_teardown_(send_close_on_teardown) {}
  absl::Status Init(std::string dxs_addr, std::string dxs_port,
                    std::string source_port);

  static absl::StatusOr<std::unique_ptr<DxsClientInterface>> CreateInternal(
      std::string nic_addr, std::string dxs_addr, std::string dxs_port,
      std::string source_port, std::unique_ptr<LlcmMemoryInterface> llcm_memory,
      bool send_close_on_teardown);

  std::unique_ptr<SendSocketInterface> RegisterSendSocket(
      DataSocketHandle handle, WireSocketAddr peer) ABSL_LOCKS_EXCLUDED(mu_);
  std::unique_ptr<ListenSocketInterface> RegisterListenSocket(
      ListenSocketHandle handle) ABSL_LOCKS_EXCLUDED(mu_);

  // Handlers for various DXS messages. Return 0 on success, -1 on failure.
  void HandleVersionedInitAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleInitLlcmAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleListenAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleIncomingConnection(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleIncomingConnectionV2(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleConnectAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleAcceptAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleSendAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleRecvAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandlePong(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  DataSocketHandle NextDataSocketHandle() {
    return DataSocketHandle{next_data_socket_handle_.Next()};
  }
  ListenSocketHandle NextListenSocketHandle() {
    return ListenSocketHandle{next_listen_socket_handle_.Next()};
  }

  const std::string nic_addr_;
  std::unique_ptr<LlcmMemoryInterface> llcm_memory_;
  const bool send_close_on_teardown_;

  SequenceNumber next_op_id_{1000};
  SequenceNumber next_data_socket_handle_{1};
  SequenceNumber next_listen_socket_handle_{1};

  struct ReadState {
    uint8_t* data = nullptr;
    std::optional<absl::StatusOr<uint64_t>> size_out;
  };

  absl::Mutex mu_;
  absl::flat_hash_map<OpId, std::unique_ptr<SendOp::SharedState>>
      outstanding_send_ops_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<OpId, std::unique_ptr<LinearizedRecvOp::SharedState>>
      outstanding_linearized_recv_ops_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<OpId, ReadState> outstanding_read_ops_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<OpId, std::optional<absl::Status>> outstanding_write_ops_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<DataSocketHandle,
                      std::unique_ptr<RelaxedAtomic<DataSockStatus>>>
      outstanding_data_sockets_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<ListenSocketHandle,
                      std::unique_ptr<ListenSocket::SharedState>>
      outstanding_listen_sockets_ ABSL_GUARDED_BY(mu_);

  // True iff we have received an init ack from DXS.
  bool initialized_ ABSL_GUARDED_BY(mu_) = false;
  std::string server_build_id_ ABSL_GUARDED_BY(mu_) = "<unknown>";
  uint64_t server_version_ ABSL_GUARDED_BY(mu_) = 0;
  bool failed_ ABSL_GUARDED_BY(mu_) = false;

  std::optional<LlcmInfo> llcm_info_ ABSL_GUARDED_BY(mu_);

  uint32_t ping_seq_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_map<uint32_t, absl::Time> outstanding_pings_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<uint32_t, absl::Duration> ping_results_
      ABSL_GUARDED_BY(mu_);

  ControlMessageHandlerInterface& message_handler() ABSL_LOCKS_EXCLUDED(mu_) {
    return *message_handler_do_not_access_directly_;
  }
  std::unique_ptr<ControlMessageHandlerInterface>
      message_handler_do_not_access_directly_;
  // Hostname reported by DXS during authentication.
  std::string authenticated_dxs_hostname_;
};

struct PeriodicStatsOptions {
  absl::Duration periodic_stats_interval =
      absl::GetFlag(FLAGS_TEST_ONLY_dxs_client_default_periodic_stats_interval);
  std::function<void(const PeriodicStatsUpdate&)> handler;
};

struct BufferManagerOptions {
  std::string source_addr;
  std::string source_port = std::to_string(kBufferManagerSourcePort);
  std::optional<std::string> dest_addr_override;
  std::string dest_port = kDefaultDxsPort;
  std::optional<PeriodicStatsOptions> periodic_stats_options;
};

class BufferManager : public BufferManagerInterface,
                      public ControlMessageReceiverInterface {
 public:
  // Create a new BufferManager.
  static absl::StatusOr<std::unique_ptr<BufferManagerInterface>> Create(
      BufferManagerOptions options);
  static absl::StatusOr<std::unique_ptr<BufferManagerInterface>> Create(
      std::string nic_addr);
  static absl::StatusOr<std::unique_ptr<BufferManagerInterface>> Create(
      std::string nic_addr, std::string dxs_addr, std::string dxs_port,
      std::string source_port);

  absl::StatusOr<Reg> RegBuffer(absl::Span<const iovec> gpas) override;

  absl::Status DeregBuffer(Reg reg_handle) override;

  // ControlMessageReceiverInterface implementation
  void ReceiveControlMessage(absl::Span<const uint8_t> buffer) override;
  void OnControlChannelFailure() override;

  bool HealthCheck() const override;

  // Returns the server's build ID.
  absl::string_view GetServerBuildId() override ABSL_LOCKS_EXCLUDED(mu_);

  // Returns the server version, or zero iff the client is not connected to DXS.
  uint64_t GetServerVersion() override ABSL_LOCKS_EXCLUDED(mu_);

 private:
  BufferManager() = default;
  absl::Status Init(std::string source_addr, std::string source_port,
                    std::string dest_addr, std::string dest_port,
                    std::optional<PeriodicStatsOptions> periodic_stats_options);
  // Implementation for (de)registering buffers with DxsServer.
  absl::StatusOr<Reg> RegBufferWithDxsServer(absl::Span<const iovec> gpas);
  absl::Status DeregBufferWithDxsServer(Reg reg_handle);

  // Handlers for various DXS messages. Return 0 on success, -1 on failure.
  void HandleVersionedInitAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleRegBufferAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleDeregBufferAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleInitLlcmAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandleInitReverseLlcmAck(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void HandlePeriodicStatsUpdate(absl::Span<const uint8_t> packet)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Llcm information.
  uint32_t bar_ = 0;
  uint64_t offset_ = 0;
  uint64_t challenge_ = 0;
  uint64_t num_4kib_ = 0;

  absl::Mutex mu_;
  // True iff we have received an init ack from DXS.
  bool initialized_ ABSL_GUARDED_BY(mu_) = false;
  bool failed_ ABSL_GUARDED_BY(mu_) = false;
  std::string server_build_id_ ABSL_GUARDED_BY(mu_) = "<unknown>";
  uint64_t server_version_ ABSL_GUARDED_BY(mu_) = 0;

  int outstanding_registrations_ ABSL_GUARDED_BY(mu_) = 0;
  RegBufferAckMessage last_reg_buffer_ack_ ABSL_GUARDED_BY(mu_);

  bool awaiting_unreg_buffer_ ABSL_GUARDED_BY(mu_) = false;
  DeregBufferAckMessage last_dereg_buffer_ack_ ABSL_GUARDED_BY(mu_);

  bool llcm_got_response_ ABSL_GUARDED_BY(mu_) = false;
  bool reverse_llcm_got_response_ ABSL_GUARDED_BY(mu_) = false;
  bool llcm_init_success_ ABSL_GUARDED_BY(mu_) = false;
  bool reverse_llcm_init_success_ ABSL_GUARDED_BY(mu_) = false;

  std::function<void(const PeriodicStatsUpdate&)> periodic_stats_handler_
      ABSL_GUARDED_BY(mu_);

  const ControlMessageHandlerInterface& message_handler() const
      ABSL_LOCKS_EXCLUDED(mu_) {
    return *message_handler_do_not_access_directly_;
  }

  ControlMessageHandlerInterface& message_handler() ABSL_LOCKS_EXCLUDED(mu_) {
    return *message_handler_do_not_access_directly_;
  }
  std::unique_ptr<ControlMessageHandlerInterface>
      message_handler_do_not_access_directly_;
};

// Construct a new ControlMessageHandler.
//
// `receiver` must outlive the returned object.
absl::StatusOr<std::unique_ptr<ControlMessageHandlerInterface>>
CreateControlMessageHandler(
    std::string nic_addr, std::string dxs_addr, std::string dxs_port,
    std::string source_port, bool is_buffer_manager, bool enable_llcm,
    ABSL_ATTRIBUTE_LIFETIME_BOUND ControlMessageReceiverInterface& receiver);

}  // namespace dxs

#endif  // DXS_CLIENT_DXS_CLIENT_H_

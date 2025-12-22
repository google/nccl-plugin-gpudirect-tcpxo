/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpxo_prober/src/nic-manager.h"

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client.h"
#include "cuda_helpers/cuda_helpers.h"
#include "cuda_helpers/gpu_mem_helper_interface.h"
#include "cuda_helpers/gpu_mem_validator_interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/dxs-client.h"
#include "dxs/client/oss/status_macros.h"
#include "tcpxo_prober/src/connection-manager.grpc.pb.h"
#include "tcpxo_prober/src/connection-manager.pb.h"
#include "tcpxo_prober/src/connection.h"
#include "tcpxo_prober/src/prober.pb.h"

using ::cuda_helpers::find_gpu_pci_for_ip;
using ::cuda_helpers::GpuMemHelper;
using ::cuda_helpers::GpuMemValidator;

namespace dxs::prober {

constexpr absl::string_view kDefaultSourcePort = "0";

namespace {
// Generates and returns a DxsRR payload of size 'payload_size' bytes. Must
// be deterministic as this function is used to validate the payload on the
// receiver.
std::unique_ptr<char[]> GeneratePayload(size_t payload_size) {
  auto send_payload = std::make_unique<char[]>(payload_size);
  constexpr char kPayload[] = "Hello, DXS! ";
  for (int i = 0; i < payload_size; ++i) {
    (send_payload.get())[i] = kPayload[i % (sizeof(kPayload) - 1)];
  }
  return send_payload;
}
}  // namespace

NicManager::NicManager(Options options) : options_(options) {}

constexpr size_t kMinBufferSize = 2 << 20ull;

absl::Status NicManager::SetUp() {
  LOG(INFO) << "Setting up DXS client and buffers for NIC with IP: "
            << options_.nic_ip;
  RETURN_IF_ERROR(cuda_helpers::InitCuda());

  ASSIGN_OR_RETURN(buffer_manager_client_,
                   tcpdirect::BufferManagerClient::Create(options_.nic_ip));
  ASSIGN_OR_RETURN(std::string gpu_pci, find_gpu_pci_for_ip(options_.nic_ip));
  LOG(INFO) << "Using GPU " << gpu_pci << "...";
  mem_validator_ = std::make_unique<GpuMemValidator>(gpu_pci, false);
  mem_helper_ = std::make_unique<GpuMemHelper>(gpu_pci);
  RETURN_IF_ERROR(mem_validator_->Init());
  RETURN_IF_ERROR(mem_helper_->Init());
  ASSIGN_OR_RETURN(dxs_client_,
                   dxs::DxsClient::Create(
                       options_.nic_ip, options_.dxs_ip, options_.dxs_port,
                       std::string(kDefaultSourcePort), options_.use_llcm,
                       options_.llcm_device_directory));

  // Create a buffer and write payload to memory so we can TX data directly
  // without copying it to host memory.
  const uint64_t buffer_size = std::max(options_.payload_size, kMinBufferSize);
  ASSIGN_OR_RETURN(send_buffer_id_, mem_helper_->CreateBuffer(buffer_size));
  std::unique_ptr<char[]> payload = GeneratePayload(options_.payload_size);
  RETURN_IF_ERROR(mem_helper_->WriteBuffer(
      send_buffer_id_, static_cast<const void*>(payload.get()), 0,
      options_.payload_size));
  ASSIGN_OR_RETURN(const int send_buffer_fd,
                   mem_helper_->GetFd(send_buffer_id_));
  ASSIGN_OR_RETURN(send_buffer_handle_,
                   buffer_manager_client_->RegBuf(send_buffer_fd, buffer_size));

  // Prepare RX buffer.
  ASSIGN_OR_RETURN(recv_buffer_id_, mem_helper_->CreateBuffer(buffer_size));
  ASSIGN_OR_RETURN(const int recv_buffer_fd,
                   mem_helper_->GetFd(recv_buffer_id_));
  ASSIGN_OR_RETURN(recv_buffer_handle_,
                   buffer_manager_client_->RegBuf(recv_buffer_fd, buffer_size));

  return absl::OkStatus();
}

absl::StatusOr<int> NicManager::SetUpClient(
    const std::string& server_gpu_ip, int recv_port,
    const std::chrono::system_clock::time_point& deadline) {
  std::string grpc_address_str;
  if (!absl::StrContains(server_gpu_ip, ':')) {
    grpc_address_str = absl::StrCat(server_gpu_ip, ":", options_.grpc_port);
  } else {
    grpc_address_str =
        absl::StrCat("[", server_gpu_ip, "]", ":", options_.grpc_port);
  }
  std::shared_ptr<grpc::ChannelCredentials> creds =
      grpc::InsecureChannelCredentials();  // NOLINT
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(absl::StrCat("dns:///", grpc_address_str), creds);
  std::unique_ptr<dxs::prober::ConnectionManager::Stub> stub(
      dxs::prober::ConnectionManager::NewStub(channel));

  grpc::ClientContext context;
  context.set_deadline(deadline);
  dxs::prober::SetUpRequest request;
  request.set_client_gpu_ip(options_.nic_ip);
  request.set_client_dxs_port(recv_port);
  request.set_server_gpu_ip(server_gpu_ip);
  dxs::prober::SetUpReply reply;

  grpc::Status status = stub->SetUpPongConnection(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to set up connection: " << status.error_message();
    return absl::InternalError("Failed to set up connection");
  }
  return reply.dxs_port();
}

void NicManager::TearDownClient(const std::string& client_gpu_ip,
                                const std::string& server_gpu_ip,
                                bool ping_ip_is_servers_nic) {
  std::string grpc_address_str;
  if (!absl::StrContains(server_gpu_ip, ':')) {
    grpc_address_str = absl::StrCat(server_gpu_ip, ":", options_.grpc_port);
  } else {
    grpc_address_str =
        absl::StrCat("[", server_gpu_ip, "]", ":", options_.grpc_port);
  }
  std::shared_ptr<grpc::ChannelCredentials> creds =
      grpc::InsecureChannelCredentials();  // NOLINT
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(absl::StrCat("dns:///", grpc_address_str), creds);
  std::unique_ptr<dxs::prober::ConnectionManager::Stub> stub(
      dxs::prober::ConnectionManager::NewStub(channel));

  grpc::ClientContext context;
  context.set_deadline(
      absl::ToChronoTime(absl::Now() + options_.max_rpc_deadline));
  dxs::prober::TearDownRequest request;
  request.set_ping_ip_is_servers_nic(ping_ip_is_servers_nic);
  if (ping_ip_is_servers_nic) {
    request.set_ping_ip(server_gpu_ip);
    request.set_pong_ip(client_gpu_ip);
  } else {
    request.set_ping_ip(client_gpu_ip);
    request.set_pong_ip(server_gpu_ip);
  }
  dxs::prober::TearDownReply reply;

  grpc::Status status = stub->TearDownConnection(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to tear down connection: " << status.error_message();
  }
}

absl::StatusOr<NicManager::ListenResult> NicManager::Listen(
    const std::chrono::system_clock::time_point& deadline) {
  ListenResult res;
  ASSIGN_OR_RETURN(res.dxs_listen_sock, dxs_client_->Listen());

  // Check if DXS server listening socket is ready.
  std::optional<absl::Status> ready = res.dxs_listen_sock->SocketReady();
  while (!ready.has_value()) {
    if (std::chrono::system_clock::now() > deadline) {
      return absl::DeadlineExceededError("Listen timed out");
    }
    ready = res.dxs_listen_sock->SocketReady();
  }
  RETURN_IF_ERROR(*ready);

  // Fetch the port number that DXS is listening on.
  res.recv_port = res.dxs_listen_sock->Port();
  if (res.recv_port <= 0) {
    return absl::InternalError(absl::StrCat(
        "ListenSocket::SocketReady() failure, recv_port is invalid: ",
        res.recv_port));
  }
  return res;
}

absl::StatusOr<std::unique_ptr<SendSocketInterface>> NicManager::Connect(
    const std::string& remote_nic_ip, int send_port,
    const std::chrono::system_clock::time_point& deadline) {
  ASSIGN_OR_RETURN(std::unique_ptr<SendSocketInterface> dxs_send_sock,
                   dxs_client_->Connect(remote_nic_ip, send_port));

  // Ask the DXS on the client side to connect to the DXS on the server side.
  std::optional<absl::Status> ready = dxs_send_sock->SocketReady();
  while (!ready.has_value()) {
    if (std::chrono::system_clock::now() > deadline) {
      return absl::DeadlineExceededError("Connect timed out");
    }
    ready = dxs_send_sock->SocketReady();
  }
  RETURN_IF_ERROR(*ready);
  return dxs_send_sock;
}

absl::StatusOr<std::unique_ptr<RecvSocketInterface>> NicManager::Accept(
    ListenSocketInterface* listen_sock,
    const std::chrono::system_clock::time_point& deadline) {
  // Ask the DXS on the server side to accept the connection.
  std::unique_ptr<RecvSocketInterface> dxs_recv_sock;
  while (!dxs_recv_sock) {
    if (std::chrono::system_clock::now() > deadline) {
      return absl::DeadlineExceededError("Accept timed out");
    }
    ASSIGN_OR_RETURN(dxs_recv_sock, listen_sock->Accept());
  }
  // Check if DXS server recv socket is ready.
  std::optional<absl::Status> ready = dxs_recv_sock->SocketReady();
  while (!ready.has_value()) {
    if (std::chrono::system_clock::now() > deadline) {
      return absl::DeadlineExceededError("Accept timed out");
    }
    ready = dxs_recv_sock->SocketReady();
  };

  RETURN_IF_ERROR(*ready);
  return dxs_recv_sock;
}

absl::StatusOr<std::unique_ptr<PayloadVerificationContext>>
NicManager::CreatePayloadVerificationContext() {
  const uint64_t buffer_size = std::max(options_.payload_size, kMinBufferSize);
  ASSIGN_OR_RETURN(uint64_t recv_id, mem_helper_->CreateBuffer(buffer_size));
  // cleanup is canceled if the buffers are successfully created.
  absl::Cleanup buffer_cleanup = [recv_id, mem_helper = mem_helper_.get()] {
    mem_helper->FreeBuffer(recv_id);
  };
  ASSIGN_OR_RETURN(const int recv_buffer_fd, mem_helper_->GetFd(recv_id));
  ASSIGN_OR_RETURN(Reg recv_handle,
                   buffer_manager_client_->RegBuf(recv_buffer_fd, buffer_size));
  std::move(buffer_cleanup).Cancel();
  return std::make_unique<PayloadVerificationContext>(
      send_buffer_id_, recv_id, recv_handle, mem_helper_.get(),
      mem_validator_.get(), buffer_manager_client_.get());
}

constexpr absl::string_view CreatePingConnectionErrorMessage =
    "Failed to create an outgoing connection: ";

ConnectionResults::Result NicManager::CreatePingConnection(
    const std::string& remote_nic_ip, bool verify_payload,
    const std::chrono::system_clock::time_point& rpc_deadline) {
  absl::MutexLock lock(&mu_ping_connections_);
  LOG(INFO) << "Create outgoing connection with remote nic: " << remote_nic_ip
            << " and local NIC: " << options_.nic_ip;
  if (options_.max_connections_per_nic > 0 &&
      ping_connections_.size() >= options_.max_connections_per_nic) {
    return ConnectionResults::RESULT_FAILURE_CONNECTION_LIMIT;
  }

  std::unique_ptr<PayloadVerificationContext> payload_verification;
  Reg recv_buffer_handle = recv_buffer_handle_;
  if (verify_payload) {
    absl::StatusOr<std::unique_ptr<PayloadVerificationContext>> status_or =
        CreatePayloadVerificationContext();
    if (!status_or.ok()) {
      LOG(ERROR) << CreatePingConnectionErrorMessage << status_or.status();
      return ConnectionResults::RESULT_FAILURE;
    }
    payload_verification = std::move(*status_or);
    recv_buffer_handle = payload_verification->recv_handle;
  }

  std::chrono::system_clock::time_point deadline =
      std::min(rpc_deadline,
               absl::ToChronoTime(absl::Now() + options_.max_rpc_deadline));

  absl::StatusOr<ListenResult> listen_result = Listen(deadline);
  if (!listen_result.ok()) {
    LOG(ERROR) << CreatePingConnectionErrorMessage << listen_result.status();
    return ConnectionResults::RESULT_FAILURE;
  }

  absl::StatusOr<int> connect_port =
      SetUpClient(remote_nic_ip, listen_result->recv_port, deadline);
  if (!connect_port.ok()) {
    LOG(ERROR) << CreatePingConnectionErrorMessage << connect_port.status();
    return ConnectionResults::RESULT_FAILURE;
  }

  absl::StatusOr<std::unique_ptr<SendSocketInterface>> dxs_send_sock =
      Connect(remote_nic_ip, *connect_port, deadline);
  if (!dxs_send_sock.ok()) {
    LOG(ERROR) << CreatePingConnectionErrorMessage << dxs_send_sock.status();
    return ConnectionResults::RESULT_FAILURE;
  }

  absl::StatusOr<std::unique_ptr<RecvSocketInterface>> dxs_recv_sock =
      Accept(listen_result->dxs_listen_sock.get(), deadline);
  if (!dxs_recv_sock.ok()) {
    LOG(ERROR) << CreatePingConnectionErrorMessage << dxs_recv_sock.status();
    return ConnectionResults::RESULT_FAILURE;
  }
  ping_connections_.push_back(std::make_unique<PingConnection>(
      Connection{options_.nic_ip, remote_nic_ip, options_.payload_size,
                 send_buffer_handle_, recv_buffer_handle,
                 std::move(*dxs_send_sock), std::move(*dxs_recv_sock)},
      std::move(payload_verification)));
  return ConnectionResults::RESULT_SUCCESS;
}

absl::StatusOr<int> NicManager::CreatingPongConnection(
    const dxs::prober::SetUpRequest* request,
    const std::chrono::system_clock::time_point& deadline) {
  LOG(INFO) << "Create pong connection with remote nic: "
            << request->client_gpu_ip()
            << " and local NIC: " << options_.nic_ip;

  ASSIGN_OR_RETURN(ListenResult listen_result, Listen(deadline));
  int recv_port = listen_result.recv_port;

  // Create a detached thread to complete incoming connection setup after
  // the RPC returns local dxs port to the client.
  std::thread t([this, remote_ip_address = request->client_gpu_ip(),
                 connect_port = request->client_dxs_port(),
                 listen_result = std::move(listen_result), deadline]() {
    absl::Status status = CompletePongConnectionSetup(
        remote_ip_address, connect_port, listen_result.recv_port,
        listen_result.dxs_listen_sock.get(), deadline);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to complete pong connection setup: " << status;
    }
  });
  t.detach();
  return recv_port;
}

absl::Status NicManager::CompletePongConnectionSetup(
    const std::string& remote_nic_ip, int connect_port, int recv_port,
    ListenSocketInterface* dxs_listen_sock,
    const std::chrono::system_clock::time_point& deadline) {
  absl::MutexLock lock(&mu_pong_connections_);
  if (options_.max_connections_per_nic > 0 &&
      pong_connections_.size() >= options_.max_connections_per_nic) {
    return absl::ResourceExhaustedError(
        "Maximum number of pong connections per NIC reached.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<RecvSocketInterface> dxs_recv_sock,
                   Accept(dxs_listen_sock, deadline));

  ASSIGN_OR_RETURN(std::unique_ptr<SendSocketInterface> dxs_send_sock,
                   Connect(remote_nic_ip, connect_port, deadline));

  pong_connections_.push_back(std::make_unique<PongConnection>(
      Connection{options_.nic_ip, remote_nic_ip, options_.payload_size,
                 send_buffer_handle_, recv_buffer_handle_,
                 std::move(dxs_send_sock), std::move(dxs_recv_sock)}));
  return absl::OkStatus();
}

void NicManager::TearDownAllPingConnections() {
  absl::MutexLock lock(&mu_ping_connections_);
  for (std::unique_ptr<PingConnection>& c : ping_connections_) {
    TearDownClient(c->GetLocalIp(), c->GetPeerIp(),
                   /*ping_ip_is_servers_nic=*/false);
  }
  ping_connections_.clear();
}

void NicManager::TearDownAllPongConnections() {
  absl::MutexLock lock(&mu_pong_connections_);
  for (std::unique_ptr<PongConnection>& c : pong_connections_) {
    TearDownClient(c->GetLocalIp(), c->GetPeerIp(),
                   /*ping_ip_is_servers_nic=*/true);
  }
  pong_connections_.clear();
}

absl::Status NicManager::RemoveConnection(
    const dxs::prober::TearDownRequest* request) {
  if (request->ping_ip_is_servers_nic()) {
    std::string remote_ip = request->pong_ip();
    absl::MutexLock lock(&mu_ping_connections_);
    for (auto it = ping_connections_.begin(); it != ping_connections_.end();
         ++it) {
      if ((*it)->GetPeerIp() == remote_ip) {
        LOG(INFO) << "Tearing down ping connection with remote NIC: "
                  << remote_ip;
        ping_connections_.erase(it);
        return absl::OkStatus();
      }
    }
  } else {
    absl::MutexLock lock(&mu_pong_connections_);
    std::string remote_ip = request->ping_ip();
    for (auto it = pong_connections_.begin(); it != pong_connections_.end();
         ++it) {
      if ((*it)->GetPeerIp() == remote_ip) {
        LOG(INFO) << "Tearing down pong connection with remote NIC: "
                  << remote_ip;
        pong_connections_.erase(it);
        return absl::OkStatus();
      }
    }
  }
  return absl::NotFoundError("No connection with remote NIC found");
}

void NicManager::DoPingOps() {
  // Acquire the lock within the loop to set up and tear down connections
  // faster. This could skip or duplicate ping operations if the ping list
  // changes size, but this is is okay as DoPingOps is called frequently and
  // the ping_connections_ does not change often.
  int i = 0;
  while (true) {
    absl::MutexLock lock(&mu_ping_connections_);
    if (i >= ping_connections_.size()) {
      return;
    }
    ping_connections_[i]->SendPingAndRecordResult();
    i++;
  }
}

void NicManager::PrepareNewPingRound() {
  absl::MutexLock lock(&mu_ping_connections_);
  for (std::unique_ptr<PingConnection>& conn : ping_connections_) {
    conn->PrepareNewPingRound();
  }
}

void NicManager::DoPongOps() {
  absl::MutexLock lock(&mu_pong_connections_);
  for (std::unique_ptr<PongConnection>& conn : pong_connections_) {
    conn->DoPongOp();
  }
}

std::vector<std::string> NicManager::GetAndClearResults() {
  absl::MutexLock lock(&mu_ping_connections_);
  std::vector<std::string> results;
  for (std::unique_ptr<PingConnection>& conn : ping_connections_) {
    std::vector<std::string> conn_results = conn->GetAndClearResults();
    for (const auto& r : conn_results) {
      results.push_back(r);
    }
  }
  return results;
}

absl::Status NicManager::Cleanup() {
  LOG(INFO) << "Notifying remote agents that local agent is shutting down.";
  TearDownAllPingConnections();
  TearDownAllPongConnections();

  LOG(INFO) << "Deallocating GPU memory.";
  RETURN_IF_ERROR(buffer_manager_client_->DeregBuf(send_buffer_handle_));
  RETURN_IF_ERROR(buffer_manager_client_->DeregBuf(recv_buffer_handle_));
  mem_helper_->FreeBuffer(send_buffer_id_);
  mem_helper_->FreeBuffer(recv_buffer_id_);
  return absl::OkStatus();
}

}  // namespace dxs::prober

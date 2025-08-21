/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/fastrak_gpu_mem_importer.h"

#include <fcntl.h>
#include <linux/types.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/uio.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "buffer_mgmt_daemon/common/uds_helpers.h"
#include "buffer_mgmt_daemon/fastrak_addr_translator.h"
#include "buffer_mgmt_daemon/fastrak_buffer_resource_tracker.h"
#include "buffer_mgmt_daemon/fastrak_gpu_nic_info.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "buffer_mgmt_daemon/status_utils.h"
#include "buffer_mgmt_daemon/unix_socket_server.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h"
#include "google/rpc/status.pb.h"

namespace tcpdirect {

static constexpr uint32_t const kFtsClientMagic = 0x465453;

FasTrakGpuMemImporter::FasTrakGpuMemImporter(
    absl::Span<const tcpdirect::FasTrakGpuNicInfo> nic_infos,
    std::function<void()> all_clients_exited_callback,
    std::optional<std::string> dmabuf_import_path)
    : all_clients_exited_callback_(std::move(all_clients_exited_callback)) {
  for (auto& nic_info : nic_infos) {
    struct NicBinding binding = {
        .ip_addr = nic_info.ip_addr,
        .nic_pci_addr = nic_info.nic_pci_addr,
        .addr_translator = std::make_unique<FasTrakAddrTranslator>(
            nic_info.nic_pci_addr, dmabuf_import_path),
        .dxs_client = nic_info.dxs_client.get(),
        .resource_tracker = std::make_unique<FastrakBufferResourceTracker>(),
    };
    struct NicService service = {
        .binding = std::move(binding),
        .unix_socket_server = nullptr,
    };
    nic_services_.emplace_back(std::move(service));
  }
}

FasTrakGpuMemImporter::FasTrakGpuMemImporter(
    std::vector<NicBinding>& bindings) {
  for (auto& binding : bindings) {
    NicBinding binding_ = {
        .ip_addr = binding.ip_addr,
        .nic_pci_addr = binding.nic_pci_addr,
        .addr_translator = std::move(binding.addr_translator),
        .dxs_client = binding.dxs_client,
        .resource_tracker = std::move(binding.resource_tracker),
    };
    struct NicService service = {
        .binding = std::move(binding_),
        .unix_socket_server = nullptr,
    };
    nic_services_.emplace_back(std::move(service));
  }
}

void FasTrakGpuMemImporter::HandleRequest(
    int client, NicBinding& binding, tcpdirect::UnixSocketMessage&& request,
    tcpdirect::UnixSocketMessage* response, bool* fin) {
  *fin = false;
  tcpdirect::BufferOpReq req;
  tcpdirect::BufferOpResp resp;

  absl::Status status = absl::OkStatus();

  auto return_resp = absl::MakeCleanup([response, &resp, &status, fin] {
    if (!status.ok()) {
      LOG(WARNING) << status;
      *fin = true;
    }
    tcpdirect::SaveStatusToProto(status, resp.mutable_status());

    if (!resp.SerializeToString(response->mutable_text())) {
      LOG(WARNING)
          << "Memory importer: Failed to serialize response, abort connection.";
      *fin = true;
    }
  });

  if (!request.has_text() || !req.ParseFromString(request.text())) {
    status = absl::InvalidArgumentError(
        "Memory importer: Failed to parse request, abort connection.");
    return;
  }

  if (!req.has_fts_magic_value() || req.fts_magic_value() != kFtsClientMagic) {
    status = absl::InvalidArgumentError(
        "Memory importer: Request does not have valid authentication for DXS "
        "clients.");
    return;
  }
  if (req.op_type() != tcpdirect::REG_BUFFER &&
      req.op_type() != tcpdirect::DEREG_BUFFER) {
    status = absl::InvalidArgumentError(absl::StrFormat(
        "Memory importer: Request does not have a valid op type, needs to be "
        "one of tcpdirect::BufferOpType::REG_BUFFER(1) or "
        "tcpdirect::BufferOpType::DEREG_BUFFER(2). Actual value: %d",
        req.op_type()));
    return;
  }

  status = req.op_type() == tcpdirect::REG_BUFFER
               ? HandleRegBuffer(client, binding, request, &resp, fin)
               : HandleDeregBuffer(client, binding, req, &resp, fin);
}

absl::Status FasTrakGpuMemImporter::HandleRegBuffer(
    int client, NicBinding& binding, tcpdirect::UnixSocketMessage& request,
    tcpdirect::BufferOpResp* resp, bool* fin) {
  auto& addr_translator = binding.addr_translator;
  auto& dxs_client = binding.dxs_client;
  auto& resource_tracker = binding.resource_tracker;
  if (!request.has_fd()) {
    return absl::InvalidArgumentError(
        "Buffer registration request does not have a "
        "valid fd set.");
  }
  int dmabuf_fd = request.fd();
  auto map_status = binding.addr_translator->Map(dmabuf_fd);
  if (!map_status.status().ok()) {
    return absl::InternalError(
        absl::StrFormat("Addr translator failed to map dmabuf fd %d: %s",
                        dmabuf_fd, map_status.status().message()));
  }
  uint64_t id = *map_status;
  auto unmap_dmabuf =
      absl::MakeCleanup([id, &addr_translator] { addr_translator->Unmap(id); });
  auto status = addr_translator->GetIovecs(id);
  if (!status.status().ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to get iovecs for buffer registration request: %s",
        status.status().message()));
  }
  ASSIGN_OR_RETURN(dxs::Reg reg_handle,
                   dxs_client->RegBuffer(std::move(*status)));
  RETURN_IF_ERROR(
      resource_tracker->TrackBuffer(client, dmabuf_fd, id, reg_handle));
  resp->set_reg_handle(reg_handle);
  std::move(unmap_dmabuf).Cancel();
  return absl::OkStatus();
}

absl::Status FasTrakGpuMemImporter::HandleDeregBuffer(
    int client, NicBinding& binding, tcpdirect::BufferOpReq& req,
    tcpdirect::BufferOpResp* response, bool* fin) {
  auto& addr_translator = binding.addr_translator;
  auto& dxs_client = binding.dxs_client;
  auto& resource_tracker = binding.resource_tracker;

  if (!req.has_reg_handle()) {
    return absl::InvalidArgumentError("Missing registration handle.");
  }

  ASSIGN_OR_RETURN(auto dma_id,
                   resource_tracker->GetDmabufId(client, req.reg_handle()));
  RETURN_IF_ERROR(dxs_client->DeregBuffer(req.reg_handle()));
  addr_translator->Unmap(dma_id);

  RETURN_IF_ERROR(resource_tracker->UntrackBuffer(client, req.reg_handle()));
  return absl::OkStatus();
}

void FasTrakGpuMemImporter::CleanUp(int client, NicBinding& binding) {
  auto& addr_translator = binding.addr_translator;
  auto& dxs_client = binding.dxs_client;
  auto& resource_tracker = binding.resource_tracker;

  auto reg_handles = resource_tracker->GetRegHandles(client);
  if (!reg_handles.ok()) {
    // If the client never registered any buffers, resource_tracker will return
    // kNotFound, which is safe to ignore.
    if (absl::IsNotFound(reg_handles.status())) {
      LOG(INFO) << "Client " << client
                << " has no registered buffers to deregister.";
      return;
    }
    LOG(ERROR) << reg_handles.status();
    return;
  }

  for (auto& reg_handle : *reg_handles) {
    absl::Status s = dxs_client->DeregBuffer(reg_handle);
    if (!s.ok()) {
      LOG(ERROR) << "Memory importer: Failed to deregister buffer with DXS: "
                 << "DeregSendBuffer returned " << s;
    }
    auto dma_id = resource_tracker->GetDmabufId(client, reg_handle);
    if (!dma_id.ok()) {
      LOG(ERROR) << "Failed to get dma_id for reg_handle: " << reg_handle
                 << ": " << dma_id.status();
    }

    addr_translator->Unmap(*dma_id);
    auto status = resource_tracker->UntrackBuffer(client, reg_handle);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to untrack buffer for client: " << client << ": "
                 << status;
    }
  }

  resource_tracker->UnregisterClient(client);
  VLOG(1) << "Unregistered client: " << client;
}

absl::Status FasTrakGpuMemImporter::InitializeServers() {
  for (auto& service : nic_services_) {
    auto& binding = service.binding;
    service.unix_socket_server = std::make_unique<tcpdirect::UnixSocketServer>(
        tcpdirect::BufOpUdsPath(binding.ip_addr),
        [&binding, this](int client, tcpdirect::UnixSocketMessage&& request,
                         tcpdirect::UnixSocketMessage* response, bool* fin) {
          this->AddConnectedClient(client, binding);
          HandleRequest(client, binding, std::move(request), response, fin);
        },
        [&binding, this](int client) {
          RemoveConnectedClient(client, binding);
        });
  }
  return absl::OkStatus();
}

void FasTrakGpuMemImporter::AddConnectedClient(int client,
                                               NicBinding& binding) {
  binding.resource_tracker->RegisterClient(client);
  absl::MutexLock lock(&mutex_);
  auto [_, inserted] = connected_clients_.insert(client);
  if (!inserted) {
    return;
  }
  LOG(INFO) << "Memory importer: Added client: " << client;
}

void FasTrakGpuMemImporter::RemoveConnectedClient(int client,
                                                  NicBinding& binding) {
  CleanUp(client, binding);
  absl::MutexLock lock(&mutex_);
  connected_clients_.erase(client);
  LOG(INFO) << "Memory importer: Removed client: " << client;

  if (connected_clients_.empty() && all_clients_exited_callback_ != nullptr) {
    LOG(INFO) << "Memory importer: Cleanup callback called due to no active "
                 "DXS connection.";
    all_clients_exited_callback_();
  }
}

absl::Status FasTrakGpuMemImporter::Initialize() {
  for (auto& service : nic_services_) {
    if (!service.binding.addr_translator->Init()) {
      return absl::UnavailableError(absl::StrFormat(
          "Memory importer: failed to initialize addr translator on NIC IP %s.",
          service.binding.ip_addr));
    }
  }
  auto status = InitializeServers();
  if (!status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to initialize unix socket servers for memory importer: %s",
        status.message()));
  }
  return absl::OkStatus();
}

absl::Status FasTrakGpuMemImporter::Start() {
  LOG(INFO) << "Starting Unix socket servers ...";
  for (auto& service : nic_services_) {
    if (service.unix_socket_server == nullptr) {
      return absl::UnavailableError(
          absl::StrFormat("Failed to start memory importer: Unix socket "
                          "servers on NIC IP %s are not "
                          "initialized yet.",
                          service.binding.ip_addr));
    }
    auto status = service.unix_socket_server->Start();
    if (!status.ok()) {
      return absl::InternalError(
          absl::StrFormat("Failed to start Unix socket server on NIC IP %s for "
                          "memory importer: %s",
                          service.binding.ip_addr, status.message()));
    }
  }
  LOG(INFO) << "Memory import servers started ...";
  return absl::OkStatus();
}

FasTrakGpuMemImporter::~FasTrakGpuMemImporter() {
  for (auto& service : nic_services_) {
    if (service.unix_socket_server != nullptr) {
      service.unix_socket_server->Stop();
    }
    auto& resource_tracker = service.binding.resource_tracker;

    // Cleanup resources for each client
    for (auto& client : resource_tracker->GetClients()) {
      CleanUp(client, service.binding);
    }
  }
}

}  // namespace tcpdirect

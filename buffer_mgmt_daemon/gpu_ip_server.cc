/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/gpu_ip_server.h"

#include <memory>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "buffer_mgmt_daemon/common/uds_helpers.h"
#include "buffer_mgmt_daemon/gpu_rxq_configurator_interface.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "buffer_mgmt_daemon/status_utils.h"
#include "buffer_mgmt_daemon/unix_socket_server.h"
#include "google/rpc/status.pb.h"

namespace tcpdirect {

GpuIpServer::GpuIpServer(
    absl::Span<const tcpdirect::GpuRxqConfiguration> configs)
    : configs_(configs.begin(), configs.end()) {}

namespace {

bool HandleGetNicIp(const GpuRxqConfiguration& config,
                    tcpdirect::UnixSocketMessage& request,
                    tcpdirect::UnixSocketMessage* response) {
  tcpdirect::GetNicIpReq req;
  tcpdirect::GetNicIpResp resp;

  absl::Status status = absl::OkStatus();

  auto return_resp = absl::MakeCleanup([response, &resp, &status] {
    if (!status.ok()) {
      LOG(WARNING) << status;
    }
    tcpdirect::SaveStatusToProto(status, resp.mutable_status());

    if (!resp.SerializeToString(response->mutable_text())) {
      LOG(WARNING) << "GPU-to-IP server: Failed to serialize response, abort "
                      "connection.";
    }
  });

  if (!request.has_text() || !req.ParseFromString(request.text())) {
    status = absl::InvalidArgumentError(
        "GPU-to-IP server: Failed to parse request, abort connection.");
    return false;
  }
  if (!req.has_gpu_pci()) {
    status = absl::InvalidArgumentError(
        "GPU-to-IP server: Request does not have GPU PCI set.");
    return true;
  }
  if (req.gpu_pci() != config.gpu_pci_addr) {
    status = absl::InvalidArgumentError(absl::StrFormat(
        "GPU-to-IP server: Request's GPU PCI: %s\n does not match the GPU PCI "
        "%s\n ",
        req.gpu_pci(), config.gpu_pci_addr));
    return true;
  }
  resp.set_nic_ip(config.ip_addr);
  return true;
}

bool HandleGetNicMapping(
    absl::Span<const tcpdirect::GpuRxqConfiguration> configs,
    tcpdirect::UnixSocketMessage& request,
    tcpdirect::UnixSocketMessage* response) {
  tcpdirect::GetNicMappingReq req;
  tcpdirect::GetNicMappingResp resp;
  // Can't check for the has-bit of `text` since protocol doesn't preserve field
  // presence if text is empty but set.
  if (request.fd() != 0) {
    LOG(WARNING) << "GPU-to-IP server: GetNicMapping UnixSocketMessage request "
                    "doesn't contain only text, abort connection.";
    return false;
  }
  if (!req.ParseFromString(request.text())) {
    LOG(WARNING) << "GPU-to-IP server: Failed to parse request, abort "
                    "connection. Request: "
                 << request;
    return false;
  }
  for (auto& config : configs) {
    (*resp.mutable_pci_nic_map())[config.gpu_pci_addr].add_closest_nic_ip(
        config.ip_addr);
  }
  response->set_text(resp.SerializeAsString());
  return true;
}
}  // namespace

absl::Status GpuIpServer::Initialize() {
  for (auto& config : configs_) {
    nic_services_.push_back(std::make_unique<tcpdirect::UnixSocketServer>(
        tcpdirect::NicIpUdsPath(config.gpu_pci_addr),
        [&](int /*client*/, tcpdirect::UnixSocketMessage&& request,
            tcpdirect::UnixSocketMessage* response,
            bool* fin) { *fin = !HandleGetNicIp(config, request, response); }));
  }
  nic_services_.push_back(std::make_unique<tcpdirect::UnixSocketServer>(
      tcpdirect::NicMappingUdsPath(),
      [&](int /*client*/, tcpdirect::UnixSocketMessage&& request,
          tcpdirect::UnixSocketMessage* response, bool* fin) {
        *fin = !HandleGetNicMapping(configs_, request, response);
      }));
  return absl::OkStatus();
}

absl::Status GpuIpServer::Start() {
  LOG(INFO) << "Starting Unix socket servers.";
  for (auto& service : nic_services_) {
    auto start_status = service->Start();
    if (!start_status.ok()) {
      return absl::InternalError(absl::StrFormat(
          "GPU-to-IP server: Failed to start Unix socket servers: %s",
          start_status.message()));
    }
  }
  return absl::OkStatus();
}

GpuIpServer::~GpuIpServer() {
  for (auto& service : nic_services_) {
    service->Stop();
  }
}

}  // namespace tcpdirect

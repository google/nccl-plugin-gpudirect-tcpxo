/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/client/buffer_mgr_client.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "buffer_mgmt_daemon/client/unix_socket_client.h"
#include "buffer_mgmt_daemon/common/uds_helpers.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h"
#include "google/rpc/code.pb.h"
#include "google/rpc/status.pb.h"

namespace tcpdirect {
namespace {

constexpr int const kFtsMagic = 0x465453;
constexpr int const kNumUdsRetry = 5;
constexpr absl::Duration const kUdsRetryWaitTime = absl::Seconds(6);

}  // namespace

BufferManagerClient::BufferManagerClient(
    absl::string_view server_ip_addr,
    std::unique_ptr<UnixSocketClient> buf_op_client)
    : server_ip_addr_(server_ip_addr),
      buf_op_client_(std::move(buf_op_client)) {}

absl::StatusOr<std::unique_ptr<BufferManagerClientInterface>>
BufferManagerClient::Create(absl::string_view server_ip_addr) {
  LOG(INFO) << absl::StrFormat("Creating Buf op uds client  on server: %s",
                               BufOpUdsPath(server_ip_addr).c_str());
  auto buf_op_client =
      std::make_unique<UnixSocketClient>(BufOpUdsPath(server_ip_addr));

  absl::Status connect_status;
  for (int i = 0; i < kNumUdsRetry; ++i) {
    connect_status = buf_op_client->Connect();
    if (connect_status.ok()) break;
    absl::SleepFor(kUdsRetryWaitTime);
  }
  LOG_IF(ERROR, !connect_status.ok())
      << absl::StrFormat("Failed to connect to buffer operation client: %s,",
                         connect_status.ToString().c_str());
  RETURN_IF_ERROR(connect_status);

  return absl::WrapUnique(
      new BufferManagerClient(server_ip_addr, std::move(buf_op_client)));
}

absl::StatusOr<BufferOpResp> BufferManagerClient::ParseBufferOpResp(
    const UnixSocketMessage& ret_msg) {
  BufferOpResp resp;
  if (!resp.ParseFromString(ret_msg.text())) {
    return absl::InternalError("Got malformed response from buffer manager.");
  }
  if (resp.status().code() != google::rpc::Code::OK) {
    absl::Status ret(static_cast<absl::StatusCode>(resp.status().code()),
                     resp.status().message());
    return ret;
  }
  return resp;
}

absl::StatusOr<dxs::Reg> BufferManagerClient::RegBuf(int fd, size_t size) {
  BufferOpReq req;
  UnixSocketMessage msg;
  req.set_fts_magic_value(kFtsMagic);
  req.set_op_type(REG_BUFFER);
  req.set_size(size);
  msg.set_text(req.SerializeAsString());
  msg.set_fd(fd);
  ASSIGN_OR_RETURN(auto ret_msg, buf_op_client_->MakeRequest(std::move(msg)));
  ASSIGN_OR_RETURN(BufferOpResp resp, ParseBufferOpResp(ret_msg));
  if (!resp.has_reg_handle()) {
    LOG(ERROR) << absl::StrFormat("Reg handle not present in server response.");
    return absl::NotFoundError("Reg handle not present in server response.");
  }
  return resp.reg_handle();
}

absl::Status BufferManagerClient::DeregBuf(dxs::Reg handle) {
  BufferOpReq req;
  UnixSocketMessage msg;
  req.set_fts_magic_value(kFtsMagic);
  req.set_op_type(DEREG_BUFFER);
  req.set_reg_handle(handle);
  msg.set_text(req.SerializeAsString());
  ASSIGN_OR_RETURN(auto ret_msg, buf_op_client_->MakeRequest(msg));
  ASSIGN_OR_RETURN(BufferOpResp resp, ParseBufferOpResp(ret_msg));
  return absl::OkStatus();
}

std::string get_nic_ip_by_gpu_pci(absl::string_view gpu_pci) {
  std::string gpu_pci_sanitized(gpu_pci);
  absl::AsciiStrToLower(&gpu_pci_sanitized);
  UnixSocketClient client(NicIpUdsPath(gpu_pci_sanitized));
  if (!client.Connect().ok()) return "";
  GetNicIpReq req;
  GetNicIpResp resp;
  UnixSocketMessage msg;
  req.set_gpu_pci(gpu_pci_sanitized);
  msg.set_text(req.SerializeAsString());
  auto ret_msg = client.MakeRequest(std::move(msg));
  if (!ret_msg.status().ok() || !resp.ParseFromString(ret_msg->text())) {
    LOG(WARNING) << absl::StrFormat(
        "Failed to get nic ip from buffer manager because of %s",
        ret_msg.status().ToString().c_str());
    return "";
  }
  if (resp.status().code() != ::google::rpc::Code::OK) return "";
  return resp.nic_ip();
}

std::optional<GetNicMappingResp> get_nic_mapping() {
  UnixSocketClient client(NicMappingUdsPath());
  if (!client.Connect().ok()) return std::nullopt;
  GetNicMappingReq req;
  GetNicMappingResp resp;
  UnixSocketMessage msg;
  msg.set_text(req.SerializeAsString());
  auto ret_msg = client.MakeRequest(std::move(msg));
  if (!ret_msg.status().ok() || !resp.ParseFromString(ret_msg->text())) {
    LOG(WARNING) << absl::StrFormat(
        "Failed to get nic mapping from buffer manager because of %s",
        ret_msg.status().ToString().c_str());
    return std::nullopt;
  }
  return resp;
}

absl::Status rxdm_running() {
  UnixSocketClient client(NicMappingUdsPath());
  return client.Connect();
}

}  // namespace tcpdirect

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/fastrak_addr_translator.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "buffer_mgmt_daemon/pci_utils.h"
#include "dmabuf_bridge/dmabuf_importer.h"
#include "dxs/client/coalesce-iovecs.h"
#include "dxs/client/oss/status_macros.h"

ABSL_FLAG(bool, TESTONLY_buffer_manager_coalesce_iovecs, true,
          "If false, skip coalescing iovecs.");

namespace tcpdirect {

void FasTrakAddrTranslator::unmap_ctx_internal(AddrCtx& ctx) {
  if (dmabuf_importer_unmap(&ctx.buf)) {
    LOG(WARNING) << "Failed to unmap dmabuf";
  }
}

FasTrakAddrTranslator::FasTrakAddrTranslator(
    absl::string_view nic_pci_addr,
    std::optional<std::string> dmabuf_import_path)
    : nic_pci_addr_(nic_pci_addr),
      dmabuf_import_path_(dmabuf_import_path),
      next_id_(0),
      next_attach_handle_(0) {}

bool FasTrakAddrTranslator::Init() {
  LOG(INFO) << "FasTrakAddrTranslator: Initializing on PCI BDF "
            << nic_pci_addr_;
  if (tcpdirect::parse_pci_addr(nic_pci_addr_.c_str(), &nic_pci_.domain,
                                &nic_pci_.bus, &nic_pci_.device,
                                &nic_pci_.func))
    return false;
  if (dmabuf_importer_create_ctx(dmabuf_import_path_.has_value()
                                     ? dmabuf_import_path_->c_str()
                                     : nullptr,
                                 &ctx_)) {
    LOG(ERROR) << "Failed to create dmabuf importer context.";
    return false;
  }
  return true;
}

FasTrakAddrTranslator::~FasTrakAddrTranslator() {
  for (auto& [id, _addr_ctx] : addr_ctxs_) {
    unmap_ctx_internal(_addr_ctx);
  }
  if (ctx_ != nullptr) dmabuf_importer_destroy_ctx(ctx_);
}

absl::StatusOr<std::vector<iovec>> FasTrakAddrTranslator::GetIovecs(
    uint64_t id) {
  if (addr_ctxs_.find(id) == addr_ctxs_.end())
    return absl::InternalError(
        absl::StrFormat("Cannot find id %llu in Addr Translator", id));
  AddrCtx& ctx = addr_ctxs_[id];
  if (!ctx.vecs.empty()) return ctx.vecs;
  ASSIGN_OR_RETURN(ctx.vecs, PrepareIovecs(ctx.buf));
  return ctx.vecs;
}

absl::StatusOr<std::vector<iovec>> FasTrakAddrTranslator::PrepareIovecs(
    dmabuf_importer_buf& buf) {
  std::vector<iovec> iovecs(buf.num_iovecs);
  if (!dmabuf_importer_get_iovecs(&buf, iovecs.data())) {
    return absl::InternalError("Failed to retrieve iovecs");
  }
  if (!absl::GetFlag(FLAGS_TESTONLY_buffer_manager_coalesce_iovecs)) {
    return iovecs;
  }
  return dxs::CoalesceIovecs(std::move(iovecs));
}

absl::StatusOr<uint64_t> FasTrakAddrTranslator::Map(int dmabuf_fd) {
  AddrCtx ctx;
  dmabuf_importer_map_req map_req = {
      .dmabuf_fd = dmabuf_fd,
      .pci_dev_domain = nic_pci_.domain,
      .pci_dev_bus = nic_pci_.bus,
      .pci_dev_device = nic_pci_.device,
      .pci_dev_func = nic_pci_.func,
      .attach_handle = next_attach_handle_,
  };
  if (int ret = dmabuf_importer_map(ctx_, &map_req, &ctx.buf); ret != 0) {
    return absl::InternalError(absl::StrFormat(
        "Failed to map dmabuf fd %d, dmabuf_importer_map returned %d, errno %s",
        dmabuf_fd, ret, strerror(errno)));
  }
  next_attach_handle_++;
  uint64_t ret_id = next_id_++;
  addr_ctxs_[ret_id] = std::move(ctx);
  return ret_id;
}

void FasTrakAddrTranslator::Unmap(uint64_t id) {
  if (addr_ctxs_.find(id) != addr_ctxs_.end()) {
    AddrCtx& ctx = addr_ctxs_[id];
    unmap_ctx_internal(ctx);
    addr_ctxs_.erase(id);
  }
}

}  // namespace tcpdirect

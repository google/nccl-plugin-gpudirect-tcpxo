/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_ADDR_TRANSLATOR_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_ADDR_TRANSLATOR_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "buffer_mgmt_daemon/addr_translator_interface.h"
#include "dmabuf_bridge/dmabuf_importer.h"

namespace tcpdirect {

class FasTrakAddrTranslator : public AddrTranslatorInterface {
 public:
  explicit FasTrakAddrTranslator(
      absl::string_view nic_pci_addr,
      std::optional<std::string> dmabuf_import_path = std::nullopt);
  ~FasTrakAddrTranslator() override;
  bool Init() override;
  absl::StatusOr<uint64_t> Map(int dmabuf_fd) override;
  absl::StatusOr<std::vector<iovec>> GetIovecs(uint64_t id) override;
  void Unmap(uint64_t id) override;

 private:
  absl::StatusOr<std::vector<iovec>> PrepareIovecs(dmabuf_importer_buf& buf);

  struct AddrCtx {
    dmabuf_importer_buf buf;
    std::vector<iovec> vecs;
  };
  dmabuf_importer_ctx* ctx_ = nullptr;
  absl::flat_hash_map<uint64_t, AddrCtx> addr_ctxs_;
  struct pci_bdf {
    uint16_t domain;
    uint16_t bus;
    uint16_t device;
    uint16_t func;
  };
  pci_bdf nic_pci_;
  std::string nic_pci_addr_;
  std::optional<std::string> dmabuf_import_path_;
  uint64_t next_id_{1};
  int next_attach_handle_;
  void unmap_ctx_internal(AddrCtx& ctx);
};
}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_FASTRAK_ADDR_TRANSLATOR_H_

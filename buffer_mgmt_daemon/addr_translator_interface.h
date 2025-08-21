/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_ADDR_TRANSLATOR_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_ADDR_TRANSLATOR_INTERFACE_H_

#include <stdint.h>
#include <sys/uio.h>

#include <vector>

#include "absl/status/statusor.h"

namespace tcpdirect {
class AddrTranslatorInterface {
 public:
  virtual ~AddrTranslatorInterface() = default;
  virtual bool Init() = 0;
  /**
   * Pin a DMA-BUF fd and get its GPAs
   * Returns an absl::Status error on failure with the corresponding error code,
   * and a valid allocation id on success.
   */
  virtual absl::StatusOr<uint64_t> Map(int dmabuf_fd) = 0;
  virtual absl::StatusOr<std::vector<iovec>> GetIovecs(uint64_t id) = 0;
  // Unmaps a DMA-BUF
  virtual void Unmap(uint64_t id) = 0;
};
}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_ADDR_TRANSLATOR_INTERFACE_H_

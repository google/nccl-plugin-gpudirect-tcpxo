/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_PAGE_ALLOCATOR_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_PAGE_ALLOCATOR_INTERFACE_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/status/statusor.h"
#include "buffer_mgmt_daemon/client/bounce_buffer_handle.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "cuda.h"

namespace tcpdirect {

static constexpr int kMemHandleSize = sizeof(CUipcMemHandle);

struct IpcMetadata {
  tcpdirect::ExportHandleType handle_type;
  union {
    char mem_handle[kMemHandleSize];
    struct MemFdMetadata mem_fd;
  };
};

class PageAllocatorInterface {
 public:
  /**
   * Allocates GPU buffers that are no smaller than the requested size
   * and aligned to the current granularity of GPU vRAM, returns a non-zero
   * id corresponding to the created buffer(s) on success.
   */
  virtual absl::StatusOr<uint64_t> AllocatePage(size_t pool_size) = 0;
  /* Frees GPU buffers associated with the id. */
  virtual void FreePage(uint64_t id) = 0;
  /* Returns a ptr for the buffer associated with the id. */
  virtual void* GetMem(uint64_t id) = 0;
  /* Returns the dmabuf fd for the buffer associated with the id. */
  virtual int GetFd(uint64_t id) = 0;
  virtual void Reset() = 0;
  /**
   * Returns all metadata needed to import the buffer from another process.
   * This includes:
   * 1. The export handle of the buffer.
   * 2. Buffer size.
   * 3. GPU page alignment granularity.
   */
  virtual IpcMetadata GetIpcMetadata(uint64_t id) = 0;
  virtual ~PageAllocatorInterface() = default;
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_PAGE_ALLOCATOR_INTERFACE_H_

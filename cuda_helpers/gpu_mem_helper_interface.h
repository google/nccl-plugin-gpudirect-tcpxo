/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef CUDA_HELPERS_GPU_MEM_HELPER_INTERFACE_H_
#define CUDA_HELPERS_GPU_MEM_HELPER_INTERFACE_H_

#include <stdint.h>

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "buffer_mgmt_daemon/client/bounce_buffer_handle.h"

namespace cuda_helpers {

class GpuMemHelperInterface {
 public:
  virtual ~GpuMemHelperInterface() = default;
  virtual absl::Status Init() = 0;
  virtual absl::StatusOr<uint64_t> ImportBounceBuffer(
      const tcpdirect::BounceBufHandle& bounce_buf_handle) = 0;
  virtual absl::StatusOr<uint64_t> CreateBuffer(size_t size) = 0;
  virtual absl::StatusOr<int> GetFd(uint64_t id) = 0;
  virtual absl::StatusOr<void*> GetMem(uint64_t id) = 0;
  virtual absl::Status WriteBuffer(uint64_t id, const void* src, size_t offset,
                                   size_t len) = 0;
  virtual absl::Status ReadBuffer(uint64_t id, void* dst, size_t offset,
                                  size_t len) = 0;
  virtual void FreeBuffer(uint64_t id) = 0;
};

}  // namespace cuda_helpers

#endif  // CUDA_HELPERS_GPU_MEM_HELPER_INTERFACE_H_

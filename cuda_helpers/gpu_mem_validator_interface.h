/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef CUDA_HELPERS_GPU_MEM_VALIDATOR_INTERFACE_H_
#define CUDA_HELPERS_GPU_MEM_VALIDATOR_INTERFACE_H_

#include <stdint.h>
#include <sys/uio.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace cuda_helpers {

class GpuMemValidatorInterface {
 public:
  virtual ~GpuMemValidatorInterface() = default;
  virtual absl::Status Init() = 0;
  virtual absl::StatusOr<bool> MemCmp(void* a, void* b, uint64_t len) = 0;
  virtual absl::Status GatherRxData(iovec* iovecs, uint32_t num_iovecs,
                                    void* dst, void* src) = 0;
};

}  // namespace cuda_helpers

#endif  // CUDA_HELPERS_GPU_MEM_VALIDATOR_INTERFACE_H_

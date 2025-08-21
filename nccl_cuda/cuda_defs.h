/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef NCCL_CUDA_CUDA_DEFS_H_
#define NCCL_CUDA_CUDA_DEFS_H_

#include <cstddef>

#include "cuda.h"

namespace fastrak {

constexpr size_t kMaxGpuDevices = 8;
// Max is 4 + 4 + 2 + 1 + 3 + 1 (for \0) = 15 bytes
constexpr size_t kPciAddrLen = 16;

struct gpuDev {
  CUdevice dev = CU_DEVICE_INVALID;
  CUcontext ctx;
  int freq;
  char pci_addr[kPciAddrLen];
};

}  // namespace fastrak

#endif  // NCCL_CUDA_CUDA_DEFS_H_

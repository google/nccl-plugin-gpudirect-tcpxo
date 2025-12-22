/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_COMMON_NVIDIA_MEM_SHARE_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_COMMON_NVIDIA_MEM_SHARE_INTERFACE_H_

#include <cstddef>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "cuda.h"

namespace tcpdirect {

class NvidiaMemShareUtilInterface {
 public:
  virtual ~NvidiaMemShareUtilInterface() = default;
  virtual absl::StatusOr<int> GetDmabuf(absl::string_view gpu_pci_addr,
                                        CUdeviceptr gpu_mem_ptr,
                                        size_t size) = 0;
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_COMMON_NVIDIA_MEM_SHARE_INTERFACE_H_

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_COMMON_NVIDIA_TCPXO_MEM_SHARE_H_
#define BUFFER_MGMT_DAEMON_COMMON_NVIDIA_TCPXO_MEM_SHARE_H_

#include <unistd.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "buffer_mgmt_daemon/common/nvidia_mem_share_interface.h"
#include "buffer_mgmt_daemon/cuda_logging.h"
#include "cuda.h"

namespace tcpdirect {

class NvidiaTcpxoMemShareUtil : public NvidiaMemShareUtilInterface {
 public:
  absl::StatusOr<int> GetDmabuf(absl::string_view gpu_pci_addr,
                                CUdeviceptr gpu_mem_ptr, size_t size) override {
    int dma_buf_fd;

    bool get_dmabuf_fd_success = CUCallSuccess(cuMemGetHandleForAddressRange(
        reinterpret_cast<void*>(&dma_buf_fd), gpu_mem_ptr, size,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    if (!get_dmabuf_fd_success) {
      return absl::InternalError(
          absl::StrFormat("Failed to get dmabuf fd for gpu_mem_ptr:%p, size:%d",
                          (void*)gpu_mem_ptr, size));
    }
    return dma_buf_fd;
  }
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_COMMON_NVIDIA_TCPXO_MEM_SHARE_H_

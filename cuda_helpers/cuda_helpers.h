/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef CUDA_HELPERS_CUDA_HELPERS_CU_H_
#define CUDA_HELPERS_CUDA_HELPERS_CU_H_

#include <stdint.h>
#include <sys/uio.h>

#include <cstddef>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "buffer_mgmt_daemon/client/bounce_buffer_handle.h"
#include "cuda.h"
#include "cuda_helpers/gpu_mem_helper_interface.h"
#include "cuda_helpers/gpu_mem_validator_interface.h"
#include "driver_types.h"
#include "nccl.h"

namespace cuda_helpers {

absl::Status InitCuda();

class GpuMemHelper : public GpuMemHelperInterface {
 public:
  explicit GpuMemHelper(const std::string& gpu_pci);
  absl::Status Init() override;
  absl::StatusOr<uint64_t> ImportBounceBuffer(
      const tcpdirect::BounceBufHandle& bounce_buf_handle) override;
  absl::StatusOr<uint64_t> CreateBuffer(size_t size) override;
  absl::StatusOr<int> GetFd(uint64_t id) override;
  /**
   * Retrieves the address of the CUDA buffer associated with <id>, the address
   * is casted to void *.
   */
  absl::StatusOr<void*> GetMem(uint64_t id) override;
  absl::Status WriteBuffer(uint64_t id, const void* src, size_t offset,
                           size_t len) override;
  absl::Status ReadBuffer(uint64_t id, void* dst, size_t offset,
                          size_t len) override;
  void FreeBuffer(uint64_t id) override;

 private:
  enum AllocationType {
    BOUNCE_BUFFER = 0,
    TX_BUFFER = 1,
    TX_BUFFER_WITH_SHIM = 2,
  };

  struct GpuMemInfo {
    AllocationType allocation_type;
    CUdeviceptr ptr;
    CUmemGenericAllocationHandle handle;
    int fd;
    size_t size;
  };

  absl::Status AllocateGpuMem(GpuMemInfo& info);

  std::string gpu_pci_;

  CUdevice dev_;
  CUcontext ctx_;

  uint64_t next_id_{0};
  absl::flat_hash_map<uint64_t, GpuMemInfo> mem_infos_;
};

/* Helper class for all functionalities related to verifying GPU memory content.
 */
class GpuMemValidator : public GpuMemValidatorInterface {
 public:
  explicit GpuMemValidator(const std::string& gpu_pci, bool new_ctx = false);
  absl::Status Init() override;
  /**
   * Compares the memory content of 2 buffers, using a CUDA kernel.
   * Returns true if the contents are identical, and false otherwise.
   */
  absl::StatusOr<bool> MemCmp(void* a, void* b, uint64_t len) override;
  /**
   * Reads iovec info in `iovec` and copies from the corresponding offset of
   * src buffer into dst buffer.
   *
   * No check is performed on memory boundaries so it is the user's
   * responsibility to check iovec and buffer validity.
   */
  absl::Status GatherRxData(iovec* iovecs, uint32_t num_iovecs, void* dst,
                            void* src) override;
  ~GpuMemValidator() override;

 private:
  std::string gpu_pci_;
  bool new_ctx_;
  CUdevice dev_;
  CUcontext ctx_;
  cudaStream_t stream_;
};

/**
 * Given a NIC IP, find the PCI BDF of the GPU that is associated with that
 * NIC.
 * This function assumes that a functional buffer manager daemon is running on
 * the same host.
 */
absl::StatusOr<std::string> find_gpu_pci_for_ip(const std::string& ip);

// Wait for RxDM to come up before connecting to its services
//
// timeout_seconds: Number of seconds to wait for it to come up
absl::Status wait_for_rxdm(int timeout_seconds);

absl::Status cu_call_success(CUresult err);

absl::Status cuda_call_success(cudaError_t err);

// Copies the bounce buffer memory pointed by iovecs in `handle` in `req_idx`
// into device memory `dest` by launch kernel and waiting for its completion.
absl::Status CopyGpuMemFromDeviceHandle(void* handle, CUdeviceptr dest,
                                        int req_idx);

}  // namespace cuda_helpers

#endif  // CUDA_HELPERS_CUDA_HELPERS_CU_H_

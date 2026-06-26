/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "nccl_cuda/cuda_common.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "cuda.h"
#include "driver_types.h"
#include "dxs/client/oss/status_macros.h"
#include "nccl.h"
#include "nccl_cuda/cuda_defs.h"

const char* GetNcclErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess:
      return "no error";
    case ncclUnhandledCudaError:
      return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError:
      return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError:
      return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument:
      return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage:
      return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError:
      return "remote process exited or there was a network error";
    case ncclInProgress:
      return "NCCL operation in progress";
    default:
      return "unknown result code";
  }
}

namespace fastrak {

absl::StatusOr<int> GetDeviceIndex() {
  CUdevice device;
  RETURN_IF_ERROR(cu_call_success(cuCtxGetDevice(&device)));
  return static_cast<int>(device);
}

absl::StatusOr<gpuDev> initGpuDev() {
  gpuDev gpu;
  RETURN_IF_ERROR(cu_call_success(cuCtxGetDevice(&gpu.dev)));
  RETURN_IF_ERROR(
      cu_call_success(cuDeviceGetPCIBusId(gpu.pci_addr, kPciAddrLen, gpu.dev)));
  RETURN_IF_ERROR(cu_call_success(cuDeviceGetAttribute(
      &gpu.freq, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, gpu.dev)));
  return gpu;
}

namespace {}  // namespace

absl::StatusOr<int> getDmabufFd(void* ptr, size_t size,
                                absl::string_view gpu_pci_addr) {
  int fd = -1;
  RETURN_IF_ERROR(cu_call_success(cuMemGetHandleForAddressRange(
      &fd, reinterpret_cast<CUdeviceptr>(ptr), size,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)));
  return fd;
}

absl::StatusOr<int> getDeviceCount() {
  int num_devices = -1;
  RETURN_IF_ERROR(cu_call_success(cuDeviceGetCount(&num_devices)));
  return num_devices;
}

absl::Status checkDeviceCount(int max_devices) {
  ASSIGN_OR_RETURN(int num_devices, getDeviceCount());
  if (num_devices > max_devices) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "NET/FasTrak: Number of GPUs (%d) is larger than maximum number: %d",
        num_devices, max_devices));
  }
  return absl::OkStatus();
}

absl::Status cu_call_success(CUresult err) {
  if (err != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* reason = nullptr;
    if (cuGetErrorName(err, &name)) {
      return absl::InternalError(absl::StrFormat(
          "Error: error getting error name from CU error %d", err));
    }
    if (cuGetErrorString(err, &reason)) {
      return absl::InternalError(absl::StrFormat(
          "Error: error getting error string from CU error %d", err));
    }
    return absl::InternalError(absl::StrFormat(
        "cuda error detected! name: %s; string: %s", name, reason));
  }
  return absl::OkStatus();
}

absl::Status nccl_call_success(ncclResult_t err) {
  if (err != ncclSuccess) {
    const char* name = GetNcclErrorString(err);
    if (name == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Error: error getting error name from NCCL error %d", err));
    }
    return absl::InternalError(
        absl::StrFormat("nccl error detected! name: %s.", name));
  }
  return absl::OkStatus();
}

}  // namespace fastrak

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
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
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
  int device;
  RETURN_IF_ERROR(cuda_call_success(cudaGetDevice(&device)));
  return device;
}

absl::StatusOr<gpuDev> initGpuDev() {
  gpuDev gpu;
  RETURN_IF_ERROR(cuda_call_success(cudaGetDevice(&gpu.dev)));
  RETURN_IF_ERROR(cuda_call_success(
      cudaDeviceGetPCIBusId(gpu.pci_addr, kPciAddrLen, gpu.dev)));
  RETURN_IF_ERROR(cu_call_success(cuDeviceGetAttribute(
      &gpu.freq, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, gpu.dev)));
  return gpu;
}

absl::StatusOr<int> getDeviceId(const void* ptr) {
  cudaPointerAttributes attrs;
  RETURN_IF_ERROR(cuda_call_success(cudaPointerGetAttributes(&attrs, ptr)));
  return attrs.type == cudaMemoryTypeDevice ? attrs.device : -1;
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

absl::Status cuda_call_success(cudaError_t err) {
  if (err != cudaSuccess) {
    const char* name = cudaGetErrorName(err);
    const char* reason = cudaGetErrorString(err);
    if (name == nullptr || reason == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Faile to get error name and reason from CUDA error %d", err));
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

/**
 * Random number generator kernel for generation/validation.
 * If match == nullptr, then the default behavior is to generate, otherwise
 * validate.
 */
__global__ void rng_kernel(uint64_t seed, uint32_t* data, uint64_t num_elems,
                           volatile bool* match) {
  curandState state;
  curand_init(seed, threadIdx.x, 0, &state);
  __shared__ bool mismatch_happened;
  if (threadIdx.x == 0) {
    mismatch_happened = false;
  }
  __syncthreads();
  for (uint64_t i = threadIdx.x; i < num_elems; i += blockDim.x) {
    if (mismatch_happened) {
      break;
    }
    uint32_t rand_num = curand(&state);
    /* Populate the first and last element of the buffer with special values
       (current seed and the next seed) for debugging purposes, telling us if
       data has been corrupted. */
    if (i == 0) {
      rand_num = seed;
    } else if (i == num_elems - 1) {
      rand_num = seed + 1;
    }
    if (match == nullptr) {
      // Generation mode
      data[i] = rand_num;
    } else {
      // Validation mode
      if (data[i] != rand_num) {
        mismatch_happened = true;
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && match != nullptr) {
    *match = !mismatch_happened;
    if (mismatch_happened) {
      printf("Start of array has value %u, expected value %lu.\n", data[0],
             seed);
      printf("End of array has value %u, expected value %lu.\n",
             data[num_elems - 1], seed + 1);
    }
  }
}

absl::Status GeneratePayload(int seed, void* ptr, uint64_t size) {
  uint64_t num_elems = size / sizeof(uint32_t);
  rng_kernel<<<1, 1024, 0>>>(seed, reinterpret_cast<uint32_t*>(ptr), num_elems,
                             nullptr);
  RETURN_IF_ERROR(cuda_call_success(cudaStreamSynchronize(CU_STREAM_LEGACY)));
  return absl::OkStatus();
}

absl::Status ValidatePayload(int seed, void* ptr, uint64_t size) {
  uint64_t num_elems = size / sizeof(uint32_t);
  void* match;
  CUdeviceptr match_device;
  RETURN_IF_ERROR(cu_call_success(
      cuMemHostAlloc(&match, sizeof(bool),
                     CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)));
  auto dealloc_mem = absl::MakeCleanup([match] { cuMemFreeHost(match); });
  RETURN_IF_ERROR(
      cu_call_success(cuMemHostGetDevicePointer(&match_device, match, 0)));
  rng_kernel<<<1, 1024, 0>>>(seed, reinterpret_cast<uint32_t*>(ptr), num_elems,
                             reinterpret_cast<bool*>(match_device));
  RETURN_IF_ERROR(cuda_call_success(cudaStreamSynchronize(CU_STREAM_LEGACY)));
  if (!*reinterpret_cast<bool*>(match)) {
    return absl::InternalError(
        absl::StrFormat("GpuRandomPayloadGenerator: Payload content does not "
                        "match random numbers generated with seed %d",
                        seed));
  }
  return absl::OkStatus();
}

}  // namespace fastrak

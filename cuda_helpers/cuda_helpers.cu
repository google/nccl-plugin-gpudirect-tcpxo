/*
* Copyright 2025 Google LLC
*
* Use of this source code is governed by a BSD-style
* license that can be found in the LICENSE.md file or at
* https://developers.google.com/open-source/licenses/bsd
 */

#include "cuda_helpers/cuda_helpers.h"
#if defined(__PPC__)
static inline void wc_store_fence(void) { asm volatile("sync"); }
#elif defined(__x86_64__)
#include <immintrin.h>
static inline void wc_store_fence(void) { _mm_sfence(); }
#elif defined(__aarch64__)
#ifdef __cplusplus
#include <atomic>
static inline void wc_store_fence(void) {
  std::atomic_thread_fence(std::memory_order_release);
}
#else
#include <stdatomic.h>
static inline void wc_store_fence(void) {
  atomic_thread_fence(memory_order_release);
}
#endif
#endif

#include "buffer_mgmt_daemon/client/buffer_mgr_client.h"
#include "buffer_mgmt_daemon/common/nvidia_tcpxo_mem_share.h"
#include "buffer_mgmt_daemon/common/nvidia_mem_share_interface.h"
#include "buffer_mgmt_daemon/cuda_logging.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cuda.h"
#include "curand_kernel.h"
#include "nccl.h"
#include "tcpdirect_plugin/fastrak_offload/shared_defs.h"
#include "dxs/client/oss/status_macros.h"

constexpr int kWaitRXDMSeconds = 30;

// Launches a simplified version of the memory copy kernel in
// NCCL that copies the bounce buffers.
__global__ void iovec_cpy_kernel(void *ohandle, uint8_t *dst, int req_idx) {
  unpackNetDeviceHandle *unpack_handle =
      reinterpret_cast<struct unpackNetDeviceHandle *>(ohandle);
  loadMeta *mem = unpack_handle->meta->mem[req_idx];
  int cnt = unpack_handle->meta->cnt[req_idx];
  uint8_t *src = reinterpret_cast<uint8_t *>(unpack_handle->bounce_buf);
  int thread_idx = threadIdx.x;
  for (int i = thread_idx; i < cnt; i += blockDim.x) {
    union loadMeta *mem_entry = &mem[i];
    uint32_t src_off = mem_entry->src_off;
    uint64_t dst_off = mem_entry->dst_off;

    for (uint32_t j = 0; j < mem_entry->len; ++j) {
      dst[dst_off + j] = src[src_off + j];
    }
  }
}

namespace cuda_helpers {

absl::Status cu_call_success(CUresult err) {
  if (err != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *reason = nullptr;
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
    const char *name = cudaGetErrorName(err);
    const char *reason = cudaGetErrorString(err);
    if (name == nullptr || reason == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Faile to get error name and reason from CUDA error %d", err));
    }
    return absl::InternalError(absl::StrFormat(
        "cuda error detected! name: %s; string: %s", name, reason));
  }
  return absl::OkStatus();
}

absl::Status InitCuda() { return cu_call_success(cuInit(/*flags=*/0)); }

GpuMemHelper::GpuMemHelper(const std::string &gpu_pci) : gpu_pci_(gpu_pci) {}

absl::Status GpuMemHelper::Init() {
  auto err = cu_call_success(cuDeviceGetByPCIBusId(&dev_, gpu_pci_.c_str()));
  if (!err.ok()) {
    return err;
  }
  auto ctx_err = cu_call_success(cuDevicePrimaryCtxRetain(&ctx_, dev_));
  if (!ctx_err.ok()) {
    return ctx_err;
  }
  RETURN_IF_ERROR(cu_call_success(cuCtxPushCurrent(ctx_)));
  CUcontext old_ctx;
  cuCtxPopCurrent(&old_ctx);
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> GpuMemHelper::ImportBounceBuffer(
    const tcpdirect::BounceBufHandle &bounce_buf_handle) {
  if (bounce_buf_handle.handle_type != tcpdirect::MEM_GPU_FD) {
    return absl::UnimplementedError("Only MEM_GPU_FD is supported.");
  }
  struct GpuMemInfo info;
  info.allocation_type = BOUNCE_BUFFER;
  CUmemAllocationHandleType handleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUmemAccessDesc desc = {};
  auto &fd = bounce_buf_handle.mem_fd.fd;
  auto &size = bounce_buf_handle.mem_fd.size;
  auto &align = bounce_buf_handle.mem_fd.align;
  auto handle = &info.handle;
  auto &ptr = info.ptr;
  auto push_ctx_status = cu_call_success(cuCtxPushCurrent(ctx_));
  if (!push_ctx_status.ok()) {
    LOG(ERROR) << "Error in ImportBounceBuffer: " << push_ctx_status;
    return absl::InternalError("Failed to push CU context.");
  }
  auto pop_ctx = absl::MakeCleanup([this] {
    CUcontext old_ctx;
    cuCtxPopCurrent(&old_ctx);
  });
  auto import_status = cu_call_success(cuMemImportFromShareableHandle(
      handle, reinterpret_cast<void *>((uint64_t)fd), handleType));
  if (!import_status.ok()) {
    LOG(ERROR) << "Error in ImportBounceBuffer: " << import_status;
    return absl::InternalError(
        absl::StrFormat("Failed to import from shareable handle %d", fd));
  }
  auto release_mem = absl::MakeCleanup([handle] { cuMemRelease(*handle); });
  auto mem_reserve_status =
      cu_call_success(cuMemAddressReserve(&ptr, size, align, 0, 0));
  if (!mem_reserve_status.ok()) {
    LOG(ERROR) << "Error in ImportBounceBuffer: " << mem_reserve_status;
    return absl::InternalError("Failed to reserve CU mem.");
  }
  auto free_addr =
      absl::MakeCleanup([ptr, size] { cuMemAddressFree(ptr, size); });
  auto mem_map_status = cu_call_success(cuMemMap(ptr, size, 0, *handle, 0));
  if (!mem_map_status.ok()) {
    LOG(ERROR) << "Error in ImportBounceBuffer: " << mem_map_status;
    return absl::InternalError("Failed to map CU mem.");
  }
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = dev_;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  auto unmap_mem = absl::MakeCleanup([ptr, size] { cuMemUnmap(ptr, size); });
  auto mem_set_access_status =
      cu_call_success(cuMemSetAccess(ptr, size, &desc, 1));
  if (!mem_set_access_status.ok()) {
    LOG(ERROR) << "Error in ImportBounceBuffer: " << mem_set_access_status;
    return absl::InternalError("Failed to set access to CU mem.");
  }
  std::move(unmap_mem).Cancel();
  std::move(free_addr).Cancel();
  std::move(release_mem).Cancel();
  cuMemGetHandleForAddressRange(reinterpret_cast<void *>(&info.fd), ptr, size,
                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  info.size = size;
  mem_infos_[next_id_] = std::move(info);
  return next_id_++;
}

// Allocate GPU memory that fits requirements for dma-buf sharing with NICs.
absl::Status GpuMemHelper::AllocateGpuMem(GpuMemInfo &info) {
  info.allocation_type = TX_BUFFER;
  return cu_call_success(cuMemAlloc(&info.ptr, info.size));
}

namespace {
std::unique_ptr<tcpdirect::NvidiaMemShareUtilInterface> GetMemShareUtil() {
  return std::make_unique<tcpdirect::NvidiaTcpxoMemShareUtil>();
}
}  // namespace

absl::StatusOr<uint64_t> GpuMemHelper::CreateBuffer(size_t size) {
  struct GpuMemInfo info;
  auto push_ctx_status = cu_call_success(cuCtxPushCurrent(ctx_));
  if (!push_ctx_status.ok()) {
    LOG(ERROR) << "Error in CreateBuffer: " << push_ctx_status;
    return absl::InternalError("Failed to push CU context.");
  }
  auto pop_ctx = absl::MakeCleanup([this] {
    CUcontext old_ctx;
    cuCtxPopCurrent(&old_ctx);
  });

  // Allocate gpu memory.
  info.size = size;
  RETURN_IF_ERROR(AllocateGpuMem(info));

  // Get dma-buf fd.
  std::unique_ptr<tcpdirect::NvidiaMemShareUtilInterface> mem_share_util =
      GetMemShareUtil();
  absl::StatusOr<int> dma_fd =
      mem_share_util->GetDmabuf(gpu_pci_, info.ptr, size);

  // Free allocated memory if it didn't work.
  if (!dma_fd.ok()) {
    LOG(ERROR) << "Error in CreateBuffer: " << dma_fd.status();
    cuMemFree(info.ptr);
    return absl::InternalError("Failed to get DMA buffer fd.");
  }
  info.fd = dma_fd.value();

  mem_infos_[next_id_] = std::move(info);
  return next_id_++;
}

absl::StatusOr<int> GpuMemHelper::GetFd(uint64_t id) {
  if (!mem_infos_.contains(id)) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid id %llu.", id));
  }
  auto &info = mem_infos_[id];
  return info.fd;
}

absl::Status GpuMemHelper::WriteBuffer(uint64_t id, const void *src,
                                       size_t offset, size_t len) {
  if (!mem_infos_.contains(id)) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid id %llu.", id));
  }
  auto &ptr = mem_infos_[id].ptr;
  auto push_ctx_status = cu_call_success(cuCtxPushCurrent(ctx_));
  if (!push_ctx_status.ok()) {
    LOG(ERROR) << "Error in WriteBuffer: " << push_ctx_status;
    return absl::InternalError("Failed to push CU context.");
  }
  auto pop_ctx = absl::MakeCleanup([this] {
    CUcontext old_ctx;
    cuCtxPopCurrent(&old_ctx);
  });
  auto memcpy_status = cu_call_success(cuMemcpyHtoD(ptr + offset, src, len));
  if (!memcpy_status.ok()) {
    LOG(ERROR) << "Error in WriteBuffer: " << memcpy_status;
    return absl::InternalError(
        absl::StrFormat("Failed to write to GPU buffer for id %llu.", id));
  }
  return absl::OkStatus();
}

absl::StatusOr<void *> GpuMemHelper::GetMem(uint64_t id) {
  if (!mem_infos_.contains(id)) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid id %llu", id));
  }
  auto &ptr = mem_infos_[id].ptr;
  return reinterpret_cast<void *>(ptr);
}

absl::Status GpuMemHelper::ReadBuffer(uint64_t id, void *dst, size_t offset,
                                      size_t len) {
  if (!mem_infos_.contains(id)) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid id %llu.", id));
  }
  auto &ptr = mem_infos_[id].ptr;
  auto push_ctx_status = cu_call_success(cuCtxPushCurrent(ctx_));
  if (!push_ctx_status.ok()) {
    LOG(ERROR) << "Error in ReadBuffer: " << push_ctx_status;
    return absl::InternalError("Failed to push CU context.");
  }
  auto pop_ctx = absl::MakeCleanup([this] {
    CUcontext old_ctx;
    cuCtxPopCurrent(&old_ctx);
  });
  auto memcpy_status = cu_call_success(cuMemcpyDtoH(dst, ptr + offset, len));
  if (!memcpy_status.ok()) {
    LOG(ERROR) << "Error in ReadBuffer: " << memcpy_status;
    return absl::InternalError(
        absl::StrFormat("Failed to read from GPU buffer for id %llu.", id));
  }
  return absl::OkStatus();
}

void GpuMemHelper::FreeBuffer(uint64_t id) {
  if (!mem_infos_.contains(id)) {
    LOG(ERROR) << "Error in FreeBuffer: invalid id " << id;
    return;
  }
  auto push_ctx_status = cu_call_success(cuCtxPushCurrent(ctx_));
  if (!push_ctx_status.ok()) {
    LOG(ERROR) << "Error in FreeBuffer: " << push_ctx_status;
    return;
  }
  auto pop_ctx = absl::MakeCleanup([this] {
    CUcontext old_ctx;
    cuCtxPopCurrent(&old_ctx);
  });
  auto &info = mem_infos_[id];
  if (info.fd > 0) {
    close(info.fd);
  }
  if (info.allocation_type == TX_BUFFER) {
    cuMemFree(info.ptr);
  } else if (info.allocation_type == TX_BUFFER_WITH_SHIM) {
    cuMemUnmap(info.ptr, info.size);
    cuMemRelease(info.handle);
    cuMemAddressFree(info.ptr, info.size);
  } else {
    cuMemUnmap(info.ptr, info.size);
    cuMemRelease(info.handle);
    cuMemAddressFree(info.ptr, info.size);
  }
  mem_infos_.erase(id);
}

absl::Status wait_for_rxdm(int timeout_seconds) {
  absl::Status status = tcpdirect::rxdm_running();
  for (int i = 0; i < timeout_seconds; ++i) {
    if (status.ok()) {
      return status;
    }
    absl::SleepFor(absl::Seconds(1));
    status = tcpdirect::rxdm_running();
  }
  return absl::InternalError("RXDM is not ready");
}

absl::StatusOr<std::string> find_gpu_pci_for_ip(const std::string &ip) {
  static constexpr size_t pci_addr_len = 16;
  int dev_count = -1;
  RETURN_IF_ERROR(cu_call_success(cuDeviceGetCount(&dev_count)));
  RETURN_IF_ERROR(cuda_helpers::wait_for_rxdm(kWaitRXDMSeconds))
      << "Failed waiting for RXDM to start";
  char pci_addr[pci_addr_len];
  for (int i = 0; i < dev_count; ++i) {
    CUdevice dev;
    RETURN_IF_ERROR(cu_call_success(cuDeviceGet(&dev, i)));
    RETURN_IF_ERROR(
        cu_call_success(cuDeviceGetPCIBusId(pci_addr, pci_addr_len, dev)));
    std::string gpu_pci(pci_addr);
    absl::AsciiStrToLower(&gpu_pci);
    auto result_ip = tcpdirect::get_nic_ip_by_gpu_pci(gpu_pci);
    if (result_ip == ip) {
      return gpu_pci;
    }
  }
  return absl::InternalError(absl::StrFormat(
      "Cannot find associated GPU PCI with NIC IP %s", ip.c_str()));
}

GpuMemValidator::GpuMemValidator(const std::string &gpu_pci, bool new_ctx)
    : gpu_pci_(gpu_pci), new_ctx_(new_ctx) {}

absl::Status GpuMemValidator::Init() {
  RETURN_IF_ERROR(
      cu_call_success(cuDeviceGetByPCIBusId(&dev_, gpu_pci_.c_str())));
  absl::Status ctx_err;
  if (new_ctx_) {
#if CUDA_VERSION < 13000
    ctx_err = cu_call_success(cuCtxCreate(&ctx_, 0, dev_));
#else
    ctx_err = cu_call_success(cuCtxCreate(&ctx_, nullptr, 0, dev_));
#endif
  } else {
    ctx_err = cu_call_success(cuDevicePrimaryCtxRetain(&ctx_, dev_));
  }
  RETURN_IF_ERROR(ctx_err);
  if (!new_ctx_) {
    RETURN_IF_ERROR(cu_call_success(cuCtxPushCurrent(ctx_)));
  }
  RETURN_IF_ERROR(cuda_call_success(
      cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)));
  CUcontext old_ctx;
  cuCtxPopCurrent(&old_ctx);
  return absl::OkStatus();
}

__global__ void memcmp_kernel(uint32_t *a, uint32_t *b, uint64_t num_elems,
                              bool *match) {
  volatile __shared__ bool mismatch_happened;
  if (threadIdx.x == 0) {
    mismatch_happened = false;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
    if (a[i] != b[i]) {
      mismatch_happened = true;
      break;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *match = !mismatch_happened;
  }
}
__global__ void memcpy_kernel(ulong3 *scatter_list, uint8_t *dst,
                              uint8_t *src) {
  int block_idx = blockIdx.x;
  ulong3 blk = scatter_list[block_idx];
  uint64_t src_off = blk.x;
  uint64_t sz = blk.y;
  uint64_t dst_off = blk.z;

  for (int i = threadIdx.x; i < sz; i += blockDim.x) {
    dst[dst_off + i] = src[src_off + i];
  }
}

absl::StatusOr<bool> GpuMemValidator::MemCmp(void *a, void *b, uint64_t len) {
  RETURN_IF_ERROR(cu_call_success(cuCtxPushCurrent(ctx_)));
  auto pop_ctx = absl::MakeCleanup([this] {
    CUcontext old_ctx;
    cuCtxPopCurrent(&old_ctx);
  });
  void *match;
  CUdeviceptr match_device;
  RETURN_IF_ERROR(cu_call_success(
      cuMemHostAlloc(&match, sizeof(bool),
                     CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)));
  auto dealloc_mem = absl::MakeCleanup([match] { cuMemFreeHost(match); });
  RETURN_IF_ERROR(
      cu_call_success(cuMemHostGetDevicePointer(&match_device, match, 0)));
  // We must only launch one block because of the single __shared__ flag
  memcmp_kernel<<<1, 256, sizeof(bool), stream_>>>(
      reinterpret_cast<uint32_t *>(a), reinterpret_cast<uint32_t *>(b),
      len / sizeof(uint32_t), reinterpret_cast<bool *>(match_device));
  RETURN_IF_ERROR(cuda_call_success(cudaStreamSynchronize(stream_)));
  /* Explicit fencing to make sure we are reading the right data. */
  wc_store_fence();
  bool result = *reinterpret_cast<bool *>(match);
  return result;
}

absl::Status GpuMemValidator::GatherRxData(iovec *iovecs, uint32_t num_iovecs,
                                           void *dst, void *src) {
  RETURN_IF_ERROR(cu_call_success(cuCtxPushCurrent(ctx_)));
  auto pop_ctx = absl::MakeCleanup([this] {
    CUcontext old_ctx;
    cuCtxPopCurrent(&old_ctx);
  });
  uint64_t offset = 0;
  void *memcpy_meta;
  RETURN_IF_ERROR(cu_call_success(
      cuMemHostAlloc(&memcpy_meta, sizeof(long3) * num_iovecs,
                     CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)));
  auto dealloc_mem =
      absl::MakeCleanup([memcpy_meta] { cuMemFreeHost(memcpy_meta); });
  ulong3 *memcpy_meta_longs = reinterpret_cast<ulong3 *>(memcpy_meta);
  for (uint32_t i = 0; i < num_iovecs; i++) {
    auto &entry = memcpy_meta_longs[i];
    entry.x = reinterpret_cast<uint64_t>(iovecs[i].iov_base);
    entry.y = reinterpret_cast<uint64_t>(iovecs[i].iov_len);
    entry.z = reinterpret_cast<uint64_t>(offset);
    offset += iovecs[i].iov_len;
  }
  CUdeviceptr memcpy_meta_device;
  RETURN_IF_ERROR(cu_call_success(
      cuMemHostGetDevicePointer(&memcpy_meta_device, memcpy_meta, 0)));
  wc_store_fence();
  memcpy_kernel<<<num_iovecs, 256, 0, stream_>>>(
      reinterpret_cast<ulong3 *>(memcpy_meta_device),
      reinterpret_cast<uint8_t *>(dst), reinterpret_cast<uint8_t *>(src));
  RETURN_IF_ERROR(cuda_call_success(cudaStreamSynchronize(stream_)));
  return absl::OkStatus();
}

GpuMemValidator::~GpuMemValidator() {
  cudaStreamDestroy(stream_);
  if (new_ctx_) {
    cuCtxDestroy(ctx_);
  } else {
    cuDevicePrimaryCtxRelease(dev_);
  }
}

absl::Status CopyGpuMemFromDeviceHandle(void *meta, CUdeviceptr dest,
                                        int req_idx) {
  uint8_t *dest_ptr = reinterpret_cast<uint8_t *>(dest);
  iovec_cpy_kernel<<<1, 1024, 0>>>((void *)meta, dest_ptr, req_idx);
  return cuda_call_success(cudaDeviceSynchronize());
}

}  // namespace cuda_helpers

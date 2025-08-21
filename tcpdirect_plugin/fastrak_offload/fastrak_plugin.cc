/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "dxs/client/oss/status_macros.h"
#include "nccl.h"
#include "nccl_common.h"
#include "nccl_cuda/cuda_common.h"
#include "nccl_cuda/cuda_defs.h"
#include "net_device.h"
#include "tcpdirect_plugin/fastrak_offload/common.h"
#include "tcpdirect_plugin/fastrak_offload/init.h"
#include "tcpdirect_plugin/fastrak_offload/nccl_shim.h"
#include "tcpdirect_plugin/fastrak_offload/nic_client_router.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_factory.h"
#include "tcpdirect_plugin/fastrak_offload/request.h"
#include "tcpdirect_plugin/nccl_compat/nccl_net_compat.h"

namespace fastrak {
namespace {

absl::Status TestNonnull(void* p, const char* repr) {
  if (p == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s may not be null.", repr));
  }
  return absl::OkStatus();
}

#define RETURN_IF_NULL(arg) \
  RETURN_IF_ERROR(TestNonnull(arg, #arg)).LogWarning().With(&ToNccl)

// An array of never-destroyed clients per-GPU. Init is expected to happen
// before, but may not be the same thread as, threads that call other functions.
//
// Inits for the same GPU are not allowed to happen concurrently.
std::array<std::mutex, kMaxGpuDevices> per_gpu_init_locks;
std::array<std::atomic<NcclShim*>, kMaxGpuDevices> per_gpu_clients;

// Initializes nccl shim on the provided cuda device.
absl::Status init_shim(int device) {
  std::lock_guard<std::mutex> lock(per_gpu_init_locks[device]);

  std::atomic<NcclShim*>& slot = per_gpu_clients[device];
  if (slot.load(std::memory_order_relaxed) != nullptr) return absl::OkStatus();
  ASSIGN_OR_RETURN(std::unique_ptr<NcclShim> shim,
                   NcclShim::Create(GetProfilerFactory), _.LogWarning());

  slot.store(shim.release(), std::memory_order_release);
  return absl::OkStatus();
}

ncclResult_t Init(ncclDebugLogger_t logFunction) {
  absl::Status init_status = fastrak::PluginCoreInit(logFunction);
  if (!init_status.ok()) {
    // Terminate the process on init failure to prevent
    // silent fallback to stock net socket plug-in.
    LOG(FATAL) << absl::StrFormat(
        "Plug-in core initialization failed (%s), aborting...",
        init_status.ToString());
  }

  auto device = GetDeviceIndex();
  if (!device.ok() || *device < 0 ||
      static_cast<size_t>(*device) >= kMaxGpuDevices) {
    LOG(FATAL) << absl::StrFormat(
        "Failed to get device index for current GPU %d, aborting...",
        device.ok() ? *device : -1);
  }

  LOG(INFO) << absl::StrFormat("Initialize nccl shim on device:%d during Init.",
                               *device);

  absl::Status shim_status = init_shim(*device);
  if (!shim_status.ok()) {
    LOG(FATAL) << absl::StrFormat(
        "Failed to initialize nccl shim on device:%d (%s), aborting...",
        *device, shim_status.ToString());
  }

  return ncclSuccess;
}

absl::StatusOr<NcclShim*> GetShim() {
  ASSIGN_OR_RETURN(int device, GetDeviceIndex(), _.LogWarning());
  if (device < 0 || static_cast<size_t>(device) >= kMaxGpuDevices) {
    return absl::InternalError(absl::StrFormat(
        "Failed to get device index for current GPU %d", device));
  }
  std::atomic<NcclShim*>& slot = per_gpu_clients[device];
  NcclShim* shim = slot.load(std::memory_order_acquire);
  if (shim != nullptr) {
    return shim;
  }
  // Nccl shim on the current GPU device is not initialized during
  // init; initialize the nccl shim ondemand.
  RETURN_IF_ERROR(init_shim(device));
  shim = slot.load(std::memory_order_acquire);
  LOG(INFO) << absl::StrFormat(
      "Initialized nccl shim on device:%d during GetShim.", device);
  return shim;
}

ncclResult_t Devices(int* ndev) {
  *ndev = NcclShim::Devices();
  return ncclSuccess;
}
ncclResult_t Listen(int dev, void* handle, void** listenComm) {
  ASSIGN_OR_RETURN(NcclShim * shim, GetShim(), _.LogWarning().With(&ToNccl));
  RETURN_IF_NULL(handle);
  RETURN_IF_NULL(listenComm);
  ASSIGN_OR_RETURN(auto comms, shim->Listen(dev), _.LogWarning().With(&ToNccl));
  // handle is a stack buffer and may not be properly aligned.
  std::memcpy(handle, &comms.socket_handle, sizeof(comms.socket_handle));
  *listenComm = comms.listen_comm.release();
  return ncclSuccess;
}

ncclResult_t Connect(int dev, void* opaqueHandle, void** sendComm,
                     ncclNetDeviceHandle_v7_t** sendDevComm) {
  ASSIGN_OR_RETURN(NcclShim * shim, GetShim(), _.LogWarning().With(&ToNccl));
  RETURN_IF_NULL(opaqueHandle);
  RETURN_IF_NULL(sendComm);
  RETURN_IF_NULL(sendDevComm);
  // opaqueHandle is byte-copied from the peer into a possibly non-aligned stack
  // buffer, copy its state in on call and out on return.
  ncclSocketHandle handle;
  memcpy(&handle, opaqueHandle, sizeof(handle));
  absl::Cleanup restore_handle = [&] {
    memcpy(opaqueHandle, &handle, sizeof(handle));
  };
  if (handle.version != kNcclSocketHandleVersion) {
    LOG(ERROR) << absl::StrFormat(
        "VERSION MISMATCH: Received Connect request with ncclSocketHandle "
        "version %d, but this plugin is version %d",
        handle.version, kNcclSocketHandleVersion);
    return ncclRemoteError;
  }
  if (handle.fastrak_idx == kFastrakInvalidIdx) {
    LOG(ERROR) << "SEVERE: FasTrak index is uninitialized in the "
                  "ncclSocketHandle. Remote rank is misbehaving.";
    return ncclRemoteError;
  }
  ASSIGN_OR_RETURN(std::optional<NcclShim::Comms> comms,
                   shim->Connect(dev, handle), _.LogWarning().With(&ToNccl));
  if (!comms.has_value()) return ncclSuccess;
  *sendComm = comms->comm.release();
  if (*sendDevComm != nullptr) **sendDevComm = comms->dev_comm;
  return ncclSuccess;
}

ncclResult_t Accept(void* listenComm, void** recvComm,
                    ncclNetDeviceHandle_v7_t** recvDevComm) {
  ASSIGN_OR_RETURN(NcclShim * shim, GetShim(), _.LogWarning().With(&ToNccl));
  RETURN_IF_NULL(listenComm);
  RETURN_IF_NULL(recvComm);
  RETURN_IF_NULL(recvDevComm);
  ASSIGN_OR_RETURN(std::optional<NcclShim::Comms> comms,
                   shim->Accept(*static_cast<ListenCommunication*>(listenComm)),
                   _.LogWarning().With(&ToNccl));
  if (!comms.has_value()) return ncclSuccess;
  *recvComm = comms->comm.release();
  if (*recvDevComm != nullptr) **recvDevComm = comms->dev_comm;
  return ncclSuccess;
}

ncclResult_t Isend(void* sendComm, void* data, int size, int tag,
                   void* opaqueHandle, void** opaqueRequest) {
  RETURN_IF_NULL(sendComm);
  RETURN_IF_NULL(opaqueHandle);
  RETURN_IF_NULL(opaqueRequest);
  auto* comm = static_cast<Communication*>(sendComm);
  ASSIGN_OR_RETURN(
      auto request,
      comm->shim->Isend(
          *comm, {.iov_base = data, .iov_len = static_cast<size_t>(size)}, tag,
          *static_cast<MemoryHandle*>(opaqueHandle)),
      _.LogWarning().With(&ToNccl));
  *opaqueRequest = request.release();
  return ncclSuccess;
}

ncclResult_t Irecv(void* recvComm, int n, void** data, int* sizes, int* tags,
                   void** opaqueHandles, void** opaqueRequest) {
  if (n != 1) {
    LOG(WARNING) << "PXN is not supported";
    return ncclSystemError;
  }
  RETURN_IF_NULL(recvComm);
  RETURN_IF_NULL(data);
  RETURN_IF_NULL(sizes);
  RETURN_IF_NULL(tags);
  RETURN_IF_NULL(opaqueHandles);
  RETURN_IF_NULL(opaqueHandles[0]);
  RETURN_IF_NULL(opaqueRequest);
  auto* comm = static_cast<Communication*>(recvComm);
  ASSIGN_OR_RETURN(
      auto request,
      comm->shim->Irecv(
          *comm,
          {.iov_base = data[0], .iov_len = static_cast<size_t>(sizes[0])},
          tags[0], *static_cast<MemoryHandle*>(opaqueHandles[0])),
      _.LogWarning().With(&ToNccl));
  *opaqueRequest = request.release();
  return ncclSuccess;
}

ncclResult_t Test(void* request, int* done, int* size) {
  RETURN_IF_NULL(request);
  RETURN_IF_NULL(done);
  auto* req = static_cast<ncclSocketRequest*>(request);
  if (req->error) {
    return ToNccl(absl::InternalError("Request is in error state"));
  }
  auto req_status = req->comm->shim->Test(*req);
  if (!req_status.ok()) {
    req->error = true;
  }
  ASSIGN_OR_RETURN(std::optional<int> result, req_status,
                   _.LogWarning().With(&ToNccl));
  *done = result.has_value();
  if (size) *size = result.value_or(0);
  return ncclSuccess;
}

ncclResult_t RegMr(void* regComm, void* data, int size, int type,
                   void** opaqueHandle) {
  RETURN_IF_NULL(regComm);
  RETURN_IF_NULL(opaqueHandle);
  auto* comm = static_cast<Communication*>(regComm);
  ASSIGN_OR_RETURN(
      std::unique_ptr<MemoryHandle> handle,
      comm->shim->RegMrDmaBuf(
          *comm, {.iov_base = data, .iov_len = static_cast<size_t>(size)}, type,
          0ULL, std::nullopt),
      _.LogWarning().With(&ToNccl));
  *opaqueHandle = handle.release();
  return ncclSuccess;
}

ncclResult_t RegMrDmaBuf(void* regComm, void* data, size_t size, int type,
                         uint64_t offset, int fd, void** opaqueHandle) {
  RETURN_IF_NULL(regComm);
  RETURN_IF_NULL(opaqueHandle);
  auto* comm = static_cast<Communication*>(regComm);
  ASSIGN_OR_RETURN(
      std::unique_ptr<MemoryHandle> handle,
      comm->shim->RegMrDmaBuf(
          *comm, {.iov_base = data, .iov_len = static_cast<size_t>(size)}, type,
          offset, fd),
      _.LogWarning().With(&ToNccl));
  *opaqueHandle = handle.release();
  return ncclSuccess;
}

ncclResult_t DeregMr(void* deregComm, void* opaqueHandle) {
  RETURN_IF_NULL(deregComm);
  RETURN_IF_NULL(opaqueHandle);
  auto* comm = static_cast<Communication*>(deregComm);
  RETURN_IF_ERROR(
      comm->shim->DeregMr(
          *comm, absl::WrapUnique(static_cast<MemoryHandle*>(opaqueHandle))))
      .LogWarning()
      .With(&ToNccl);
  return ncclSuccess;
}

ncclResult_t GetProperties(int dev, ncclNetProperties_v7_t* props) {
  RETURN_IF_NULL(props);
  ASSIGN_OR_RETURN(NcclShim * shim, GetShim(), _.LogWarning().With(&ToNccl));
  ASSIGN_OR_RETURN(*props, shim->GetProperties(dev),
                   _.LogWarning().With(&ToNccl));
  return ncclSuccess;
}

ncclResult_t Close(void* opaqueComm) {
  RETURN_IF_NULL(opaqueComm);
  auto* comm = static_cast<Communication*>(opaqueComm);
  comm->shim->Close(absl::WrapUnique(comm));
  return ncclSuccess;
}

ncclResult_t CloseListen(void* opaqueListenComm) {
  RETURN_IF_NULL(opaqueListenComm);
  ASSIGN_OR_RETURN(NcclShim * shim, GetShim(), _.LogWarning().With(&ToNccl));
  RETURN_IF_ERROR(shim->CloseListen(absl::WrapUnique(
                      static_cast<ListenCommunication*>(opaqueListenComm))))
      .LogWarning()
      .With(&ToNccl);
  return ncclSuccess;
}

ncclResult_t GetDeviceMr(void* opaqueComm, void* mhandle, void** dptr_mhandle) {
  // Not implemented.
  return ncclSuccess;
}

ncclResult_t Iflush(void* recvComm, int n, void** data, int* sizes,
                    void** mhandle, void** request) {
  // Not implemented.
  return ncclSuccess;
}

}  // namespace

// Helper for testing.
absl::StatusOr<NcclShim*> TestonlyGetNcclShim() { return GetShim(); }

void ResetNcclShim() {
  for (auto& slot : per_gpu_clients) {
    delete slot.exchange(nullptr, std::memory_order_acq_rel);
  }
}

void ResetNetIfs() {
  for (int i = 0; i < MAX_IFS; i++) {
    if (kNcclSocketDevs[i].mr_cache.slots != nullptr) {
      free(kNcclSocketDevs[i].mr_cache.slots);
      kNcclSocketDevs[i].mr_cache.slots = nullptr;
    }
    kNcclSocketDevs[i].mr_cache.capacity = 0;
    kNcclSocketDevs[i].mr_cache.population = 0;
  }
}

}  // namespace fastrak

extern "C" {

__attribute__((destructor)) void cleanup() {
  fastrak::ResetNcclShim();
  fastrak::ResetNetIfs();
  fastrak::GetNicClientRouter().Shutdown();
}

volatile ncclNet_v7_t ncclNetPlugin_v7 = {
    "FasTrak",
    fastrak::Init,
    fastrak::Devices,
    fastrak::GetProperties,
    fastrak::Listen,
    fastrak::Connect,
    fastrak::Accept,
    fastrak::RegMr,
    fastrak::RegMrDmaBuf,
    fastrak::DeregMr,
    fastrak::Isend,
    fastrak::Irecv,
    fastrak::Iflush,
    fastrak::Test,
    fastrak::Close,
    fastrak::Close,
    fastrak::CloseListen,
    fastrak::GetDeviceMr,
    /*irecvConsumed=*/nullptr,
};

}  // extern "C"

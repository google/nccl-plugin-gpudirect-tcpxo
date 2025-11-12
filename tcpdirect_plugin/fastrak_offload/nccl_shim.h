/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_NCCL_SHIM_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_NCCL_SHIM_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "nccl_cuda/cuda_defs.h"
#include "nccl_device/net_device.h"
#include "tcpdirect_plugin/fastrak_offload/common.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_factory.h"
#include "tcpdirect_plugin/fastrak_offload/request.h"
#include "tcpdirect_plugin/nccl_compat/nccl_net_compat.h"

namespace fastrak {

// The main entrypoint that matches the NCCL APIs but is type-safe and testable.
//
// Thread safe.
class NcclShim {
 public:
  // Main entrypoint for NCCL.
  static absl::StatusOr<std::unique_ptr<NcclShim>> Create(
      std::function<std::unique_ptr<ProfilerFactoryInterface>(
          const ProfilerFactoryOptions&)>
          createProfilerFactory);

  static int Devices();

  absl::StatusOr<ncclNetProperties_v7_t> GetProperties(int dev);

  struct ListenComms {
    std::unique_ptr<ListenCommunication> listen_comm;
    ncclSocketHandle socket_handle;
  };

  absl::StatusOr<ListenComms> Listen(int dev);

  struct Comms {
    std::unique_ptr<Communication> comm;
    ncclNetDeviceHandle_v7_t dev_comm;
  };

  absl::StatusOr<std::optional<Comms>> Connect(int dev,
                                               ncclSocketHandle& handle);

  absl::StatusOr<std::optional<Comms>> Accept(ListenCommunication& listenComm);

  absl::StatusOr<absl_nonnull std::unique_ptr<ncclSocketRequest>> Isend(
      Communication& comm, iovec data, int tag, MemoryHandle& mhandle);

  absl::StatusOr<absl_nonnull std::unique_ptr<ncclSocketRequest>> Irecv(
      Communication& comm, iovec data, int tag, MemoryHandle& mhandle);

  absl::StatusOr<std::optional<int>> Test(ncclSocketRequest& request);

  void Close(std::unique_ptr<Communication> comm);

  absl::StatusOr<std::unique_ptr<MemoryHandle>> RegMrDmaBuf(
      Communication& comm, iovec data, int type, uint64_t offset,
      std::optional<int> fd);

  absl::Status DeregMr(Communication& comm,
                       std::unique_ptr<MemoryHandle> mhandle);

  absl::Status CloseListen(std::unique_ptr<ListenCommunication> listenComm);

  const gpuDev& GetGpuDev() { return gpu_; }

 private:
  // Raw constructor.
  NcclShim(gpuDev gpu, int netdev, uint8_t fastrak_idx,
           std::unique_ptr<ProfilerFactoryInterface> profiler_factory);

  absl::StatusOr<dxs::DxsClientInterface* absl_nonnull> ListenDxs();
  absl::StatusOr<tcpdirect::BufferManagerClientInterface* absl_nonnull>
  ListenBufferManager();
  // Internal implementation of RegMrDmaBuf that actually registers the memory.
  absl::StatusOr<MemoryHandle> RegMrDmaBufInternal(Communication& comm,
                                                   iovec data, int offset,
                                                   std::optional<int> fd);
  absl::Status DeregMrInternal(Communication& comm,
                               std::unique_ptr<MemoryHandle> mhandle);

  const gpuDev gpu_;
  const int netdev_;
  const uint8_t fastrak_idx_;
  // Obtained from environment and constructed once.
  const absl::Duration connect_timeout_;
  const absl::Duration accept_timeout_;
  const absl::Duration data_transfer_timeout_;

  uint32_t comm_counter_ = 0;

  // A listen should complete almost immediately, so no need
  // for a programmable param. This is just to catch a deadlock and log it.
  static const int kListenTimeoutMs = 10000;

  std::unique_ptr<ProfilerFactoryInterface> profiler_factory_;
};

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_NCCL_SHIM_H_

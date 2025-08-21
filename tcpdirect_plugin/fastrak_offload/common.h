/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

/**
 * Header file for all general, non-GPU-related data structure declarations and
 * internal data members.
 */
#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_COMMON_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_COMMON_H_

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/uio.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "nccl.h"
#include "nccl_cuda/cuda_defs.h"
#include "tcpdirect_plugin/fastrak_offload/params.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_factory.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"
#include "tcpdirect_plugin/fastrak_offload/stats.h"

/* Sparing for potential 16+1 NICs machine in the future */
#define MAX_IFS 17
#define MAX_IF_NAME_SIZE 16

namespace fastrak {

inline constexpr size_t kIpAddrMaxLen = 40;
inline constexpr uint8_t kNcclSocketHandleVersion = 1;
inline constexpr uint8_t kFastrakInvalidIdx =
    std::numeric_limits<uint8_t>::max();

using CommTraceId = uint32_t;

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

struct MemoryHandle {
  // Used for memory registered with DXS only, dxs::kInvalidRegistration
  // otherwise
  dxs::Reg reg_handle = dxs::kInvalidRegistration;
  int mem_type = 0;            // memory type: CPU/GPU
  void* start_addr = nullptr;  // starting addr of memory region
  int size;
  int dmabuf_fd = -1;  // dmabuf fd used by the memory
};

struct ncclMr {
  struct MemoryHandle handle;
  int pages;
  int refs;
};

struct ncclMrCache {
  struct ncclMr* slots = nullptr;
  int capacity = 0;
  int population = 0;
};

struct ncclSocketDev {
  pthread_mutex_t reg_mutex;
  socketAddress addr;
  char ip_addr[kIpAddrMaxLen];
  char dev_name[MAX_IF_NAME_SIZE];
  char* pci_path;
  struct ncclMrCache mr_cache;
};

enum class ncclSocketCommState : uint8_t {
  kStart = 0,
  kConnecting = 1,
  kAccepting = 2,
  kDone = 3,
};

struct ncclSocketCommStage {
  ncclSocketCommState state = ncclSocketCommState::kStart;
  Communication* comm = nullptr;
};

struct ListenCommunication {
  int nccl_listen = -1;
  std::unique_ptr<dxs::ListenSocketInterface> dxs_listen[kFastrakMaxNumFlows];
  const gpuDev* gpu = nullptr;
  ncclSocketCommStage stage;
  socketAddress nccl_addr;
};

struct ncclSocketHandle {
  // Bump whenever this struct changes.
  uint8_t version = kNcclSocketHandleVersion;
  // The order in which the GPUs are enumerated by the host. Expected to be
  // consistent across all FasTrak deployments:
  uint8_t fastrak_idx = kFastrakInvalidIdx;
  socketAddress nccl_addr;
  char dxs_addr[kIpAddrMaxLen];
  // Subject to changes, might be up to 128 bits
  uint16_t dxs_ports[kFastrakMaxNumFlows];
  ncclSocketCommStage stage;
};

// Forward declare NcclShim to access it without needing to do a device lookup.
class NcclShim;

// A Communication object handed off to NCCL.
struct Communication {
  explicit Communication(NcclShim* shim,
                         ProfilerFactoryInterface& profiler_factory,
                         const gpuDev* gpu, uint32_t idx, bool send)
      : shim(shim),
        gpu(gpu),
        idx(idx),
        send(send),
        profiler_factory(profiler_factory),
        stats{gpu->pci_addr, idx, send} {}
  // Cached shim to avoid the need to look up on every send, recv and test.
  NcclShim* const shim;
  const gpuDev* const gpu;
  // The n-th comm object created for the current GPU.
  const uint32_t idx;
  const bool send;
  ProfilerFactoryInterface& profiler_factory;
  std::unique_ptr<ProfilerInterface> profilers[kFastrakMaxNumFlows];
  fastrak::CommStats stats;

  // Data socket for connections on DXS
  std::unique_ptr<dxs::ConnectedSocketInterface> dxs_socks[kFastrakMaxNumFlows];
  // unowned buffer manager client
  tcpdirect::BufferManagerClientInterface* buffer_manager = nullptr;
  fastrak::Timer connection_timeout;
  // Time since the Isend/Irecv call that resulted
  // in the first metadata transfer attempt for this request.
  fastrak::Timer last_xfer;
  // State used to prevent resetting the last_xfer timer
  // on subsequent Isend/Irecv calls due to metadata retries.
  bool ready_for_ts_capture = true;
  // This must remain synchronized between send and recv.
  // Gets incremented for each op. Contains the index
  // of the first flow of the current group.
  int curr_flow_group_base = 0;
  dxs::DxsClientInterface* dxs = nullptr;
  // The dev actually used. Can be different from the `dev` passed in by NCCL
  // due to rail alignment.
  int selected_dev = -1;
  // The unique connection identifier correlating the Comm object between
  // Tx and Rx sides.
  // Upon Tx side Connect(), plugin will generate `comm_trace_id` and
  // send it to Rx side via control channel.
  // Upon Rx side Accept(), plugin will receive this `comm_trace_id` and
  // save as the `comm_trace_id`.
  CommTraceId trace_id{0};
  char dxs_addr[kIpAddrMaxLen];
  // Subject to changes, might be up to 128 bits
  uint16_t dxs_ports[kFastrakMaxNumFlows];
};

extern int kNcclNetIfs;

extern struct ncclSocketDev kNcclSocketDevs[MAX_IFS];

/* Netdev used for control-level traffic.
   In order to specify the netdevice to use, user should set the environment
   NCCL_FASTRAK_CTRL_DEV to the interface specified, otherwise no specific
   binding on the net if will be done during connection establishment. */
extern struct ncclSocketDev kNcclCtrlSocketDev;

/**
 * Main function for discovering all network interfaces and recording them for
 * later network workloads.
 *
 * For FasTrak, this discovers and records fabric NICs for offload traffic, and
 * dcn NIC for onload traffic / ctrl msgs.
 */
ncclResult_t initializeNetIfs();

// Checks if an IP is discovered and recorded during init
ncclResult_t ipAddrDiscovered(const char* ip_addr, int* idx);

// Given a GPU PCI DBDF, return the idx of the closest NIC to that GPU.
absl::StatusOr<int> getClosestNetdev(std::string_view gpu_pci);

// Read the network speed of a particular netdev
int GetSpeed(std::string_view dev_name);

absl::StatusOr<uint8_t> GetFastrakIdxFromPci(absl::string_view pci_addr);
absl::StatusOr<std::string> GetPciFromFastrakIdx(uint8_t fastrak_idx);
// Test only function; useful to populate the map if sysfs walking isn't viable.
absl::flat_hash_map<std::string, uint8_t>& TestOnlyGetPciAddrToFastrakIdxMap();

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_COMMON_H_

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/nccl_shim.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/control-command.h"
#include "dxs/client/derive_dxs_address.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/oss/status_macros.h"
#include "nccl.h"
#include "nccl_cuda/cuda_common.h"
#include "nccl_cuda/cuda_defs.h"
#include "net_device.h"
#include "tcpdirect_plugin/fastrak_offload/common.h"
#include "tcpdirect_plugin/fastrak_offload/nic_client_router.h"
#include "tcpdirect_plugin/fastrak_offload/params.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_factory.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"
#include "tcpdirect_plugin/fastrak_offload/request.h"
#include "tcpdirect_plugin/fastrak_offload/stats.h"
#include "tcpdirect_plugin/fastrak_offload/syscalls.h"
#include "tcpdirect_plugin/fastrak_offload/utilities.h"
#include "tcpdirect_plugin/nccl_compat/nccl_net_compat.h"

namespace fastrak {
namespace {

constexpr size_t kRepresentativeFlowIdx = 0;

absl::Status BeginDxsConnect(int dev, ncclSocketHandle& handle,
                             Communication& comm,
                             dxs::DxsClientInterface& dxs) {
  for (auto i = 0; i < kFastrakNumFlows; ++i) {
    LOG(INFO) << absl::StrFormat(
        "DXS flow ID %d connecting to %s:%d "
        "comm_trace_id:0x%lx "
        "comm_idx:0x%lx on net_dev:%d",
        i, handle.dxs_addr, handle.dxs_ports[i], comm.trace_id, comm.idx, dev);
    ASSIGN_OR_RETURN(comm.dxs_socks[i],
                     dxs.Connect(handle.dxs_addr, handle.dxs_ports[i]));
  }
  // Store remote DXS addr and ports
  strncpy(comm.dxs_addr, handle.dxs_addr, sizeof(comm.dxs_addr));
  memcpy(comm.dxs_ports, handle.dxs_ports, kFastrakNumFlows * sizeof(uint16_t));
  for (auto i = 0; i < kFastrakNumFlows; ++i) {
    std::string local_ip_addr =
        static_cast<dxs::SendSocketInterface*>(comm.dxs_socks[i].get())
            ->Address();
    uint16_t local_port = 0;
    std::string remote_ip_addr = comm.dxs_addr;
    uint16_t remote_port = comm.dxs_ports[i];
    comm.profilers[i] =
        comm.profiler_factory.Create(&comm, {.local_ip_addr = local_ip_addr,
                                             .remote_ip_addr = remote_ip_addr,
                                             .local_port = local_port,
                                             .remote_port = remote_port,
                                             .flow_id = (uint32_t)i});
  }
  return absl::OkStatus();
}

std::string GetSocketHandleInfo(const ncclSocketHandle& handle, int flow_id) {
  return absl::StrFormat("FasTrak listening on %s:%d for flow_id=%d",
                         handle.dxs_addr, handle.dxs_ports[flow_id], flow_id);
}

// Returns important NCCL comm info, including:
// Comm address and trace_id
// For send comm: local DXS addr, remote DXS addr and port
// For recv comm: local DXS addr and port
// If flow_id is a valid number, then we dump the single port for that flow.
// Otherwise, we dump all ports.
// Number of scheduled reqs and completed reqs so far.
std::string GetNcclCommInfo(const struct Communication& comm) {
  std::string basic_comm_info = absl::StrFormat(
      "%s comm:%p comm_idx:0x%lx comm_trace_id:0x%lx",
      comm.send ? "Send" : "Recv", &comm, comm.idx, comm.trace_id);
  // port1 port2 ...
  std::string dxs_port_info =
      absl::StrJoin(absl::MakeSpan(comm.dxs_ports, kFastrakNumFlows), " ");
  std::string dxs_info;
  if (comm.send) {
    auto sock = static_cast<dxs::SendSocketInterface*>(comm.dxs_socks[0].get());
    dxs_info = absl::StrFormat(
        "local_dxs_addr:%s remote_dxs_addr:%s remote_dxs_ports:[%s]",
        sock->Address(), comm.dxs_addr, dxs_port_info);

    // Log iff the peer is set.
    if (sock->Peer() != dxs::WireSocketAddr{}) {
      auto peer_addr = dxs::UnpackIpAddress(sock->Peer()).value_or("");
      dxs_info.append(absl::StrFormat(
          "\nSend connection established from sender (%s:%d) to receiver "
          "(%s:%d).\n ",
          sock->Address(), 0, peer_addr, sock->Peer().port));
    }
  } else {
    auto sock = static_cast<dxs::RecvSocketInterface*>(comm.dxs_socks[0].get());
    dxs_info = absl::StrFormat("local_dxs_addr:%s local_dxs_ports:[%s]",
                               comm.dxs_addr, dxs_port_info);

    // Log iff the peer is set.
    if (sock->Peer() != dxs::WireSocketAddr{}) {
      auto peer_addr = dxs::UnpackIpAddress(sock->Peer()).value_or("");
      dxs_info.append(absl::StrFormat(
          "Receive connection established from sender (%s:%d) "
          "to receiver (%s:[%s])",
          peer_addr, sock->Peer().port, comm.dxs_addr, dxs_port_info));
    }
  }
  auto req_stats =
      absl::StrFormat("num_scheduled_reqs:%d, num_completed_reqs:%d",
                      comm.stats.request.offload_scheduled,
                      comm.stats.request.offload_completed);
  return absl::StrCat(basic_comm_info, "\n", dxs_info, "\n", req_stats);
}

}  // namespace

NcclShim::NcclShim(gpuDev gpu, int netdev, uint8_t fastrak_idx,
                   std::unique_ptr<ProfilerFactoryInterface> profiler_factory)
    : gpu_(std::move(gpu)),
      netdev_(netdev),
      fastrak_idx_(fastrak_idx),
      connect_timeout_(absl::Milliseconds(kFastrakPluginConnectTimeoutMs)),
      accept_timeout_(absl::Milliseconds(kFastrakPluginAcceptTimeoutMs)),
      data_transfer_timeout_(absl::Milliseconds(kFastrakDataTransferTimeoutMs)),
      profiler_factory_(std::move(profiler_factory)) {
  LOG(INFO) << absl::StrFormat("Using %ld total flows", kFastrakNumFlows);
}

absl::StatusOr<std::unique_ptr<NcclShim>> NcclShim::Create(
    std::function<std::unique_ptr<ProfilerFactoryInterface>(
        const ProfilerFactoryOptions&)>
        createProfilerFactory) {
  if (!kEnableFasTrak) {
    return absl::FailedPreconditionError("NET/FasTrak disabled");
  }
  RETURN_IF_ERROR(checkDeviceCount(kMaxGpuDevices));

  ASSIGN_OR_RETURN(gpuDev gpu, initGpuDev());
  ASSIGN_OR_RETURN(auto fastrak_idx, GetFastrakIdxFromPci(gpu.pci_addr));
  LOG(INFO) << absl::StrFormat(
      "NcclShim::Create: GPU PCI addr: %s, FasTrak idx: %d", gpu.pci_addr,
      fastrak_idx);
  absl::StatusOr<int> netdev_or = getClosestNetdev(gpu.pci_addr);
  int netdev = netdev_or.value_or(-1);
  if (!netdev_or.ok()) {
    // Testing-only option to debug single NIC errors. If enabled, don't fail
    // if the NIC is not found.
    if (!kFastrakAllowNicSubset) return netdev_or.status();
  } else {
    LOG(INFO) << absl::StrFormat(
        "Initialized shim for GPU with CUDA index %d, FasTrak index %d, and "
        "pci_path: %s, the corresponding closest net device is: %d (%s)",
        gpu.dev, fastrak_idx, gpu.pci_addr, netdev,
        kNcclSocketDevs[netdev].dev_name);
    const char* ip_addr = kNcclSocketDevs[netdev].ip_addr;
    RETURN_IF_ERROR(GetNicClientRouter().GetDxsClient(ip_addr).status());
    LOG(INFO) << absl::StrFormat(
        "DXS Client initialized on IP addr "
        "%s",
        ip_addr);
    RETURN_IF_ERROR(
        GetNicClientRouter().GetBufferManagerClient(ip_addr).status());
    LOG(INFO) << absl::StrFormat(
        "Buffer Manager Client initialized on IP addr %s", ip_addr);
  }
  std::unique_ptr<ProfilerFactoryInterface> profiler_factory =
      createProfilerFactory(ProfilerFactoryOptions{
          .gpu_dev = gpu,
          .netdev = netdev,
      });
  LOG(INFO) << absl::StrFormat("Plugin initialized for %s.", gpu.pci_addr);
  return absl::WrapUnique(new NcclShim(std::move(gpu), netdev, fastrak_idx,
                                       std::move(profiler_factory)));
}

absl::StatusOr<dxs::DxsClientInterface* absl_nonnull> NcclShim::ListenDxs() {
  return GetNicClientRouter().GetDxsClient(kNcclSocketDevs[netdev_].ip_addr);
}

absl::StatusOr<tcpdirect::BufferManagerClientInterface* absl_nonnull>
NcclShim::ListenBufferManager() {
  return GetNicClientRouter().GetBufferManagerClient(
      kNcclSocketDevs[netdev_].ip_addr);
}

int NcclShim::Devices() { return kNcclNetIfs; }

ncclNetDeviceHandle_v7_t createSendDevComm(Communication& netComm) {
  LOG(INFO) << absl::StrFormat("Allocate send dev handle, comm %p", &netComm);
  ncclNetDeviceHandle_v7_t devComm;
  devComm.netDeviceType = NCCL_NET_DEVICE_HOST;
  devComm.netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
  devComm.handle = nullptr;
  devComm.size = 0;
  devComm.needsProxyProgress = 0;
  LOG(INFO) << absl::StrFormat("Comm %p, createSendDevComm %p", &netComm,
                               &devComm);
  return devComm;
}

absl::StatusOr<ncclNetDeviceHandle_v7_t> createRecvDevComm(
    Communication& netComm) {
  LOG(INFO) << absl::StrFormat(
      "Allocate recv dev handle for comm:%p on gpu_pci:%s", &netComm,
      netComm.gpu->pci_addr);
  ncclNetDeviceHandle_v7_t devComm;
  devComm.netDeviceType = NCCL_NET_DEVICE_HOST;
  devComm.netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
  devComm.handle = nullptr;
  devComm.size = 0;
  devComm.needsProxyProgress = 0;
  LOG(INFO) << absl::StrFormat("Linearized: comm %p, createRecvDevComm %p",
                               &netComm, &devComm);
  return devComm;
}
absl::StatusOr<NcclShim::ListenComms> NcclShim::Listen(int dev) {
  // Check if dev number is valid
  if (dev < 0 || dev >= kNcclNetIfs) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid dev %d", dev));
  }
  // WARN about misusage, but no direct aborts
  if (dev != netdev_) {
    LOG(INFO) << absl::StrFormat(
        "Netdev %d is specified for connection to netdev %d.", dev, netdev_);
  }
  static_assert(sizeof(ncclSocketHandle) <= NCCL_NET_HANDLE_MAXSIZE,
                "NCCL socket handle size too large");
  ncclSocketHandle socketHandle;
  /* Always use host NIC for NCCL-to-NCCL traffic. */
  socketHandle.nccl_addr = kNcclCtrlSocketDev.addr;

  /* Use host NIC for NCCL-to-NCCL traffic. */
  socketHandle.nccl_addr = kNcclCtrlSocketDev.addr;

  auto comm = std::make_unique<ListenCommunication>();
  comm->nccl_addr = socketHandle.nccl_addr;
  ASSIGN_OR_RETURN(comm->nccl_listen,
                   createListenSocket(socketHandle.nccl_addr));
  comm->gpu = &gpu_;
  ASSIGN_OR_RETURN(auto* listen_dxs, ListenDxs());
  // It should be possible to use a single listen for all flows,
  // but connection stalls were encountered with this approach,
  // so create one listen for each flow for now.
  for (auto i = 0; i < kFastrakNumFlows; ++i) {
    /* DXS listen*/
    ASSIGN_OR_RETURN(comm->dxs_listen[i], listen_dxs->Listen());
  }
  for (auto i = 0; i < kFastrakNumFlows; ++i) {
    int port = 0;
    Timer timer;
    while (port == 0) {
      RETURN_IF_ERROR(timer.CheckTimeout(absl::Milliseconds(kListenTimeoutMs)))
          << "Listen timed out for Flow ID " << i;
      std::optional<absl::Status> status = comm->dxs_listen[i]->SocketReady();
      if (status.has_value()) {
        RETURN_IF_ERROR(*std::move(status));
        port = comm->dxs_listen[i]->Port();
      }
    }
    socketHandle.dxs_ports[i] = port;
    LOG(INFO) << absl::StrFormat(
        "DXS listen socket ready for flow ID %d, port %d", i, port);
  }

  // Set the IP address of the NIC
  // Each listen will have the same IP but different port, so just use
  // listen 0 IP.
  std::string ip_addr = comm->dxs_listen[0]->Address();
  if (snprintf(socketHandle.dxs_addr, kIpAddrMaxLen, "%s", ip_addr.c_str()) <
      0) {
    return absl::InternalError(
        absl::StrFormat("ip_addr is too long %s", ip_addr));
  }
  LOG(INFO) << absl::StrFormat(
      "DXS listening on %s (%s) for GPU %s, ctrl listening on %s",
      socketHandle.dxs_addr, kNcclSocketDevs[netdev_].dev_name, gpu_.pci_addr,
      socketToString(&socketHandle.nccl_addr));
  socketHandle.stage.state = ncclSocketCommState::kStart;
  socketHandle.fastrak_idx = fastrak_idx_;
  comm->stage.state = ncclSocketCommState::kStart;
  return ListenComms{std::move(comm), std::move(socketHandle)};
}

// Generates the unique 32-bit connection identifier correlating Tx and Rx side.
// 0 - 15 bit: comm_idx
// 16 - 31 bit: comm identifier.
CommTraceId GenerateCommTraceId(void* comm, uint32_t comm_idx) {
  uint32_t comm_trace_id =
      (0xffff & comm_idx) | ((0xffff & reinterpret_cast<uint64_t>(comm)) << 16);
  return static_cast<CommTraceId>(comm_trace_id);
}

absl::StatusOr<std::optional<NcclShim::Comms>> NcclShim::Connect(
    int dev, ncclSocketHandle& handle) {
  if (dev < 0 || dev >= kNcclNetIfs) {
    return absl::InvalidArgumentError(absl::StrFormat("Invalid dev %d", dev));
  }
  ncclSocketCommStage& stage = handle.stage;
  Communication* comm = stage.comm;
  switch (stage.state) {
    case ncclSocketCommState::kStart: {
      VLOG(1) << "NcclShim::Connect: Requested FasTrak Index: "
              << static_cast<int>(handle.fastrak_idx)
              << ". This NCCL Shim is initialized for FasTrak Index: "
              << static_cast<int>(fastrak_idx_);

      ASSIGN_OR_RETURN(const auto gpu_pci_address_for_requested_idx,
                       GetPciFromFastrakIdx(handle.fastrak_idx));
      if (kNicRailAligned) {
        ASSIGN_OR_RETURN(int rail_aligned_netdev,
                         getClosestNetdev(gpu_pci_address_for_requested_idx));
        if (rail_aligned_netdev != dev) {
          LOG(INFO) << absl::StrFormat(
              "Use netdev %d (%s) instead of %d (%s) to enforce rail alignment "
              "for connection for GPU with local FasTrak index [%d] to remote "
              "FasTrak index [%d]",
              rail_aligned_netdev,
              kNcclSocketDevs[rail_aligned_netdev].dev_name, dev,
              kNcclSocketDevs[dev].dev_name, fastrak_idx_, handle.fastrak_idx);
          dev = rail_aligned_netdev;
        }
      }

      if (handle.fastrak_idx != fastrak_idx_) {
        LOG(INFO) << absl::StrFormat(
            "Connecting to non-rail-aligned GPU (%d (local) -> %d (remote)) "
            "using %s",
            fastrak_idx_, handle.fastrak_idx, kNcclSocketDevs[dev].dev_name);
      }

      stage.comm = new Communication(this, *profiler_factory_, &gpu_,
                                     ++comm_counter_, true);
      comm = stage.comm;
      ASSIGN_OR_RETURN(comm->dxs, GetNicClientRouter().GetDxsClient(
                                      kNcclSocketDevs[dev].ip_addr));
      comm->selected_dev = dev;
      ASSIGN_OR_RETURN(comm->buffer_manager,
                       GetNicClientRouter().GetBufferManagerClient(
                           kNcclSocketDevs[dev].ip_addr));
      comm->trace_id = GenerateCommTraceId(comm, comm->idx);
      RETURN_IF_ERROR(BeginDxsConnect(dev, handle, *comm, *comm->dxs));
      stage.state = ncclSocketCommState::kConnecting;
      ABSL_FALLTHROUGH_INTENDED;
    }
    case ncclSocketCommState::kConnecting: {
      for (auto i = 0; i < kFastrakNumFlows; ++i) {
        std::optional<absl::Status> ready =
            static_cast<dxs::SendSocketInterface*>(comm->dxs_socks[i].get())
                ->SocketReady();
        if (!ready.has_value()) {
          RETURN_IF_ERROR(
              comm->connection_timeout.CheckTimeout(connect_timeout_))
              << absl::StrFormat(
                     "DXS flow ID %d connect "
                     "timeout comm_trace_id:0x%lx.\nLocal DXS addr:%s\n%s",
                     i, comm->trace_id,
                     kNcclSocketDevs[comm->selected_dev].ip_addr,
                     GetSocketHandleInfo(handle, i));
          return std::nullopt;
        }
        RETURN_IF_ERROR(*ready) << absl::StrFormat(
            "DXS::SocketReady for flow ID %d failed "
            "comm_trace_id:0x%lx.\nLocal DXS addr: %s.\n%s",
            i, comm->trace_id, kNcclSocketDevs[comm->selected_dev].ip_addr,
            GetSocketHandleInfo(handle, i));
      }
      LOG(LEVEL(getConnectionLogLevel())) << absl::StrFormat(
          "All DXS flows connected to %s comm:%p "
          "comm_trace_id:0x%lx "
          "comm_idx:0x%lx on net_dev:%d (Local IP: %s)",
          handle.dxs_addr, comm, comm->trace_id, comm->idx, dev,
          kNcclSocketDevs[dev].ip_addr);
      stage.state = ncclSocketCommState::kDone;
      return Comms{.comm = absl::WrapUnique(std::exchange(stage.comm, nullptr)),
                   .dev_comm = createSendDevComm(*comm)};
    }
    default:
      return absl::InternalError(
          absl::StrFormat("NcclShim::Connect should not be called in state %d",
                          static_cast<int>(stage.state)));
  }
}

absl::StatusOr<std::optional<NcclShim::Comms>> NcclShim::Accept(
    ListenCommunication& listenComm) {
  ncclSocketCommStage* stage = &listenComm.stage;
  Communication* rComm = stage->comm;
  switch (stage->state) {
    case ncclSocketCommState::kStart: {
      stage->comm = new Communication(this, *profiler_factory_, listenComm.gpu,
                                      ++comm_counter_, false);
      rComm = stage->comm;
      rComm->selected_dev = netdev_;
      ASSIGN_OR_RETURN(rComm->buffer_manager, ListenBufferManager());
      LOG(INFO) << absl::StrFormat(
          "Async accepting for comm:%p comm_idx:0x%lx.", rComm, rComm->idx);

      stage->state = ncclSocketCommState::kAccepting;
      ABSL_FALLTHROUGH_INTENDED;
    }
    case ncclSocketCommState::kAccepting: {
      const auto addr_len = sizeof(rComm->dxs_addr);
      strncpy(rComm->dxs_addr, listenComm.dxs_listen[0]->Address().c_str(),
              addr_len - 1);
      rComm->dxs_addr[addr_len - 1] = '\0';
      // We are not doing the assignment below to avoid redundant assignments.
      for (auto i = 0; i < kFastrakNumFlows; ++i) {
        rComm->dxs_ports[i] = listenComm.dxs_listen[i]->Port();
      }
      for (auto i = 0; i < kFastrakNumFlows; ++i) {
        auto& sock = rComm->dxs_socks[i];
        if (sock == nullptr) {
          ASSIGN_OR_RETURN(
              sock, listenComm.dxs_listen[i]->Accept(),
              _ << absl::StrFormat("\nDXS flow ID %d failed to accept comm:%p "
                                   "comm_trace_id:0x%lx "
                                   "listen_addr:%s listen_port:%d",
                                   i, rComm, rComm->trace_id, rComm->dxs_addr,
                                   rComm->dxs_ports[i]));
        }
        if (sock == nullptr || !sock->SocketReady()) {
          RETURN_IF_ERROR(
              rComm->connection_timeout.CheckTimeout(accept_timeout_))
              << absl::StrFormat(
                     "DXS flow ID %d accept timeout comm:%p "
                     "comm_trace_id:0x%lx listen_addr:%s listen_port:%d",
                     i, rComm, rComm->trace_id, rComm->dxs_addr,
                     rComm->dxs_ports[i]);
          return std::nullopt;
        }
        RETURN_IF_ERROR(*sock->SocketReady()) << absl::StrFormat(
            "\nDXS flow ID %d failed to establish comm:%p "
            "comm_trace_id:0x%lx "
            "listen_addr:%s listen_port:%d",
            i, rComm, rComm->trace_id, rComm->dxs_addr, rComm->dxs_ports[i]);
      }
      // Create profilers for each flow only when the connections are
      // established.
      for (auto i = 0; i < kFastrakNumFlows; ++i) {
        std::string remote_ip_addr = "0.0.0.0";
        uint16_t local_port = 0;
        auto recv_sock =
            static_cast<dxs::RecvSocketInterface*>(rComm->dxs_socks[i].get());
        if (recv_sock->Peer() != dxs::WireSocketAddr{}) {
          remote_ip_addr = dxs::UnpackIpAddress(recv_sock->Peer()).value_or("");
          local_port = recv_sock->Peer().port;
        }
        std::string local_ip_addr = rComm->dxs_addr;
        uint16_t remote_port = rComm->dxs_ports[i];
        rComm->profilers[i] = rComm->profiler_factory.Create(
            rComm, {.local_ip_addr = local_ip_addr,
                    .remote_ip_addr = remote_ip_addr,
                    .local_port = local_port,
                    .remote_port = remote_port,
                    .flow_id = (uint32_t)i});
      }
      stage->state = ncclSocketCommState::kDone;
      LOG(INFO) << absl::StrFormat(
          "All DXS flows accepted for comm:%p comm_trace_id:0x%lx on "
          "gpu_pci:%s on comm_idx:%llu.",
          rComm, rComm->trace_id, rComm->gpu->pci_addr, rComm->idx);
      ASSIGN_OR_RETURN(auto dev_comm, createRecvDevComm(*rComm));
      return Comms{absl::WrapUnique(std::exchange(stage->comm, nullptr)),
                   std::move(dev_comm)};
    }
    default: {
      return absl::InternalError(
          absl::StrFormat("NcclShim::Connect should not be called in state %d",
                          static_cast<int>(stage->state)));
    }
  }
}

namespace {

absl::StatusOr<absl_nonnull std::unique_ptr<ncclSocketRequest>> GetRequest(
    Communication& comm, iovec data, MemoryHandle& mhandle) {
  if (data.iov_len != 0 && (mhandle.mem_type & NCCL_PTR_HOST)) {
    // If the control channel is disabled, then it is possible for
    // 0 byte transfer requests to be issued to DXS. In this case,
    // it probably doesn't matter whether it's from a host or cuda pointer?
    return absl::InvalidArgumentError(
        absl::StrFormat("Attempted to %s %d bytes from host buffer on comm %p",
                        comm.send ? "Send" : "Recv", data.iov_len, &comm));
  }

  if (!(mhandle.mem_type & (NCCL_PTR_HOST | NCCL_PTR_CUDA))) {
    return absl::InvalidArgumentError(
        "Isend/Irecv called with incorrect memory type.");
  }

  auto r = std::make_unique<ncclSocketRequest>();
  r->comm = &comm;
  // Creating the profiler data object for the request mapped to the profiler
  // assigned to the current flow group.
  uint32_t flow_for_request = comm.curr_flow_group_base;
  r->flow_idx = 0;
  if (comm.profilers[flow_for_request] != nullptr) {
    r->profiler_data = comm.profilers[flow_for_request]->CreateRequestData();
    r->flow_idx = flow_for_request;
  }

  r->size = data.iov_len;
  r->offset = reinterpret_cast<int64_t>(data.iov_base) -
              reinterpret_cast<int64_t>(mhandle.start_addr);
  r->completion_time = absl::InfinitePast();
  r->received_size = 0;

  if (comm.ready_for_ts_capture) {
    // Measure the time since the previous initial Isend/Irecv attempt,
    // with the creation of the comm counting as the first request.
    comm.stats.request.offload_interval_bucketer.Submit(
        absl::ToInt64Nanoseconds(comm.last_xfer.GetElapsed()));
    comm.last_xfer.Restart();
    comm.ready_for_ts_capture = false;
  }

  comm.ready_for_ts_capture = true;

  // Start request timer before submitting request to DXS.
  r->start_time = absl::ToUnixNanos(r->timer.Restart());
  r->slowness_timeout = absl::Milliseconds(kFastrakDataTransferSlownessMs);

  const uint64_t offset =
      (uint64_t)data.iov_base - (uint64_t)mhandle.start_addr;

  if (comm.send) {
    ASSIGN_OR_RETURN(r->op_impl,
                     static_cast<dxs::SendSocketInterface*>(
                         comm.dxs_socks[comm.curr_flow_group_base].get())
                         ->Send(offset, r->size, mhandle.reg_handle));
  } else {
    ASSIGN_OR_RETURN(r->op_impl,
                     static_cast<dxs::RecvSocketInterface*>(
                         comm.dxs_socks[comm.curr_flow_group_base].get())
                         ->RecvLinearized(offset, r->size, mhandle.reg_handle));
  }

  const auto offload_backlog = comm.stats.request.GetOffloadBacklog();
  if (offload_backlog > comm.stats.request.offload_backlog_peak) {
    comm.stats.request.offload_backlog_peak = offload_backlog;
  }

  LOG_IF(INFO, kEnableHotpathLogging) << absl::StrFormat(
      "%s request:%p op_id:%llu "
      "scheduled on flow group base %d on DXS "
      "group base socket:%p data:%p "
      "size:%d op_impl:%p from comm %p on gpu_pci:%s, backlog:%llu",
      comm.send ? "Send" : "Recv", r.get(), r->op_impl->GetOpId(),
      comm.curr_flow_group_base,
      comm.dxs_socks[comm.curr_flow_group_base].get(), data.iov_base, r->size,
      r->op_impl.get(), &comm, comm.gpu->pci_addr, offload_backlog);

  // Send and recv must have coherent current flow ID.
  // Round robin.
  comm.curr_flow_group_base++;
  if (comm.curr_flow_group_base == kFastrakNumFlows) {
    comm.curr_flow_group_base = 0;
  }

  comm.stats.request.offload_scheduled++;
  if (kFastrakEnableSendStats == 1 &&
      comm.stats.request.offload_scheduled % kFastrakSendStatsInterval == 0) {
    comm.stats.Dump();
  }
  // Capturing the time when request was successfully scheduled on DXS and
  // mapping it to the profiler assigned to the flow group.
  if (comm.profilers[flow_for_request] != nullptr) {
    comm.profilers[flow_for_request]->OnReqScheduled(r->profiler_data.get());
  }
  return r;
}

}  // namespace

absl::StatusOr<absl_nonnull std::unique_ptr<ncclSocketRequest>> NcclShim::Isend(
    Communication& comm, iovec data, int tag, MemoryHandle& mhandle) {
  if (!comm.send) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Isend called on recv comm %p", &comm));
  }
  ASSIGN_OR_RETURN(
      auto request, GetRequest(comm, data, mhandle),
      _ << absl::StrFormat("\nFailed to schedule Send request of size %d.\n%s",
                           data.iov_len, GetNcclCommInfo(comm)));
  return request;
}

absl::StatusOr<absl_nonnull std::unique_ptr<ncclSocketRequest>> NcclShim::Irecv(
    Communication& comm, iovec data, int tag, MemoryHandle& mhandle) {
  if (comm.send) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Irecv called on send comm %p", &comm));
  }
  ASSIGN_OR_RETURN(
      auto request, GetRequest(comm, data, mhandle),
      _ << absl::StrFormat("\nFailed to schedule Recv request of size %d.\n%s",
                           data.iov_len, GetNcclCommInfo(comm)));
  return request;
}

namespace {

void PrintTest(ncclSocketRequest& r, Communication& comm) {
  absl::Status timeout_status = r.timer.CheckTimeout(r.slowness_timeout);
  if (!timeout_status.ok()) {
    LOG(WARNING) << absl::StrFormat(
        "%s Test pending request:%p, size:%d, "
        "comm:%p current_pending:%llu, "
        "total_scheduled:%llu, request age: %ld ns\n%s",
        comm.send ? "Send" : "Recv", &r, r.size, &comm,
        comm.stats.request.GetOffloadBacklog(),
        comm.stats.request.offload_scheduled,
        absl::ToInt64Nanoseconds(r.timer.GetElapsed()), GetNcclCommInfo(comm));
    // Logging backoff.
    r.slowness_timeout = r.slowness_timeout * 2;
  }
}

absl::StatusOr<bool> pollFromDxs(ncclSocketRequest& r,
                                 dxs::SendOpInterface& op) {
  std::optional<absl::Status> test_status = op.Test();
  if (!test_status.has_value()) return false;
  RETURN_IF_ERROR(*test_status);
  // Request completion time is based off of the sub-request
  // that completed last.
  if (op.GetCompletionTime() > r.completion_time) {
    r.completion_time = op.GetCompletionTime();
  }
  return true;
}

absl::StatusOr<bool> pollFromDxs(ncclSocketRequest& r,
                                 dxs::LinearizedRecvOpInterface& op) {
  std::optional<absl::StatusOr<uint64_t>> result = op.Test();
  if (!result.has_value()) return false;
  ASSIGN_OR_RETURN(uint64_t size, *result);
  // Request completion time is based off of the sub-request
  // that completed last.
  if (op.GetCompletionTime() > r.completion_time) {
    r.completion_time = op.GetCompletionTime();
  }
  r.received_size = size;
  return true;
}

absl::StatusOr<bool> pollFromDxs(ncclSocketRequest& r) {
  dxs::OpInterface& op = *ABSL_DIE_IF_NULL(r.op_impl);
  return r.comm->send
             ? pollFromDxs(r, static_cast<dxs::SendOpInterface&>(op))
             : pollFromDxs(r, static_cast<dxs::LinearizedRecvOpInterface&>(op));
}

}  // namespace

absl::StatusOr<std::optional<int>> NcclShim::Test(ncclSocketRequest& request) {
  Communication* comm = request.comm;
  ASSIGN_OR_RETURN(bool done, pollFromDxs(request),
                   _ << absl::StrFormat(
                       "\nPolling DXS op returned an error for %s request:%p "
                       "size:%d.\n%s",
                       comm->send ? "Send" : "Recv", &request, request.size,
                       GetNcclCommInfo(*comm)));
  uint32_t flow_idx = request.flow_idx;
  if (!done) {
    if (comm->profilers[flow_idx] != nullptr) {
      comm->profilers[flow_idx]->TestRequest(request.profiler_data.get(), false,
                                             0);
    }
    PrintTest(request, *comm);

    RETURN_IF_ERROR(request.timer.CheckTimeout(data_transfer_timeout_))
        << absl::StrFormat("%s Test pending request:%p size:%d\n%s",
                           comm->send ? "Send" : "Recv", &request, request.size,
                           GetNcclCommInfo(*comm));
    return std::nullopt;
  }
  if (!comm->send) {
    // Replace request size with the actual received size for recv.
    request.size = request.received_size;
  }
  // This is a workaround for a NCCL bug
  //
  // NCCL can sometimes call Test() on a request that has already returned
  // an error. Note that in not deleting the request in error cases, we are
  // effectively leaking it, but there is no alternative since NCCL does not
  // provide another hook into the request lifecycle other than Test().
  auto request_deleter = absl::WrapUnique(&request);
  if (comm->profilers[flow_idx] != nullptr) {
    comm->profilers[flow_idx]->TestRequest(request.profiler_data.get(), true,
                                           request.size);
  }
  comm->stats.request.offload_completed++;

  auto curr_time = request.timer.GetCurrTime();
  auto duration = curr_time - request.timer.GetStartTime();

  // Only submit non-empty requests as samples
  if (request.size != 0) {
    comm->stats.request.offload_duration_bucketer.Submit(
        absl::ToInt64Nanoseconds(duration));
    comm->stats.request.offload_complete_age_bucketer.Submit(
        absl::ToInt64Nanoseconds(curr_time - request.completion_time));
    comm->stats.request.offload_size_bucketer.Submit(request.size);
  }
  LOG_IF(INFO, kEnableHotpathLogging) << absl::StrFormat(
      "all %d bytes transferred after %ld nanoseconds (rate: "
      "%f MB/s) for %s request:%p comm:%p on gpu_pci:%s "
      "completed_req_cnt:%d",
      request.size, absl::ToInt64Nanoseconds(duration),
      duration > absl::ZeroDuration()
          ? static_cast<double>(request.size) /
                (absl::ToDoubleMicroseconds(duration))
          : 0.0,
      comm->send ? "Send" : "Recv", &request, comm, comm->gpu->pci_addr,
      comm->stats.request.offload_completed);
  return request.size;
}

absl::StatusOr<MemoryHandle> NcclShim::RegMrDmaBufInternal(
    Communication& comm, iovec data, int offset, std::optional<int> fd) {
  MemoryHandle mhandle;
  mhandle.start_addr = data.iov_base;
  mhandle.size = data.iov_len;
  int dmabuf_fd = -1;
  bool fd_created = false;
  if (fd.has_value()) {
    dmabuf_fd = fd.value();
  } else {
    ASSIGN_OR_RETURN(dmabuf_fd,
                     getDmabufFd(data.iov_base, data.iov_len, gpu_.pci_addr));
    fd_created = true;
  }
  absl::Cleanup fd_cleanup = [&] {
    if (fd_created) NetSysCall().Close(dmabuf_fd);
  };
  if (comm.profilers[kRepresentativeFlowIdx] != nullptr) {
    comm.profilers[kRepresentativeFlowIdx]->OnBufferPreRegRequest(mhandle.size);
  }

  ASSIGN_OR_RETURN(mhandle.reg_handle,
                   comm.buffer_manager->RegBuf(dmabuf_fd, data.iov_len));
  LOG(INFO) << absl::StrFormat(
      "RegMr[netdev=%d] CUDA memory for comm %p with "
      "handle:%lld for data:%p size:%d",
      comm.selected_dev, &comm, mhandle.reg_handle, mhandle.start_addr,
      mhandle.size);
  if (comm.profilers[kRepresentativeFlowIdx] != nullptr) {
    comm.profilers[kRepresentativeFlowIdx]->OnBufferPostRegRequest(
        mhandle.size);
  }
  mhandle.dmabuf_fd = dmabuf_fd;
  if (fd_created) {
    std::move(fd_cleanup).Cancel();
  }
  return mhandle;
}

absl::StatusOr<std::unique_ptr<MemoryHandle>> NcclShim::RegMrDmaBuf(
    Communication& comm, iovec data, int type, uint64_t offset,
    std::optional<int> fd) {
  auto mhandle = std::make_unique<MemoryHandle>();
  // Registering host memory - do nothing and simply return a dummy mhandle
  if (!(type & NCCL_PTR_CUDA)) {
    mhandle->mem_type = type;
    mhandle->start_addr = data.iov_base;
    mhandle->size = data.iov_len;
    LOG(INFO) << absl::StrFormat(
        "Registering host memory for %p data:%p size:%d", &comm,
        mhandle->start_addr, mhandle->size);
    return mhandle;
  }
  // Forked from NCCL's IB implmentation for caching registered memory regions
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct ncclMrCache* cache = &kNcclSocketDevs[comm.selected_dev].mr_cache;
  uintptr_t addr = reinterpret_cast<uintptr_t>(data.iov_base) & -pageSize;
  // Ceiling of number of pages needed for the given data.
  size_t pages = (reinterpret_cast<uintptr_t>(data.iov_base) + data.iov_len -
                  addr + pageSize - 1) /
                 pageSize;
  // Directly register memory if caching is disabled.
  if (!kFastrakCacheMr) {
    data.iov_base = reinterpret_cast<void*>(addr);
    data.iov_len = pageSize * pages;
    ASSIGN_OR_RETURN(*mhandle, RegMrDmaBufInternal(comm, data, offset, fd));
    mhandle->mem_type = type;
    return mhandle;
  }
  pthread_mutex_lock(&kNcclSocketDevs[comm.selected_dev].reg_mutex);
  absl::Cleanup unlock_reg_mutex = [&] {
    pthread_mutex_unlock(&kNcclSocketDevs[comm.selected_dev].reg_mutex);
  };
  for (int slot = 0;; slot++) {
    uintptr_t slot_addr = 0;
    if (cache->slots != nullptr && cache->population != 0) {
      slot_addr =
          reinterpret_cast<uintptr_t>(cache->slots[slot].handle.start_addr);
    }
    // Not found in cache
    if (slot == cache->population || addr < slot_addr) {
      // Need to expand cache
      if (cache->population == cache->capacity) {
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        cache->slots = static_cast<ncclMr*>(
            realloc(cache->slots, sizeof(ncclMr) * cache->capacity));
        if (cache->slots == nullptr) {
          return absl::UnavailableError(
              absl::StrFormat("Failed to grow mr cache: %s", strerror(errno)));
        }
      }
      data.iov_base = reinterpret_cast<void*>(addr);
      data.iov_len = pageSize * pages;
      ASSIGN_OR_RETURN(*mhandle, RegMrDmaBufInternal(comm, data, offset, fd));
      mhandle->mem_type = type;
      if (slot != cache->population) {
        memmove(cache->slots + slot + 1, cache->slots + slot,
                (cache->population - slot) * sizeof(ncclMr));
      }
      cache->slots[slot].handle = *mhandle;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->population++;
      break;
    } else if (addr >= slot_addr &&
               static_cast<size_t>((addr - slot_addr) / pageSize + pages) <=
                   static_cast<size_t>(cache->slots[slot].pages)) {
      // entry found, increment refs
      ++cache->slots[slot].refs;
      LOG(INFO) << absl::StrFormat(
          "RegMr[netdev=%d] CUDA memory for comm %p with existing reg found, "
          "handle:%lld, "
          "data: %p size: %d ",
          comm.selected_dev, &comm, cache->slots[slot].handle.reg_handle,
          cache->slots[slot].handle.start_addr, cache->slots[slot].handle.size);
      *mhandle = cache->slots[slot].handle;
      break;
    }
  }
  return mhandle;
}

absl::Status NcclShim::DeregMrInternal(Communication& comm,
                                       std::unique_ptr<MemoryHandle> mhandle) {
  if (mhandle->dmabuf_fd > 0) {
    NetSysCall().Close(mhandle->dmabuf_fd);
  }
  if (comm.profilers[kRepresentativeFlowIdx] != nullptr) {
    comm.profilers[kRepresentativeFlowIdx]->OnBufferPreDeregRequest(
        mhandle->size);
  }
  RETURN_IF_ERROR(comm.buffer_manager->DeregBuf(mhandle->reg_handle));
  if (comm.profilers[kRepresentativeFlowIdx] != nullptr) {
    comm.profilers[kRepresentativeFlowIdx]->OnBufferPostDeregRequest(
        mhandle->size);
  }
  return absl::OkStatus();
}

absl::Status NcclShim::DeregMr(Communication& comm,
                               std::unique_ptr<MemoryHandle> mhandle) {
  if (!(mhandle->mem_type & NCCL_PTR_CUDA)) {
    LOG(INFO) << absl::StrFormat("DeregMr host memory for %p data:%p size:%d.",
                                 &comm, mhandle->start_addr, mhandle->size);
    return absl::OkStatus();
  }
  if (mhandle->reg_handle != dxs::kInvalidRegistration) {
    LOG(INFO) << absl::StrFormat(
        "DeregMr[netdev=%d] CUDA memory for comm %p with handle:%lld "
        "data:%p size:%d.",
        comm.selected_dev, &comm, mhandle->reg_handle, mhandle->start_addr,
        mhandle->size);
    // Directly deregister the memory region if caching is disabled.
    if (!kFastrakCacheMr) {
      return DeregMrInternal(comm, std::move(mhandle));
    }
    ncclMrCache* cache = &kNcclSocketDevs[comm.selected_dev].mr_cache;
    pthread_mutex_lock(&kNcclSocketDevs[comm.selected_dev].reg_mutex);
    absl::Cleanup unlock_reg_mutex = [&] {
      pthread_mutex_unlock(&kNcclSocketDevs[comm.selected_dev].reg_mutex);
    };
    for (int i = 0; i < cache->population; i++) {
      if (mhandle->reg_handle == cache->slots[i].handle.reg_handle) {
        if (--cache->slots[i].refs == 0) {
          // Move everything by 1
          memmove(cache->slots + i, cache->slots + i + 1,
                  (cache->population - i - 1) * sizeof(ncclMr));
          if (--cache->population == 0) {
            free(cache->slots);
            cache->slots = nullptr;
            cache->capacity = 0;
          }
          RETURN_IF_ERROR(DeregMrInternal(comm, std::move(mhandle)));
        }
        return absl::OkStatus();
      }
    }
    return absl::InternalError(
        absl::StrFormat("Could not find mhandle with registration handle %lld",
                        mhandle->reg_handle));
  } else {
    LOG(INFO) << absl::StrFormat(
        "DeregMr CUDA memory for comm %p with invalid handle "
        "data:%p size:%d",
        &comm, mhandle->start_addr, mhandle->size);
  }
  return absl::OkStatus();
}

absl::StatusOr<ncclNetProperties_v7_t> NcclShim::GetProperties(int dev) {
  if (dev < 0 || dev >= kNcclNetIfs) {
    return absl::InvalidArgumentError("Invalid device number.");
  }
  ncclNetProperties_v7_t props;
  props.name = kNcclSocketDevs[dev].dev_name;
  props.pciPath = kNcclSocketDevs[dev].pci_path;
  props.guid = dev;
#ifdef HOST_PTR_ONLY
  props.ptrSupport = NCCL_PTR_HOST;
#else
  props.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
#endif
  props.speed = GetSpeed(props.name);
  props.port = 0;
  props.maxComms = 65536;
  props.latency = 0;
  props.maxRecvs = 1;
  props.netDeviceType = NCCL_NET_DEVICE_HOST;
  props.netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;
  LOG(INFO) << absl::StrFormat("props: netDeviceType %d, netDeviceVersion %d",
                               props.netDeviceType, props.netDeviceVersion);
  return props;
}

void NcclShim::Close(std::unique_ptr<Communication> comm) {
  LOG_IF(WARNING, kFastrakLogConnectionInfo && comm->send) << absl::StrFormat(
      "Closing Send comm %p (Local IP: %s, Remote IP: %s)", comm.get(),
      kNcclSocketDevs[comm->selected_dev].ip_addr, comm->dxs_addr);

  LOG_IF(INFO, !kFastrakLogConnectionInfo || !comm->send) << absl::StrFormat(
      "Closing %s comm %p", comm->send ? "Send" : "Recv", comm.get());
  if (kFastrakDumpCommStats) {
    comm->stats.Dump();
  }
  for (auto i = 0; i < kFastrakNumFlows; ++i) {
    if (comm->profilers[i] != nullptr) {
      comm->profilers[i]->OnConnectionClosed();
    }
  }
}

absl::Status NcclShim::CloseListen(
    std::unique_ptr<ListenCommunication> listenComm) {
  if (listenComm->nccl_listen != -1) {
    RETURN_IF_ERROR(SysCallResultToStatus(
        NetSysCall().Close(listenComm->nccl_listen), "close"));
  }
  return absl::OkStatus();
}

}  // namespace fastrak

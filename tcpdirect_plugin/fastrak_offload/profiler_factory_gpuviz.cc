/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/profiler_factory_gpuviz.h"

#include <sys/socket.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "GPUViz/src/nccl_stats.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "nccl.h"
#include "nccl_common.h"
#include "tcpdirect_plugin/fastrak_offload/macro.h"
#include "tcpdirect_plugin/fastrak_offload/params.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_factory.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_factory_gpuviz.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_gpuviz.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_noop.h"

namespace fastrak {

absl::Status GPUVizProfilerFactory::Init() {
  nccl_telemetry_plugin_lib_handle_ = dlopen("libGPUViz.so", RTLD_NOW);
  if (!nccl_telemetry_plugin_lib_handle_) {
    LOG(ERROR) << "Failed to load library" << dlerror();
    return absl::InternalError("Failed to load library");
  } else {
    nccl_stats_plugin_ = (ncclStatsPlugin_t*)dlsym(
        nccl_telemetry_plugin_lib_handle_, "nccl_telemetry_stats_plugin_v1");
    if (!nccl_stats_plugin_) {
      LOG(ERROR) << "Failed to load symbol" << dlerror();
      return absl::InternalError("Failed to load symbol");
    }
  }

  ncclDebugLogger_t logFunction = GetNcclLogFunc();
  LOG(INFO) << "Initializing nccl stats plugin for GPU: " << gpu_dev_.pci_addr
            << ", netdev: " << netdev_;
  if (nccl_stats_plugin_->init(
          logFunction,
          (RecvLatencySW | SendLatencySW | SendMessageSize | RecvMessageSize),
          &stats_global_handle_) != ncclSuccess) {
    nccl_stats_plugin_ = nullptr;
    stats_global_handle_ = 0;
    LOG(ERROR) << "Failed to initialize nccl stats plugin";
    return absl::InternalError("Failed to initialize nccl stats plugin");
  }
  return absl::OkStatus();
}

std::unique_ptr<ProfilerInterface> GPUVizProfilerFactory::Create(
    Communication* comm, ProfilerOptions options) {
  struct sockaddr_storage local_addr;
  struct sockaddr_storage remote_addr;
  local_addr = GetSockAddress(options.local_ip_addr, options.local_port);
  remote_addr = GetSockAddress(options.remote_ip_addr, options.remote_port);
  auto gpuviz_profiler = std::make_unique<GPUVizProfiler>(
      nccl_stats_plugin_, comm->send, options.flow_id, gpu_dev_.pci_addr,
      local_addr, remote_addr);
  if (!gpuviz_profiler->Init(stats_global_handle_).ok()) {
    LOG_FIRST_N(WARNING, 10)
        << "Failed to initialize GPUViz profiler for GPU: " << gpu_dev_.pci_addr
        << ", connection: " << comm->idx;
  }
  return gpuviz_profiler;
}
GPUVizProfilerFactory::~GPUVizProfilerFactory() {
  if (nccl_stats_plugin_ != nullptr && stats_global_handle_ != 0) {
    nccl_stats_plugin_->destroy(stats_global_handle_);
    nccl_stats_plugin_ = nullptr;
    stats_global_handle_ = 0;
  }
  if (nccl_telemetry_plugin_lib_handle_ != nullptr) {
    dlclose((void*)nccl_telemetry_plugin_lib_handle_);
    nccl_telemetry_plugin_lib_handle_ = nullptr;
  }
}
struct sockaddr_storage GPUVizProfilerFactory::GetSockAddress(
    const std::string& ipAddress, uint16_t port) {
  sockaddr_storage socketAddress;
  memset(&socketAddress, 0, sizeof(socketAddress));

  // Determine address family based on the IP address format
  int addressFamily = absl::StrContains(ipAddress, ":") ? AF_INET6 : AF_INET;

  // Convert IP address string to binary and store in sockaddr_storage
  if (addressFamily == AF_INET) {
    sockaddr_in* ipv4Address = reinterpret_cast<sockaddr_in*>(&socketAddress);
    ipv4Address->sin_family = AF_INET;
    inet_pton(AF_INET, ipAddress.c_str(), &(ipv4Address->sin_addr));
    ipv4Address->sin_port = htons(port);
  } else {  // AF_INET6
    sockaddr_in6* ipv6Address = reinterpret_cast<sockaddr_in6*>(&socketAddress);
    ipv6Address->sin6_family = AF_INET6;
    inet_pton(AF_INET6, ipAddress.c_str(), &(ipv6Address->sin6_addr));
    ipv6Address->sin6_port = htons(port);
  }
  return socketAddress;
}

std::unique_ptr<ProfilerFactoryInterface> GetProfilerFactory(
    const ProfilerFactoryOptions& options) {
  TelemetryMode telemetry_mode = TelemetryMode(kFastrakPluginTelemetryMode);
  if (telemetry_mode == TelemetryMode::kGPUVizWriteOnLocalDiskOnly ||
      telemetry_mode == TelemetryMode::kGPUVizWriteOnLocalDiskAndUpload ||
      telemetry_mode == TelemetryMode::kGPUVizUploadOnly) {
    auto gpuviz_profiler_factory = std::make_unique<GPUVizProfilerFactory>(
        options.gpu_dev, options.netdev);
    if (!gpuviz_profiler_factory->Init().ok()) {
      LOG(ERROR) << "Failed to initialize GPUViz profiler factory";
      return std::make_unique<NoOpProfilerFactory>();
    }
    return gpuviz_profiler_factory;
  }
  LOG(INFO) << "GPUViz is disabled";
  return std::make_unique<NoOpProfilerFactory>();
}
}  // namespace fastrak
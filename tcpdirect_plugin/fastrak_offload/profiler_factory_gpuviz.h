/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_GPUVIZ_FACTORY_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_GPUVIZ_FACTORY_H_

#include <dlfcn.h>

#include <cstdint>
#include <memory>

#include "GPUViz/src/nccl_stats.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "nccl_cuda/cuda_defs.h"
#include "tcpdirect_plugin/fastrak_offload/common.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_factory.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"

namespace fastrak {

class GPUVizProfilerFactory : public ProfilerFactoryInterface {
 public:
  GPUVizProfilerFactory() = delete;
  GPUVizProfilerFactory(gpuDev gpu_dev, int netdev)
      : gpu_dev_(gpu_dev), netdev_(netdev) {}

  absl::Status Init();
  std::unique_ptr<ProfilerInterface> Create(Communication* comm,
                                            ProfilerOptions options) override;
  ~GPUVizProfilerFactory() override;

 private:
  struct sockaddr_storage GetSockAddress(const std::string& ipAddress,
                                         uint16_t port);
  const ncclStatsPlugin_t* nccl_stats_plugin_ = nullptr;
  const gpuDev gpu_dev_;
  const int netdev_;
  void* nccl_telemetry_plugin_lib_handle_ = nullptr;
  uintptr_t stats_global_handle_;
};
}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_GPUVIZ_FACTORY_H_

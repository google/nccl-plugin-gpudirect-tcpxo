/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_FACTORY_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_FACTORY_H_

#include <cstdint>
#include <memory>
#include <string>

#include "nccl_cuda/cuda_defs.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"

namespace fastrak {

struct Communication;

struct ProfilerOptions {
  std::string local_ip_addr;
  std::string remote_ip_addr;
  uint16_t local_port;
  uint16_t remote_port;
  uint32_t flow_id;
};

class ProfilerFactoryInterface {
 public:
  ProfilerFactoryInterface() = default;
  virtual ~ProfilerFactoryInterface() = default;
  ProfilerFactoryInterface(const ProfilerFactoryInterface&) = delete;
  ProfilerFactoryInterface& operator=(const ProfilerFactoryInterface&) = delete;

  virtual std::unique_ptr<ProfilerInterface> Create(
      Communication* comm, ProfilerOptions options) = 0;
};

struct ProfilerFactoryOptions {
  gpuDev gpu_dev;

  // Closest netdev to the gpu.
  int netdev;
};

std::unique_ptr<ProfilerFactoryInterface> GetProfilerFactory(
    const ProfilerFactoryOptions& options);

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_FACTORY_H_

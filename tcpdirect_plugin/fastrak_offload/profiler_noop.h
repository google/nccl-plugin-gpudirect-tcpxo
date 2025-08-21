/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_NOOP_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_NOOP_H_

#include <cstddef>
#include <memory>

#include "absl/base/nullability.h"
#include "tcpdirect_plugin/fastrak_offload/common.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_factory.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"

namespace fastrak {

class NoOpProfiler : public ProfilerInterface {
  std::unique_ptr<ProfilerRequestDataInterface> CreateRequestData() override {
    return nullptr;
  }
  void OnReqScheduled(
      ProfilerRequestDataInterface* absl_nullable data) override {}
  void TestRequest(ProfilerRequestDataInterface* absl_nullable data, bool done,
                   size_t size) override {}
  void OnBufferPreRegRequest(size_t size) override {}
  void OnBufferPostRegRequest(size_t size) override {}
  void OnBufferPreDeregRequest(size_t size) override {}
  void OnBufferPostDeregRequest(size_t size) override {}
  void OnConnectionClosed() override {}
};

class NoOpProfilerFactory : public ProfilerFactoryInterface {
 public:
  std::unique_ptr<ProfilerInterface> Create(Communication* comm,
                                            ProfilerOptions options) override {
    return std::make_unique<NoOpProfiler>();
  }
};

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_NOOP_H_

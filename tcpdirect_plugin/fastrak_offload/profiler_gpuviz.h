/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_GPUVIZ_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_GPUVIZ_H_

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <memory>
#include <string>

#include "GPUViz/src/nccl_stats.h"
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "dxs/client/oss/status_macros.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"
namespace fastrak {

constexpr absl::string_view kNcclPluginName = "FasTrak";
class GPUVizProfilerRequestData : public ProfilerRequestDataInterface {
 public:
  GPUVizProfilerRequestData(uint32_t request_idx, uint32_t profiler_idx)
      : request_idx_(request_idx),
        profiler_idx_(profiler_idx),
        request_size_(0),
        timer_started_(false) {}

  uint32_t request_idx() const { return request_idx_; }
  void set_request_size(size_t size) { request_size_ = size; }
  uint32_t profiler_idx() const { return profiler_idx_; }
  size_t request_size() { return request_size_; }
  absl::StatusOr<absl::Duration> request_time_elapsed() {
    if (!timer_started_) {
      return absl::InternalError("Timer not started");
    }
    ASSIGN_OR_RETURN(absl::Time end_time, get_current_time());
    return end_time - start_time_;
  }

  absl::StatusOr<absl::Time> request_start_time() {
    if (!timer_started_) {
      return absl::InternalError("Timer not started");
    }
    return start_time_;
  }

  absl::Status start_request_timer() {
    timer_started_ = false;
    ASSIGN_OR_RETURN(start_time_, get_current_time());
    timer_started_ = true;
    return absl::OkStatus();
  }

 private:
  absl::StatusOr<absl::Time> get_current_time() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) < 0) {
      return absl::InternalError("Failed to get current time");
    }
    return absl::TimeFromTimespec(ts);
  }
  const uint32_t request_idx_;
  const uint32_t profiler_idx_;
  size_t request_size_;
  absl::Time start_time_;
  bool timer_started_;
};

class GPUVizProfiler : public ProfilerInterface {
 public:
  GPUVizProfiler() = delete;
  GPUVizProfiler(const ncclStatsPlugin_t* nccl_stats_plugin, const bool send,
                 const uint32_t flow_id, const std::string& gpu_pci_addr,
                 const struct sockaddr_storage local_addr,
                 const struct sockaddr_storage remote_addr)
      : nccl_stats_plugin_(nccl_stats_plugin),
        is_send_(send),
        profiler_idx_(flow_id),
        gpu_pci_addr_(gpu_pci_addr),
        local_addr_(local_addr),
        remote_addr_(remote_addr) {}
  absl::Status Init(uintptr_t stats_global_handle);
  ~GPUVizProfiler() override;
  std::unique_ptr<ProfilerRequestDataInterface> CreateRequestData() override;

  void OnReqScheduled(
      ProfilerRequestDataInterface* absl_nullable data) override;
  void TestRequest(ProfilerRequestDataInterface* absl_nullable data, bool done,
                   size_t size) override;
  void OnConnectionClosed() override;
  void OnBufferPreRegRequest(size_t size) override {};
  void OnBufferPostRegRequest(size_t size) override {};
  void OnBufferPreDeregRequest(size_t size) override {};
  void OnBufferPostDeregRequest(size_t size) override {};

 private:
  absl::Status ValidateRequestData(GPUVizProfilerRequestData* request_data);
  const ncclStatsPlugin_t* nccl_stats_plugin_;
  uintptr_t stats_connection_handle_;
  std::unique_ptr<ncclStatsConnectionIdentifier> connection_identifier_;
  const bool is_send_;
  const uint32_t profiler_idx_;
  const std::string gpu_pci_addr_;
  const std::string nccl_plugin_name_{kNcclPluginName};
  const struct sockaddr_storage local_addr_;
  const struct sockaddr_storage remote_addr_;
  uint32_t request_counter_ = 0;
};
}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_GPUVIZ_H_

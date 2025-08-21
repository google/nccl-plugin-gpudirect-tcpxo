/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/profiler_gpuviz.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>

#include "GPUViz/src/nccl_stats.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "nccl.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"

#define GPUVIZ_LOG_HEADER "GPUVizProfiler-" << profiler_idx_ << ": "

namespace fastrak {

absl::Status GPUVizProfiler::Init(uintptr_t stats_global_handle) {
  connection_identifier_ = absl::WrapUnique(new ncclStatsConnectionIdentifier{
      .conn_type = EntityTCPConnection,
      .nccl_plugin_type = NetPlugin,
      .nccl_plugin_name = nccl_plugin_name_.c_str(),
      .gpu_pci_addr = gpu_pci_addr_.c_str()});

  // Connection creation: Use Comm idx as connection id by initializing the
  // local port with the comm idx.
  ncclStatsTCPConnection connection;
  memset(&connection, 0, sizeof(connection));
  connection.local_endpoint = local_addr_;
  connection.remote_endpoint = remote_addr_;
  connection_identifier_->connection.tcp_conn = connection;

  if (nccl_stats_plugin_->addConnection(
          stats_global_handle,
          (const ncclStatsConnectionIdentifier*)connection_identifier_.get(),
          &stats_connection_handle_) != ncclSuccess) {
    LOG(ERROR) << "Failed to add connection for GPU: "
               << connection_identifier_->gpu_pci_addr;
    stats_connection_handle_ = 0;
    return absl::Status(absl::StatusCode::kInternal,
                        "Failed to add connection for GPU");
  }
  DCHECK_NE(stats_connection_handle_, 0);
  return absl::OkStatus();
}

std::unique_ptr<ProfilerRequestDataInterface>
GPUVizProfiler::CreateRequestData() {
  return std::make_unique<GPUVizProfilerRequestData>(request_counter_++,
                                                     profiler_idx_);
}

void GPUVizProfiler::OnReqScheduled(
    ProfilerRequestDataInterface* absl_nullable data) {
  auto request_data = static_cast<GPUVizProfilerRequestData*>(data);
  DCHECK(request_data != nullptr);
  if (!ValidateRequestData(request_data).ok()) {
    return;
  }
  if (!request_data->start_request_timer().ok()) {
    LOG_FIRST_N(WARNING, 10)
        << GPUVIZ_LOG_HEADER << "Failed to start request timer";
    return;
  };
}

void GPUVizProfiler::TestRequest(
    ProfilerRequestDataInterface* absl_nullable data, bool done, size_t size) {
  if (!done) return;
  if (stats_connection_handle_ == 0) {
    LOG_FIRST_N(WARNING, 1) << GPUVIZ_LOG_HEADER
                            << "stats_connection_handle_ is not "
                               "initialized for this connection.";
    return;
  }
  auto request_data = static_cast<GPUVizProfilerRequestData*>(data);
  DCHECK(request_data != nullptr);
  if (!ValidateRequestData(request_data).ok()) {
    return;
  }
  request_data->set_request_size(size);
  auto request_time_elapsed = request_data->request_time_elapsed();
  if (!request_time_elapsed.ok()) {
    LOG_FIRST_N(WARNING, 10)
        << GPUVIZ_LOG_HEADER << "Failed to get request time elapsed: "
        << request_time_elapsed.status();
    return;
  }
  ncclStatsOperationMetric operation_metric;
  ncclStatsLatencyMeasurement latency_measurement = {
      .latency_type = LatencySoftware,
      .latency_in_nanoseconds = static_cast<uint64_t>(
          absl::ToInt64Nanoseconds(request_time_elapsed.value()))};
  operation_metric.measurements = &latency_measurement;
  operation_metric.num_measurements = 1;
  operation_metric.op_sz = request_data->request_size();
  uint64_t start_time_ns = 0;
  auto request_start_time = request_data->request_start_time();
  if (!request_start_time.ok()) {
    LOG_FIRST_N(WARNING, 10)
        << GPUVIZ_LOG_HEADER
        << "Failed to get request start time: " << request_start_time.status();
  } else {
    timespec ts = absl::ToTimespec(request_start_time.value());
    start_time_ns = static_cast<uint64_t>(ts.tv_sec) * 1e9 + ts.tv_nsec;
  }
  operation_metric.op_start_time = start_time_ns;
  if (is_send_) {
    operation_metric.type = OperationTypeChunkSend;
  } else {
    operation_metric.type = OperationTypeChunkRecv;
  }

  if (nccl_stats_plugin_->notifyOperationMeasurement(
          stats_connection_handle_, &operation_metric) != ncclSuccess) {
    LOG_FIRST_N(WARNING, 1)
        << GPUVIZ_LOG_HEADER << "Failed to notify operation measurement";
  }
}

void GPUVizProfiler::OnConnectionClosed() {
  if (stats_connection_handle_ != 0) {
    if (nccl_stats_plugin_->deleteConnection(
            stats_connection_handle_, ConnectionCloseLocalTerminate,
            "Connection closed") == ncclSuccess) {
      stats_connection_handle_ = 0;
    }
  }
}

GPUVizProfiler::~GPUVizProfiler() {
  if (nccl_stats_plugin_ != nullptr && stats_connection_handle_ != 0) {
    nccl_stats_plugin_->deleteConnection(stats_connection_handle_,
                                         ConnectionCloseLocalTerminate,
                                         "Connection closed");
    stats_connection_handle_ = 0;
  }
}
absl::Status GPUVizProfiler::ValidateRequestData(
    GPUVizProfilerRequestData* request_data) {
  if (request_data == nullptr) {
    LOG_FIRST_N(WARNING, 1) << GPUVIZ_LOG_HEADER
                            << ": ValidateRequestData called with nullptr "
                               "request data";
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        "GPUVizProfiler: ValidateRequestData called with nullptr");
  }
  CHECK(request_data->profiler_idx() == profiler_idx_)
      << "GPUVizProfiler-" << profiler_idx_
      << ": ValidateRequestData called for request "
         "with profiler_idx: "
      << request_data->profiler_idx();
  return absl::OkStatus();
}

}  // namespace fastrak

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_REQUEST_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_REQUEST_H_

#include <cstdint>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/time/time.h"
#include "dxs/client/dxs-client-interface.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_interface.h"
#include "tcpdirect_plugin/fastrak_offload/stats.h"

namespace fastrak {

enum class RequestIdentifier : uint32_t {};

struct ncclSocketRequest {
  struct Communication* comm;
  // error is set to true if the request is in error state, if so subsequent
  // calls to Test() will directly return an error.
  bool error = false;
  uint64_t size;
  uint32_t flow_idx;
  // Offset into the send/recv buffer. Used for tracking buffer "slot" usage.
  int64_t offset;
  // Used for DXS send/recv ops, nullptr for msgs transferred directly by NCCL
  std::unique_ptr<dxs::OpInterface> op_impl;
  // Used for DXS linearized recv ops to track the received size for msgs
  uint64_t received_size;
  // Start time of a request when we first scheduled it.
  // For recv requests, this is the time after we receive the metadata.
  uint64_t start_time;
  uint64_t xfer_time;
  fastrak::Timer timer;
  absl::Duration slowness_timeout;
  absl::Time completion_time;  // From DXS client

  absl_nullable std::unique_ptr<ProfilerRequestDataInterface> profiler_data;
};

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_REQUEST_H_

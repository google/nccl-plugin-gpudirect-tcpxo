/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_INTERFACE_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_INTERFACE_H_

#include <cstddef>
#include <memory>

#include "absl/base/nullability.h"
#include "dxs/client/base-interface.h"

namespace fastrak {

// Per-request data interface.
class ProfilerRequestDataInterface : public dxs::BaseInterface {};

// Per-comm data.
class ProfilerInterface : public dxs::BaseInterface {
 public:
  // Create the profiler's per-request data.
  virtual std::unique_ptr<ProfilerRequestDataInterface> CreateRequestData() = 0;

  // A request has been submitted to the comm's queue.
  // Note that the ACTUAL request size is unknown at this point because
  // NCCL might post a larger recv buffer than the actual # of bytes
  // transferred.
  virtual void OnReqScheduled(
      ProfilerRequestDataInterface* absl_nullable data) = 0;

  // Test has been called on a request.
  // At this point the actual size of the request is known, and will be passed
  // in when the request is completed (`done` is true).
  // When `done` is false, the request is still in flight and the size passed in
  // is undefined and should be disregarded.
  virtual void TestRequest(ProfilerRequestDataInterface* absl_nullable data,
                           bool done, size_t size) = 0;

  // Buffer registration (RxDM RPC) pre/post hooks.
  virtual void OnBufferPreRegRequest(size_t size) = 0;
  virtual void OnBufferPostRegRequest(size_t size) = 0;
  virtual void OnBufferPreDeregRequest(size_t size) = 0;
  virtual void OnBufferPostDeregRequest(size_t size) = 0;

  // Connection/communication has been closed. Last chance to emit logs.
  virtual void OnConnectionClosed() = 0;
};

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PROFILER_INTERFACE_H_

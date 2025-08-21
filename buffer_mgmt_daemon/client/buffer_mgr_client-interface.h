/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_CLIENT_BUFFER_MGR_CLIENT_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_CLIENT_BUFFER_MGR_CLIENT_INTERFACE_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "dxs/client/dxs-client-types.h"

namespace tcpdirect {

// Client for managing buffers. Thread safe.
class BufferManagerClientInterface {
 public:
  virtual ~BufferManagerClientInterface() = default;

  virtual absl::StatusOr<dxs::Reg> RegBuf(int fd, size_t size) = 0;
  virtual absl::Status DeregBuf(uint64_t handle) = 0;
};
}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_CLIENT_BUFFER_MGR_CLIENT_INTERFACE_H_

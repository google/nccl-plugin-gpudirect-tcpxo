/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_BUFFER_MANAGER_SERVICE_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_BUFFER_MANAGER_SERVICE_INTERFACE_H_

#include "absl/status/status.h"
namespace tcpdirect {

/**
 * Defines a service component to serve client requests via Unix domain sockets.
 */
class BufferManagerServiceInterface {
 public:
  virtual ~BufferManagerServiceInterface() = default;
  // Initialize and create Unix Socket Server(s) for talking
  // with clients.
  virtual absl::Status Initialize() = 0;
  // Start the server(s) for serving client requests.
  virtual absl::Status Start() = 0;
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_BUFFER_MANAGER_SERVICE_INTERFACE_H_

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_HOST_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_HOST_INTERFACE_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace tcpdirect {

enum class HealthStatus {
  kInitializing,
  kHealthy,
  kUnhealthy,
};

// Base initialization function for fastrak_gpumem_manager defined
// to execute host type specific initialization.
class FastrakGpumemManagerHostInterface {
 public:
  FastrakGpumemManagerHostInterface() = default;
  virtual ~FastrakGpumemManagerHostInterface() = default;
  virtual absl::Status Setup() = 0;
  // Set the health status of the fastrak_gpumem_manager.
  // "Message" can be used to provide additional information about the state
  // transition.
  virtual void SetHealthStatus(HealthStatus status,
                               absl::string_view message) = 0;
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_HOST_INTERFACE_H_

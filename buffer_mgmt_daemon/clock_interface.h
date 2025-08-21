/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_CLOCK_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_CLOCK_INTERFACE_H_

#include "absl/time/time.h"

namespace tcpdirect {

// Interface for a simple clock that is used by FastrakGpumemManagerCloud
// for getting the current time.
// This is needed for testing purposes so a mock clock impl can be injected.
class ClockInterface {
 public:
  virtual ~ClockInterface() = default;
  virtual absl::Time TimeNow() const = 0;
  virtual void SleepFor(absl::Duration duration) = 0;
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_CLOCK_INTERFACE_H_

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_CLOCK_H_
#define BUFFER_MGMT_DAEMON_CLOCK_H_

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/clock_interface.h"

namespace tcpdirect {
// Real implementation of the clock that returns the current time by simply
// calling absl::Now().
class Clock : public ClockInterface {
 public:
  absl::Time TimeNow() const override { return absl::Now(); }
  void SleepFor(absl::Duration duration) override { absl::SleepFor(duration); }
};
}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_CLOCK_H_

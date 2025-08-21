/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_GUEST_CLOCK_H_
#define DXS_CLIENT_GUEST_CLOCK_H_

#include <time.h>

#include <cstdint>

#include "dxs/clock-interface.h"

namespace dxs {

class GuestClock : public ClockInterface {
 public:
  GuestClock() = default;

  ~GuestClock() override = default;

  int64_t GetTime() const override {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000 * 1000 * 1000 * ts.tv_sec + ts.tv_nsec;
  }
};

inline ClockInterface*& DxsGuestClockSlot() {
  static ClockInterface* monotonic_clock = new dxs::GuestClock();
  return monotonic_clock;
}

class GlobalGuestClock {
 public:
  static int64_t GetTime() { return GetClock().GetTime(); }
  static const ClockInterface& GetClock() { return *DxsGuestClockSlot(); }
};

}  // namespace dxs

#endif  // DXS_CLIENT_GUEST_CLOCK_H_

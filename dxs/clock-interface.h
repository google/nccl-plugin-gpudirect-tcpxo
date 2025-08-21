/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLOCK_INTERFACE_H_
#define DXS_CLOCK_INTERFACE_H_

#include <stdint.h>

namespace dxs {

class ClockInterface {
 public:
  virtual ~ClockInterface() = default;
  ClockInterface(const ClockInterface&) = delete;
  ClockInterface& operator=(const ClockInterface&) = delete;

  // Returns the current monotonic time, in nanoseconds.
  virtual int64_t GetTime() const = 0;

 protected:
  ClockInterface() = default;
};

}  // namespace dxs

#endif  // DXS_CLOCK_INTERFACE_H_

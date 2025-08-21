/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_MONOTONIC_TIMESTAMP_H_
#define DXS_CLIENT_MONOTONIC_TIMESTAMP_H_

#include <ctime>

#include "absl/time/time.h"

namespace dxs {

using MonotonicTs = absl::Time;

inline MonotonicTs GetMonotonicTs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return absl::TimeFromTimespec(ts);
}

}  // namespace dxs

#endif  // DXS_CLIENT_MONOTONIC_TIMESTAMP_H_

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_ATOMIC_FUTURE_H_
#define DXS_CLIENT_ATOMIC_FUTURE_H_

#include <atomic>
#include <optional>

#include "absl/log/check.h"

namespace dxs {

// A testable future which can be Prepared once and tested an unlimited number
// of times.
//
// Thread compatible. Get is const and may be called unsynchronized.
template <class T>
class AtomicFuture {
 public:
  // Retrieves the value, or nullopt if unset.
  std::optional<T> Get() const {
    // value_ is only guaranteed to be set once Prepare makes a release store.
    if (!flag_.test(std::memory_order_acquire)) return std::nullopt;
    return value_;
  }
  // Sets the value, returns true if successful.
  bool Prepare(T t) {
    if (flag_.test(std::memory_order_relaxed)) return false;
    value_ = std::move(t);
    bool value_set = flag_.test_and_set(std::memory_order_release);
    DCHECK(!value_set);
    return true;
  }

 private:
  std::atomic_flag flag_;
  std::optional<T> value_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_ATOMIC_FUTURE_H_

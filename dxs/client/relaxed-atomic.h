/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_RELAXED_ATOMIC_H_
#define DXS_CLIENT_RELAXED_ATOMIC_H_

#include <atomic>

namespace dxs {

// A thread safe value that can only be stored and loaded, and does not imply
// any memory ordering between other operations.
template <class ValueType>
class RelaxedAtomic {
 public:
  explicit RelaxedAtomic(ValueType value) : value_(value) {}
  ValueType Load() const { return value_.load(std::memory_order_relaxed); }
  void Store(ValueType value) {
    value_.store(value, std::memory_order_relaxed);
  }

 private:
  std::atomic<ValueType> value_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_RELAXED_ATOMIC_H_

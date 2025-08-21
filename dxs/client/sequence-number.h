/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_SEQUENCE_NUMBER_H_
#define DXS_CLIENT_SEQUENCE_NUMBER_H_

#include <atomic>
#include <cstdint>

namespace dxs {

// A thread safe incrementing sequence number.
class SequenceNumber {
 public:
  explicit SequenceNumber(uint64_t initial) : next_(initial) {}

  SequenceNumber(const SequenceNumber&) = delete;
  SequenceNumber& operator=(const SequenceNumber&) = delete;

  ~SequenceNumber() = default;

  uint64_t Next() { return next_.fetch_add(1, std::memory_order_relaxed); }

 private:
  std::atomic<uint64_t> next_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_SEQUENCE_NUMBER_H_

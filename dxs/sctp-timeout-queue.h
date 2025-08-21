/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_SCTP_TIMEOUT_QUEUE_H_
#define DXS_SCTP_TIMEOUT_QUEUE_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>

#include "api/units/timestamp.h"
#include "dxs/clock-interface.h"
#include "net/dcsctp/public/timeout.h"
#include "net/dcsctp/public/types.h"
#include "ortools/base/adjustable_priority_queue-inl.h"
#include "ortools/base/adjustable_priority_queue.h"

namespace dxs {

constexpr uint64_t kNanosecondsPerMicrosecond = 1000;
constexpr uint64_t kNanosecondsPerMillisecond =
    1000 * kNanosecondsPerMicrosecond;

class SctpTimeoutQueue;

// A Timeout compatible with dcSctp.  When created, the timeout isn't in the
// parent queue yet.  When Start()'d the timeout is added to the parent queue.
// When Stopped, the timeout is removed from the parent queue.
class SctpTimeout : public dcsctp::Timeout {
 public:
  explicit SctpTimeout(SctpTimeoutQueue& parent_queue,
                       const ClockInterface& clock)
      : parent_queue_(parent_queue), clock_(clock) {}
  ~SctpTimeout() override { Stop(); }

  void Start(dcsctp::DurationMs duration,
             dcsctp::TimeoutID timeout_id) override;
  void Stop() override;
  void Restart(dcsctp::DurationMs duration,
               dcsctp::TimeoutID timeout_id) override;

  dcsctp::TimeoutID GetTimeoutID() const { return timeout_id_; }
  // Expiration returned in nanoseconds.
  int64_t GetExpiration() const { return expiration_; }

  void SetHeapIndex(int i) { heap_index_ = i; }
  int GetHeapIndex() const { return heap_index_; }
  int operator<(const SctpTimeout& other) const {
    // Higher priority is expiring sooner, so invert logic.
    return (expiration_ > other.expiration_) ? 1 : 0;
  }

  // clang-format off
  SctpTimeoutQueue& parent_queue_;
  const ClockInterface& clock_;
  // clang-format on
  dcsctp::TimeoutID timeout_id_;
  int64_t expiration_;

  // heap_index_ < 0 indicates this object is not in the parent_queue_.
  int heap_index_ = -1;
};

class SctpTimeoutHandlerInterface {
 public:
  virtual ~SctpTimeoutHandlerInterface() = default;
  virtual void HandleTimeout(dcsctp::TimeoutID timeout_id) = 0;
};

// This class adapts dcsctp Timeouts for use in the USPS engine framework.
// Timeouts are kept in a priority queue by expiration time, Run() calls
// timeout_handler_.HandleTimeout when they expire.
class SctpTimeoutQueue {
 public:
  explicit SctpTimeoutQueue(SctpTimeoutHandlerInterface& timeout_handler,
                            const ClockInterface& clock);

  void Run();

  std::unique_ptr<dcsctp::Timeout> CreateTimeout() {
    return static_cast<std::unique_ptr<dcsctp::Timeout>>(
        std::make_unique<SctpTimeout>(*this, clock_));
  }

  webrtc::Timestamp GetTime() {
    return webrtc::Timestamp::Micros(clock_.GetTime() /
                                     kNanosecondsPerMicrosecond);
  }

  std::optional<int> NextTimeoutMilliseconds() {
    if (timeouts_.IsEmpty()) return std::nullopt;

    // Returns the number of milliseconds until the next timeout, rounded up to
    // ensure the top timer will have expired if the caller sleeps for this
    // long.
    int64_t expiration_nanos =
        (timeouts_.Top()->GetExpiration() - clock_.GetTime());
    int64_t divisor = kNanosecondsPerMillisecond;
    return std::max(0l, (expiration_nanos + (divisor - 1)) / divisor);
  }

 private:
  friend class SctpTimeout;
  // These methods should only be called by SctpTimeout.  Manipulating the
  // priority queue directly could result in a SctpTimeout getting out of sync.
  void AddTimeout(SctpTimeout* timeout) { timeouts_.Add(timeout); }
  void RemoveTimeout(SctpTimeout* timeout) { timeouts_.Remove(timeout); }
  void UpdateTimeout(SctpTimeout* timeout) {
    timeouts_.NoteChangedPriority(timeout);
  }

  SctpTimeoutHandlerInterface& timeout_handler_;
  const ClockInterface& clock_;
  AdjustablePriorityQueue<SctpTimeout> timeouts_;
};

}  // namespace dxs
#endif  // DXS_SCTP_TIMEOUT_QUEUE_H_

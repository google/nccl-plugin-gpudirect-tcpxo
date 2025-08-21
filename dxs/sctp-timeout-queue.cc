/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/sctp-timeout-queue.h"

#include <cstdint>

#include "absl/log/check.h"
#include "dxs/clock-interface.h"
#include "net/dcsctp/public/types.h"

namespace dxs {

using dcsctp::DurationMs;
using dcsctp::TimeoutID;

void SctpTimeout::Start(DurationMs duration, TimeoutID timeout_id) {
  DCHECK_EQ(heap_index_, -1);
  timeout_id_ = timeout_id;
  expiration_ = clock_.GetTime() + ((*duration) * kNanosecondsPerMillisecond);
  parent_queue_.AddTimeout(this);
}

void SctpTimeout::Stop() {
  if (heap_index_ >= 0) {
    expiration_ = INT64_MAX;
    parent_queue_.RemoveTimeout(this);
    heap_index_ = -1;
  }
}
void SctpTimeout::Restart(DurationMs duration, TimeoutID timeout_id) {
  DCHECK_NE(heap_index_, -1);
  timeout_id_ = timeout_id;
  expiration_ = clock_.GetTime() + ((*duration) * kNanosecondsPerMillisecond);
  parent_queue_.UpdateTimeout(this);
}

SctpTimeoutQueue::SctpTimeoutQueue(SctpTimeoutHandlerInterface& timeout_handler,
                                   const ClockInterface& clock)
    : timeout_handler_(timeout_handler), clock_(clock) {}

void SctpTimeoutQueue::Run() {
  auto now = clock_.GetTime();

  while (!timeouts_.IsEmpty()) {
    SctpTimeout* timeout = timeouts_.Top();
    if (timeout->GetExpiration() <= now) {
      // DcSctpSocket may do this too, but it is ok to call Stop() multiple
      // times on a SctpTimer.
      timeout->Stop();
      // Stop() will pop this timeout off the queue.
      DCHECK(timeouts_.IsEmpty() || timeouts_.Top() != timeout);
      timeout_handler_.HandleTimeout(timeout->GetTimeoutID());

    } else {
      // No more timers are expired.
      break;
    }
  }
  return;
}

}  // namespace dxs

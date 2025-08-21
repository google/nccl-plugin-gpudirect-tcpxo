/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_SPSC_QUEUE_SPSC_MESSAGING_QUEUE_PAIR_H_
#define DXS_CLIENT_SPSC_QUEUE_SPSC_MESSAGING_QUEUE_PAIR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dxs/client/spsc_queue/spsc_queue_pair.h"
#include "dxs/client/spsc_queue/spsc_queue_pair.pb.h"

namespace dxs {

// This is a wrapper around SpscQueuePair providing message boundary and
// cache line alignment.
// The message structure is:
// - 4bytes header (1byte reserved, 3bytes body length).
// - body, length indicated by the header.
// - padding to the next kAlignment boundary
//   i.e. 4 + body.len + padding.len == 64 * k
// Note that the padding length is not explicitly indicated,
// the receiver side infers that from the body length.
// The padding is NOT zero-filled for each message. It may contain stale data
// but all such data where once sent to/from us. So it's no more than what
// the client already knows, thus safe.
class SpscMessagingQueuePair {
 public:
  static constexpr uint8_t kAlignment = 64;
  static constexpr uint64_t kMaxMessageSize =
      16 * 1024 * 1024 - 1;  // 2^24 - 1, i.e. 16M

  struct MessageHeader {
    uint32_t reserved : 8;     // reserved, should be filled with zero.
    uint32_t body_bytes : 24;  // not including the length of the header itself.
  };
  static_assert(sizeof(MessageHeader) == 4);
  static_assert(kAlignment == ABSL_CACHELINE_SIZE, "Unexpected cacheline size");

  // The two regions should already be zero-filled.
  // They need to be zero filled before being passed to the peer.
  // They must be size of (2^N+1)*4KB and aligned to 4K.
  //
  // local_region: peer writes will be reflected at this address.
  // remote_region: writes are made to this address.
  static absl::StatusOr<std::unique_ptr<SpscMessagingQueuePair>> Create(
      absl::Span<volatile uint8_t> local_region,
      absl::Span<volatile uint8_t> remote_region);

  // Constructor for hitless restart. Restores a queue from SaveState().
  // Similar to Create(...) with a few differences:
  // - No need to zero-fill the regions. The local_region may contain new
  //   messages that the peer sent while we are away.
  // - The size of the two regions must be exactly the same as before
  //   (no resize).
  // - The virtual address of the region within the process' address space
  //   may change.
  static absl::StatusOr<std::unique_ptr<SpscMessagingQueuePair>> Restore(
      absl::Span<volatile uint8_t> local_region,
      absl::Span<volatile uint8_t> remote_region,
      const SpscQueuePairState& state);

  // No copy or move
  SpscMessagingQueuePair(const SpscMessagingQueuePair&) = delete;
  SpscMessagingQueuePair& operator=(const SpscMessagingQueuePair&) = delete;
  SpscMessagingQueuePair(SpscMessagingQueuePair&&) = delete;
  SpscMessagingQueuePair& operator=(SpscMessagingQueuePair&&) = delete;

  // Send a msg of the given size.
  // Possible error cases:
  // - Remote full : kResourceExhausted
  // - Msg too large : kInvalidArgument
  absl::StatusCode Send(absl::Span<const uint8_t> msg);

  // Receive a message. It's guaranteed that the length is preserved and all msg
  // is received in order. The queue space will be recycled after the handler
  // returns. In case of an error, the handler will not be called. If only
  // partial message is available, it's treated as if no message is available,
  // despite this should not happen if the above `Send()` function is used.
  // Every call to this function will receive at most one message, so the
  // handler will be called at most once.
  //
  // typedef void handler(Span<...> segment1, Span<...> segment2)
  //   The spans passed to the handler will cover only the body, not including
  //   the header nor paddings. If the body wraps around the end of the
  //   circular buffer, two spans will be returned. Otherwise segment2 will
  //   be a zero-length span.
  //
  // Possible error cases:
  // - Message is not ready : kUnavailable
  absl::StatusCode Receive(
      absl::FunctionRef<void(absl::Span<const volatile uint8_t>,
                             absl::Span<const volatile uint8_t>)>
          handler);

  // Same as Receive(FuncRef<...>), but do a copy and returns the data as
  // std::string. The error code is the same as the `Receive(...)` call above.
  absl::StatusOr<std::string> Receive();

  // Hitless restart helper functions
  // - The queue should not be used to Send or Receive after SaveState().
  //   So please make sure any spin poll threads are stopped before calling
  //   this function.
  // - Use SpscMessagingQueuePair::Restore(...) to restore the saved queue.
  // Note that the queue does not own the memory regions. It's the caller's
  // responsibility to transfer and recreate using the same regions.
  SpscQueuePairState SaveState() const { return qp_.SaveState(); }

 private:
  explicit SpscMessagingQueuePair(SpscQueuePair&& qp) : qp_(std::move(qp)) {}
  SpscQueuePair qp_;
};
}  // namespace dxs

#endif  // DXS_CLIENT_SPSC_QUEUE_SPSC_MESSAGING_QUEUE_PAIR_H_

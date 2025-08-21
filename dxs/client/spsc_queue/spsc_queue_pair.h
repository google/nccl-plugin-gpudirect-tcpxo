/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_SPSC_QUEUE_SPSC_QUEUE_PAIR_H_
#define DXS_CLIENT_SPSC_QUEUE_SPSC_QUEUE_PAIR_H_

#include <cstddef>
#include <cstdint>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dxs/client/spsc_queue/spsc_queue_pair.pb.h"

namespace dxs {

// SpscQueuePair comprises a pair of in-memory circular buffers that are
// intended for simple bidirectional communication over MMIO address ranges.
// SendMessage pushes data via posted MMIO writes to the local memory of the
// receiver.  ReceiveMessage consumes data in the local address space and pushes
// a running total of the amount of data consumed to the sender via an MMIO
// write.  The protocol never performs and MMIO reads.  The memory for the
// circular buffers is not owned by SpscQueuePair and must live longer than the
// SpscQueuePair object.  The methods in the class are thread compatible, but
// not thread safe.  The design of SpscQueuePair is described in

class SpscQueuePair {
 public:
  static constexpr size_t kPageSize = 4096;

  // The queue is shared by two nodes. ABSL_CACHELINE_SIZE is architecture
  // dependent, so we put an explicit assertion here.
  static_assert(ABSL_CACHELINE_SIZE == 64,
                "Unexpected cacheline size for this architecture");
  // The two variables takes one full cache line each. Occupying the beginning
  // of a single page.
  struct Doorbells {
    uint64_t bytes_produced ABSL_CACHELINE_ALIGNED;
    uint64_t remote_bytes_consumed ABSL_CACHELINE_ALIGNED;
    // 8 uint64 per cacheline, 62 unused cachelines.
    uint64_t padding[8 * 62];
  } __attribute__((aligned(kPageSize)));
  static constexpr uint64_t kDoorbellsSize = sizeof(Doorbells);
  static_assert(kDoorbellsSize == kPageSize);

  // Represents an ongoing send attempt. The batch will be implicitly aborted
  // on destruction.
  class SendBatch {
   public:
    // Returns kResourceExhausted if there's not enough space in the queue.
    absl::StatusCode Append(const void* buffer, size_t size);
    absl::StatusCode Append(absl::Span<const uint8_t> buffer) {
      return Append(buffer.data(), buffer.size());
    }
    // Advances the write head without touching data.
    absl::StatusCode Skip(size_t size);
    // This batch is invalid after Commit().
    // Call like this `std::move(batch).Commit()`
    absl::StatusCode Commit() &&;

   private:
    friend class SpscQueuePair;

    SendBatch(SpscQueuePair& queue, absl::Span<volatile uint8_t> segment,
              absl::Span<volatile uint8_t> backup_segment)
        : queue_(queue),
          segment_(segment),
          backup_segment_(backup_segment),
          pending_bytes_(0) {}

    SpscQueuePair& queue_;
    absl::Span<volatile uint8_t> segment_;
    absl::Span<volatile uint8_t> backup_segment_;
    uint64_t pending_bytes_;
  };

  // Represents an ongoing receive attempt. Doesn't update the "consumed_bytes"
  // counter until Commit is called. The available data is pointed by
  // FirstSegment() and optional SecondSegment().
  // If data does not cross ring boundary:
  //     [---<FirstSegment>--------------]
  //     [ and ] mark the whole ring.
  //     < and > mark the data region.
  //     SecondSegment is empty
  // if data crosses the ring boundary:
  //     [SecondSegment>----<FirstSegment]
  class ReceiveBatch {
   public:
    // A shortcut for memcpy(buffer, first_segment+second_segment, bytes)
    // If buffer is nullptr, then data is discarded, but pointer is still
    // advanced.
    // Returns kUnavailable if there are not enough bytes.
    absl::StatusCode Recv(void* buffer, uint64_t bytes);
    // Shortcut for Recv(nullptr, bytes)
    absl::StatusCode RemovePrefix(uint64_t bytes) {
      return Recv(nullptr, bytes);
    }

    // Look directly into the ring for rx.
    absl::Span<const volatile uint8_t> FirstSegment() const { return segment_; }
    absl::Span<const volatile uint8_t> SecondSegment() const {
      return backup_segment_;
    }

    absl::StatusCode Commit() &&;

    uint64_t RemainingBytes() const {
      return segment_.size() + backup_segment_.size();
    };

   private:
    friend class SpscQueuePair;

    SpscQueuePair* queue_ = nullptr;
    absl::Span<const volatile uint8_t> segment_;
    absl::Span<const volatile uint8_t> backup_segment_;
    uint64_t taken_bytes_ = 0;
  };

  // The caller must zerofill the local doorbells & ring before exchanging info
  // with the remote. So that stale data isn't leaked to the peer. However,
  // they don't need to be all zero when passed to this function. e.g. the
  // queue is created as part of hitless restore.
  //
  // The doorbell span must be at least kDoorbellsSize long.
  // The rings must be 2^N * 4KB in size. The two rings don't have to be the
  // same size. All doorbells and rings must be 4K page-aligned.
  //
  // local_doorbells:  the doorbell "registers" that the peer uses to indicate
  //                   the arrival & consumption of messages.
  // local_ring:       where the message from the peer should land on.
  // remote_doorbells: the doorbell "registers" that's used to notify the peer
  //                   of new & consumption of messages.
  // remote_ring:      where the message for the peer should be written to.
  //
  // The backing memory of local_* should be local (i.e. low read cost).
  // The local_* memory will be the remote_* memory for the peer.
  static absl::StatusOr<SpscQueuePair> MakeSpscQueuePair(
      absl::Span<volatile uint8_t> local_doorbells,
      absl::Span<volatile uint8_t> local_ring,
      absl::Span<volatile uint8_t> remote_doorbells,
      absl::Span<volatile uint8_t> remote_ring);

  // Special functions. Forbids coping so that we don't accidentally have two
  // objects for the same queue.
  ~SpscQueuePair() = default;
  SpscQueuePair(const SpscQueuePair&) = delete;
  SpscQueuePair(SpscQueuePair&&) = default;

  // Start a new sending batch. Must be called if the previous batch is
  // committed or destructed. There can only be one outstanding batch at a time.
  // NOTE: SendBatch holds a pointer to `*this` so it must not move while a
  // batch is outstanding.
  SendBatch BeginSend();

  // Expose ALL available bytes via `batch`.
  // Returns kUnavailable if there's no byte available.
  //
  // NOTE: ReceiveBatch holds a pointer to `*this` so it must not move while a
  // batch is outstanding.
  absl::StatusCode BeginReceive(ReceiveBatch& batch);

  // Hitless restart helper functions
  // - The queue should not be used to Send or Receive after SaveState().
  // - RestoreState() should be called immediately after construction.
  //   The queue should not be used before that.
  // - The ring size cannot change across restart. Not even increase.
  // Note that the queue does not own the memory regions. It's the caller's
  // responsibility to transfer and recreate using the same region.
  SpscQueuePairState SaveState() const;
  absl::Status RestoreState(const SpscQueuePairState& state);

 private:
  SpscQueuePair(volatile Doorbells* local_doorbells,
                absl::Span<volatile uint8_t> local_ring,
                volatile Doorbells* remote_doorbells,
                absl::Span<volatile uint8_t> remote_ring);

  // Warmup the queue by reading all doorbells and rings.
  void Warmup() const;

  // Neither local nor remote region memory is owned by SpscQueuePair. They
  // must outlive the lifetime of SpscQueuePair.
  volatile const Doorbells* local_doorbells_;
  absl::Span<volatile uint8_t> local_ring_;
  volatile Doorbells* remote_doorbells_;
  absl::Span<volatile uint8_t> remote_ring_;
  // How many bytes are consumed by local side.
  uint64_t local_bytes_consumed_;
  // How many bytes are sent to remote.
  uint64_t remote_bytes_produced_;
  // Should be equal to {local,remote}_ring_.size() - 1
  // Optimizes "mod size" to be "& mask".
  const uint64_t local_ring_size_mask_;
  const uint64_t remote_ring_size_mask_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_SPSC_QUEUE_SPSC_QUEUE_PAIR_H_

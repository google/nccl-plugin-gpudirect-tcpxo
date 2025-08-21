/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/spsc_queue/spsc_queue_pair.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dxs/client/oss/barrier.h"
#include "dxs/client/oss/mmio.h"
#include "dxs/client/oss/status_macros.h"
#include "dxs/client/spsc_queue/spsc_queue_pair.pb.h"

namespace dxs {
namespace {

// Copy n bytes from dst to src, moving 4 bytes at a time. Both dst
// and src are assumed to be properly 4-byte aligned. n is
// assumed to be a multiple of 4.
static inline void MemcpyToMmio32(volatile uint32_t* dst, uint32_t* src,
                                  size_t n) {
  DCHECK_EQ(n % 4, 0);
  while (n > 0) {
    platforms_util::MmioWriteRelaxed32(dst, *src);
    dst++;
    src++;
    n -= 4;
  }
}

static inline void MemcpyToMmio8(volatile uint8_t* dst, uint8_t* src,
                                 size_t n) {
  while (n > 0) {
    platforms_util::MmioWriteRelaxed8(dst, *src);
    dst++;
    src++;
    n -= 1;
  }
}

volatile void* MemcpyToMmio(volatile void* mmio_destination, const void* source,
                            size_t length) {
  uint64_t dst = reinterpret_cast<uint64_t>(mmio_destination);
  uint64_t src = reinterpret_cast<uint64_t>(source);

  if (dst % 4 == 0 && src % 4 == 0) {
    size_t words = length / 4;

    MemcpyToMmio32(reinterpret_cast<volatile uint32_t*>(dst),
                   reinterpret_cast<uint32_t*>(src), words * 4);
    MemcpyToMmio8(reinterpret_cast<volatile uint8_t*>(dst + words * 4),
                  reinterpret_cast<uint8_t*>(src + words * 4), length % 4);
    return mmio_destination;
  }
  MemcpyToMmio8(reinterpret_cast<volatile uint8_t*>(dst),
                reinterpret_cast<uint8_t*>(src), length);
  return mmio_destination;
}
}  // namespace

absl::StatusOr<SpscQueuePair> SpscQueuePair::MakeSpscQueuePair(
    absl::Span<volatile uint8_t> local_doorbells,
    absl::Span<volatile uint8_t> local_ring,
    absl::Span<volatile uint8_t> remote_doorbells,
    absl::Span<volatile uint8_t> remote_ring) {
  auto check_region = [](absl::Span<volatile uint8_t> doorbells,
                         absl::Span<volatile uint8_t> ring,
                         std::string_view name) -> absl::Status {
    if (doorbells.size() < kDoorbellsSize) {
      return absl::InvalidArgumentError(
          absl::StrCat(name, " doorbells size is less than kDoorbellsSize"));
    }
    if ((reinterpret_cast<uintptr_t>(doorbells.data()) % kPageSize) != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat(name, " doorbells are not page-aligned"));
    }
    if (ring.size() % kPageSize != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat(name, " ring size is not a multiple of 4K"));
    }
    if (!absl::has_single_bit(ring.size())) {
      return absl::InvalidArgumentError(
          absl::StrCat(name, " ring size is not a power of two"));
    }
    if ((reinterpret_cast<uintptr_t>(ring.data()) % kPageSize) != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat(name, " ring is not page-aligned"));
    }
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(check_region(local_doorbells, local_ring, "Local region"));
  RETURN_IF_ERROR(check_region(remote_doorbells, remote_ring, "Remote region"));
  return SpscQueuePair(
      reinterpret_cast<volatile Doorbells*>(local_doorbells.data()), local_ring,
      reinterpret_cast<volatile Doorbells*>(remote_doorbells.data()),
      remote_ring);
}

SpscQueuePair::SpscQueuePair(volatile Doorbells* local_doorbells,
                             absl::Span<volatile uint8_t> local_ring,
                             volatile Doorbells* remote_doorbells,
                             absl::Span<volatile uint8_t> remote_ring)
    : local_doorbells_(local_doorbells),
      local_ring_(local_ring),
      remote_doorbells_(remote_doorbells),
      remote_ring_(remote_ring),
      local_bytes_consumed_(0),
      remote_bytes_produced_(0),
      local_ring_size_mask_(local_ring.size() - 1),
      remote_ring_size_mask_(remote_ring.size() - 1) {
  DCHECK(absl::has_single_bit(local_ring.size()));
  DCHECK(absl::has_single_bit(remote_ring.size()));
  Warmup();
}

void SpscQueuePair::Warmup() const {
  // These reads will not be optimized away because they are all volatile.
  (void)local_doorbells_->bytes_produced;
  for (auto i = 0u; i < local_ring_.size(); i++) {
    (void)*(local_ring_.data() + i);
  }
  if (local_doorbells_->remote_bytes_consumed != 0) {
    // Reading from remote memory is expensive and it takes a few milliseconds.
    // So skipping that during hitless restart.
    return;
  }

  (void)remote_doorbells_->bytes_produced;
  (void)remote_doorbells_->remote_bytes_consumed;
  for (auto i = 0u; i < remote_ring_.size(); i++) {
    (void)*(remote_ring_.data() + i);
  }
}

SpscQueuePair::SendBatch SpscQueuePair::BeginSend() {
  // Check bytes_produced (locally visible value) against the
  // remote_bytes_consumed (exposed to PCIe) to make sure there is space.  We
  // use MmioReadAcquire to ensure correct ordering semantics against preceding
  // Mmio writes to remote_bytes_consumed, but this is not actually an MMIO
  // read; local_doorbells_ is in local memory.
  const uint64_t remote_ring_used =
      remote_bytes_produced_ - platforms_util::MmioReadAcquire64(
                                   &local_doorbells_->remote_bytes_consumed);
  DCHECK_LE(remote_ring_used, remote_ring_.size());
  const uint64_t remote_ring_free = remote_ring_.size() - remote_ring_used;
  if (remote_ring_free > remote_ring_.size()) {
    LOG_EVERY_N_SEC(DFATAL, 1) << "Broken invariant. Malicious peer?";
    return {*this, {}, {}};
  }

  // Modulo bytes_produced (locally visible value) by the data_buffer size
  // to compute the data start address MMIO write message into data_buffer
  // at the appropriate offset (potentially dealing with wrap)
  const uint64_t remote_ring_free_offset =
      remote_bytes_produced_ & remote_ring_size_mask_;
  if (remote_ring_free > remote_ring_.size() - remote_ring_free_offset) {
    // Free space wraps.
    const uint64_t segment_size = remote_ring_.size() - remote_ring_free_offset;
    const uint64_t backup_segment_size = remote_ring_free - segment_size;
    return {*this, remote_ring_.subspan(remote_ring_free_offset, segment_size),
            remote_ring_.subspan(0, backup_segment_size)};
  } else {
    // Free space does not wrap.
    return {*this,
            remote_ring_.subspan(remote_ring_free_offset, remote_ring_free),
            {}};
  }
}

absl::StatusCode SpscQueuePair::SendBatch::Append(const void* buffer,
                                                  size_t size) {
  if (segment_.size() + backup_segment_.size() < size) {
    // Appending message to buffer will overwrite unreceived data
    return absl::StatusCode::kResourceExhausted;
  }

  // Copy the first segment.
  const uint64_t len1 = std::min(size, segment_.size());
  MemcpyToMmio(segment_.data(), buffer, len1);
  segment_.remove_prefix(len1);
  pending_bytes_ += len1;
  if (len1 == size) {
    // It's valid for segment_.size == 0 && backup_segment_.size > 0.
    // backup_ring will be promoted on next call.
    return absl::StatusCode::kOk;
  }
  buffer = reinterpret_cast<const uint8_t*>(buffer) + len1;

  // Copy the second segment.
  const uint64_t len2 = size - len1;
  segment_ = backup_segment_;
  backup_segment_ = {};

  MemcpyToMmio(segment_.data(), buffer, len2);
  segment_.remove_prefix(len2);
  pending_bytes_ += len2;

  return absl::StatusCode::kOk;
}

absl::StatusCode SpscQueuePair::SendBatch::Skip(size_t size) {
  if (segment_.size() + backup_segment_.size() < size) {
    return absl::StatusCode::kResourceExhausted;
  }

  if (size <= segment_.size()) {
    segment_.remove_prefix(size);
  } else {
    backup_segment_.remove_prefix(size - segment_.size());
    segment_ = backup_segment_;
    backup_segment_ = {};
  }
  pending_bytes_ += size;

  return absl::StatusCode::kOk;
}

absl::StatusCode SpscQueuePair::SendBatch::Commit() && {
  // MMIO barrier ensures memcpy is done before the releasing store below to
  // remote_bytes_produced.
  platforms_util::MmioWriteBarrier();
  // Update bytes_produced (locally visible value)
  queue_.remote_bytes_produced_ += pending_bytes_;
  // update bytes_produced at consumer to match local value
  platforms_util::MmioWriteRelease64(&queue_.remote_doorbells_->bytes_produced,
                                     queue_.remote_bytes_produced_);
  // Flush the doorbell using sfence.
  platforms_util::MmioWriteBarrier();

  return absl::StatusCode::kOk;
}

absl::StatusCode SpscQueuePair::BeginReceive(ReceiveBatch& batch) {
  // Read bytes_produced (exposed_to_PCIe) and compare it to bytes_consumed
  // (locally visible value) to see how much data arrived.  We use
  // MmioReadAcquire to ensure correct ordering semantics against preceding Mmio
  // writes to remote_bytes_consumed, but this is not actually an MMIO read;
  // local_doorbells_ is in local memory.
  const uint64_t bytes_produced =
      platforms_util::MmioReadAcquire64(&local_doorbells_->bytes_produced);
  const uint64_t bytes_available = bytes_produced - local_bytes_consumed_;
  if (bytes_available == 0) {
    // No new data
    return absl::StatusCode::kUnavailable;
  }
  if (bytes_available > local_ring_.size()) {
    LOG_EVERY_N_SEC(DFATAL, 1) << "Broken invariant. Malicious peer?";
    return absl::StatusCode::kOutOfRange;
  }

  const uint64_t start_offset = local_bytes_consumed_ & local_ring_size_mask_;
  const uint64_t wrap_length =
      std::min(bytes_available, local_ring_.size() - start_offset);

  // First segment
  batch.queue_ = this;
  batch.segment_ = local_ring_.subspan(start_offset, wrap_length);
  batch.backup_segment_ = {};
  batch.taken_bytes_ = 0;

  // Second segment
  if (wrap_length < bytes_available) {
    batch.backup_segment_ =
        local_ring_.subspan(0, bytes_available - wrap_length);
  }

  return absl::StatusCode::kOk;
}

absl::StatusCode SpscQueuePair::ReceiveBatch::Recv(void* buffer,
                                                   uint64_t bytes) {
  if (RemainingBytes() < bytes) {
    // Complete message is not available or buffer is incorrect size
    return absl::StatusCode::kUnavailable;
  }

  // First segment
  const uint64_t take1 = std::min(bytes, segment_.size());
  if (buffer) {
    std::copy(segment_.data(), segment_.data() + take1,
              static_cast<uint8_t*>(buffer));
    buffer = reinterpret_cast<uint8_t*>(buffer) + take1;
  }
  segment_.remove_prefix(take1);
  taken_bytes_ += take1;

  // Unlike Send, we actively promote the backup_segment_.
  if (segment_.empty()) {
    segment_ = backup_segment_;
    backup_segment_ = {};
  }

  if (take1 == bytes) {
    return absl::StatusCode::kOk;
  }

  // Second segment
  const uint64_t take2 = bytes - take1;
  if (buffer) {
    std::copy(segment_.data(), segment_.data() + take2,
              static_cast<uint8_t*>(buffer));
  }
  segment_.remove_prefix(take2);
  taken_bytes_ += take2;

  return absl::StatusCode::kOk;
}

absl::StatusCode SpscQueuePair::ReceiveBatch::Commit() && {
  // MMIO barrier ensures memcpy is done before the releasing store below to
  // remote_bytes_consumed.
  platforms_util::MmioWriteBarrier();
  // Update bytes_consumed (locally visible value)
  queue_->local_bytes_consumed_ += taken_bytes_;
  // MMIO write bytes_consumed to set the producer side to be equal to the local
  // value
  platforms_util::MmioWriteRelease64(
      &queue_->remote_doorbells_->remote_bytes_consumed,
      queue_->local_bytes_consumed_);
  return absl::StatusCode::kOk;
}

SpscQueuePairState SpscQueuePair::SaveState() const {
  SpscQueuePairState state;
  state.set_local_bytes_consumed(local_bytes_consumed_);
  state.set_remote_bytes_produced(remote_bytes_produced_);
  return state;
}

absl::Status SpscQueuePair::RestoreState(const SpscQueuePairState& state) {
  if (local_bytes_consumed_ != 0 || remote_bytes_produced_ != 0) {
    return absl::FailedPreconditionError(
        "Cannot restore to an unclean SpscQueuePair");
  }

  uint64_t expected_consumed = platforms_util::MmioReadAcquire64(
      &remote_doorbells_->remote_bytes_consumed);
  uint64_t expected_produced =
      platforms_util::MmioReadAcquire64(&remote_doorbells_->bytes_produced);
  if (state.local_bytes_consumed() != expected_consumed ||
      state.remote_bytes_produced() != expected_produced) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "state mismatch: %v expected_consumed=%v expected_produced=%v", state,
        expected_consumed, expected_produced));
  }
  local_bytes_consumed_ = state.local_bytes_consumed();
  remote_bytes_produced_ = state.remote_bytes_produced();
  return absl::OkStatus();
}

}  // namespace dxs

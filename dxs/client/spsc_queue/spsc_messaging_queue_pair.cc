/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/spsc_queue/spsc_messaging_queue_pair.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "dxs/client/oss/status_macros.h"
#include "dxs/client/spsc_queue/spsc_queue_pair.h"
#include "dxs/client/spsc_queue/spsc_queue_pair.pb.h"

namespace dxs {

namespace {

// Return the largest multiple of 'align' which is <= x (x >= 0).
template <typename T>
inline constexpr T AlignDownTo(T x, T align) {
  DCHECK_GE(x, 0);
  DCHECK_GT(align, 0);
  return align * (x / align);
}

// Return the smallest multiple of 'align' which is >= x (x >= 0).
template <typename T>
inline constexpr T AlignUpTo(T x, T align) {
  return AlignDownTo(x + align - 1, align);
}

// Given body length, compute the padding length.
constexpr uint64_t PaddingBytes(uint64_t body_bytes) {
  return AlignUpTo<uint64_t>(
             body_bytes + sizeof(SpscMessagingQueuePair::MessageHeader),
             SpscMessagingQueuePair::kAlignment) -
         sizeof(SpscMessagingQueuePair::MessageHeader) - body_bytes;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SpscMessagingQueuePair>>
SpscMessagingQueuePair::Create(absl::Span<volatile uint8_t> local_region,
                               absl::Span<volatile uint8_t> remote_region) {
  ASSIGN_OR_RETURN(SpscQueuePair qp,
                   SpscQueuePair::MakeSpscQueuePair(
                       local_region.subspan(0, SpscQueuePair::kDoorbellsSize),
                       local_region.subspan(SpscQueuePair::kDoorbellsSize),
                       remote_region.subspan(0, SpscQueuePair::kDoorbellsSize),
                       remote_region.subspan(SpscQueuePair::kDoorbellsSize)));
  return absl::WrapUnique(new SpscMessagingQueuePair(std::move(qp)));
}

absl::StatusOr<std::unique_ptr<SpscMessagingQueuePair>>
SpscMessagingQueuePair::Restore(absl::Span<volatile uint8_t> local_region,
                                absl::Span<volatile uint8_t> remote_region,
                                const SpscQueuePairState& state) {
  if (state.remote_bytes_produced() % kAlignment != 0 ||
      state.local_bytes_consumed() % kAlignment != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("state values are not aligned: ", state));
  }
  ASSIGN_OR_RETURN(auto ret, Create(local_region, remote_region));
  RETURN_IF_ERROR(ret->qp_.RestoreState(state));
  return ret;
}

absl::StatusCode SpscMessagingQueuePair::Send(absl::Span<const uint8_t> msg) {
  if (msg.size() > kMaxMessageSize) {
    return absl::StatusCode::kInvalidArgument;
  }

  MessageHeader header{
      .body_bytes = static_cast<uint32_t>(msg.size()),
  };

  // Note: *this should not be moved while a batch is outstanding.
  SpscQueuePair::SendBatch batch = qp_.BeginSend();
  // header
  auto status = batch.Append(&header, sizeof(header));
  if (status != absl::StatusCode::kOk) return status;
  // body
  status = batch.Append(msg);
  if (status != absl::StatusCode::kOk) return status;
  // padding
  status = batch.Skip(PaddingBytes(msg.size()));
  if (status != absl::StatusCode::kOk) return status;

  return std::move(batch).Commit();
}

absl::StatusCode SpscMessagingQueuePair::Receive(
    absl::FunctionRef<void(absl::Span<const volatile uint8_t>,
                           absl::Span<const volatile uint8_t>)>
        handler) {
  SpscQueuePair::ReceiveBatch batch;
  // Note: *this should not be moved while a batch is outstanding.
  auto status = qp_.BeginReceive(batch);
  if (status != absl::StatusCode::kOk) return status;
  // header
  uint64_t body_bytes;
  if (batch.FirstSegment().size() < sizeof(MessageHeader)) {
    LOG_EVERY_N_SEC(DFATAL, 1) << "MessageHeader should never cross the queue "
                                  "boundary, fallback to copy";
    MessageHeader header;
    status = batch.Recv(&header, sizeof(header));
    if (status != absl::StatusCode::kOk) return status;
    body_bytes = header.body_bytes;
  } else {
    body_bytes = reinterpret_cast<const volatile MessageHeader*>(
                     batch.FirstSegment().data())
                     ->body_bytes;
    batch.RemovePrefix(sizeof(MessageHeader));
  }
  if (batch.RemainingBytes() < body_bytes) {
    LOG_EVERY_N_SEC(DFATAL, 1) << "Message received but incomplete";
    return absl::StatusCode::kUnavailable;
  }
  // Body
  const uint64_t first_segment_size = batch.FirstSegment().size();
  if (body_bytes > first_segment_size) {
    handler(batch.FirstSegment(),
            batch.SecondSegment().subspan(0, body_bytes - first_segment_size));
  } else {
    handler(batch.FirstSegment().subspan(0, body_bytes), {});
  }
  batch.RemovePrefix(body_bytes);
  // padding
  const uint64_t padding_bytes = PaddingBytes(body_bytes);
  status = batch.RemovePrefix(padding_bytes);
  if (status != absl::StatusCode::kOk) return status;

  return std::move(batch).Commit();
}

absl::StatusOr<std::string> SpscMessagingQueuePair::Receive() {
  std::string ret;
  auto append_to_ret = [&ret](absl::Span<const volatile uint8_t> first,
                              absl::Span<const volatile uint8_t> second) {
    const size_t offset_first = ret.size();
    const size_t offset_second = offset_first + first.size();
    ret.resize(offset_second + second.size());
    std::copy(first.begin(), first.end(), ret.data() + offset_first);
    std::copy(second.begin(), second.end(), ret.data() + offset_second);
  };
  absl::StatusCode status = Receive(append_to_ret);
  if (status != absl::StatusCode::kOk) {
    return absl::Status(status, "SpecQueue::Receive(...) failed");
  }
  return ret;
}

}  // namespace dxs

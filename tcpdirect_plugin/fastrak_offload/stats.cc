/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/stats.h"

#include <time.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

namespace fastrak {

absl::Time Timer::Restart() {
  init_time_ = GetMonotonicTs();
  return init_time_;
}

absl::Duration Timer::GetElapsed() { return GetMonotonicTs() - init_time_; }

absl::Status Timer::CheckTimeout(absl::Duration timeout) {
  if (timeout == absl::ZeroDuration()) {
    return absl::OkStatus();
  }
  auto time_elapsed = GetElapsed();
  if (time_elapsed > timeout) {
    return absl::DeadlineExceededError(
        absl::StrCat("Time elapsed ", absl::FormatDuration(time_elapsed),
                     " over timeout: ", absl::FormatDuration(timeout)));
  }
  return absl::OkStatus();
}

DistributionBucketer::DistributionBucketer(double max_bucket, double scale)
    : buckets_(ceil(log(max_bucket / scale) / log(1.2)) + 1) {
  for (size_t i = 0; i < buckets_.size(); ++i) {
    buckets_[i].upper_bound = std::pow(1.2, i) * scale;
  }
}

std::vector<int64_t> DistributionBucketer::GetBucketCounts() {
  std::vector<int64_t> tmp;
  tmp.reserve(buckets_.size());
  for (auto& bucket : buckets_) {
    tmp.push_back(bucket.count);
  }
  return tmp;
}

std::vector<double> DistributionBucketer::GetBucketBounds() {
  std::vector<double> tmp;
  tmp.reserve(buckets_.size());
  for (auto& bucket : buckets_) {
    tmp.push_back(bucket.upper_bound);
  }
  return tmp;
}

void DistributionBucketer::Submit(double sample) {
  const auto upper = std::upper_bound(buckets_.begin(), buckets_.end(), sample,
                                      [](double value, const bucket& bucket) {
                                        return value < bucket.upper_bound;
                                      });
  size_t index = upper - buckets_.begin();
  if (index >= buckets_.size()) index = buckets_.size() - 1;
  buckets_[index].count++;
  // Doing (avg_ * count_ + sample) / (count_ + 1) has possibility of overflow
  avg_ = avg_ * (static_cast<double>(count_) / (count_ + 1)) +
         sample / (count_ + 1);
  ++count_;
  sample_submitted_ = true;
  if (sample < min_) min_ = sample;
  if (sample > max_) max_ = sample;
}

std::optional<double> DistributionBucketer::GetMin() {
  if (sample_submitted_) return min_;
  return std::nullopt;
}

std::optional<double> DistributionBucketer::GetMax() {
  if (sample_submitted_) return max_;
  return std::nullopt;
}

std::optional<double> DistributionBucketer::GetAvg() {
  if (sample_submitted_) return avg_;
  return std::nullopt;
}

std::string DistributionBucketer::Dump() {
  std::string ret;

  double coalesced_lower_bound = 0.0;
  double coalesced_upper_bound = 0.0;
  double prev_upper_bound = 0.0;
  bool coalesced = false;

  auto min = GetMin();
  if (min.has_value()) {
    absl::StrAppend(&ret, "Min: ", min.value(), "\n");
  }

  auto max = GetMax();
  if (max.has_value()) {
    absl::StrAppend(&ret, "Max: ", max.value(), "\n");
  }

  auto avg = GetAvg();
  if (avg.has_value()) {
    absl::StrAppend(&ret, "Avg: ", avg.value(), "\n");
  }

  for (size_t i = 0; i < buckets_.size(); ++i) {
    const auto count = buckets_[i].count;

    if (!count) {
      if (!coalesced) {
        coalesced_lower_bound = prev_upper_bound;
      }
      coalesced_upper_bound = buckets_[i].upper_bound;
      coalesced = true;
      continue;
    }

    if (coalesced) {
      absl::StrAppend(&ret, "Bucket ", coalesced_lower_bound, "-",
                      coalesced_upper_bound, " Count: 0\n");
      prev_upper_bound = coalesced_upper_bound;
    }

    coalesced = false;

    absl::StrAppend(&ret, "Bucket ", prev_upper_bound, "-",
                    buckets_[i].upper_bound, " Count: ", count, "\n");
    prev_upper_bound = buckets_[i].upper_bound;
  }

  if (coalesced) {
    absl::StrAppend(&ret, "Bucket ", coalesced_lower_bound, "-",
                    coalesced_upper_bound, " Count: 0\n");
  }

  return ret;
}

void CommStats::Dump() {
  // The offload interval is guaranteed to contain at least one
  // value if there has been at least one Isend or Irecv call,
  // which means the average should be valid.
  if (!request.offload_interval_bucketer.GetAvg().has_value()) {
    LOG(INFO) << "COMM STATS: NONE (comm was unused)";
    return;
  }

  LOG(INFO)
      << "COMM STATS:"
      << absl::StrCat(
             "\nrequest/offload_scheduled: ", request.offload_scheduled,
             "\nrequest/offload_completed: ", request.offload_completed,
             "\nrequest/offload_backlog_curr: ", request.GetOffloadBacklog(),
             "\nrequest/offload_backlog_peak: ", request.offload_backlog_peak,
             "\nrequest/offload_size_distribution:\n",
             request.offload_size_bucketer.Dump(),  // Already ends in newline
             "request/offload_duration_distribution:\n",
             request.offload_duration_bucketer.Dump(),
             "request/unpack_duration_distribution:\n",
             request.unpack_duration_bucketer.Dump(),
             "request/offload_complete_age_distribution:\n",
             request.offload_complete_age_bucketer.Dump(),
             "request/unpack_num_iovs_distribution:\n",
             request.unpack_num_iovs_bucketer.Dump(),
             "request/offload_interval_distribution:\n",
             request.offload_interval_bucketer.Dump());
}

}  // namespace fastrak

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_STATS_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_STATS_H_

#include <stdint.h>
#include <time.h>

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tcpdirect_plugin/fastrak_offload/params.h"

namespace fastrak {

// A timer that can be used both for implementing
// timeouts as well as determining elapsed time.
class Timer {
 public:
  Timer() : init_time_(GetMonotonicTs()) {}

  // Sets the timer initialization time to the current time.
  // Intended for cases where the object is reused.
  absl::Time Restart();

  // Gets the time duration elapsed since creation or Restart().
  absl::Duration GetElapsed();

  // Gets the time when the timer was created or restarted
  absl::Time GetStartTime() { return init_time_; }

  // Gets the current time.
  absl::Time GetCurrTime() { return GetMonotonicTs(); }

  // Checks the timeout status and returns error if timeout threshold
  // reached.
  absl::Status CheckTimeout(absl::Duration timeout);

 private:
  absl::Time GetMonotonicTs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return absl::TimeFromTimespec(ts);
  }

  absl::Time init_time_;
};

// Thread compatible.
class DistributionBucketer {
 public:
  explicit DistributionBucketer(double max_bucket, double scale = 1.0);
  std::vector<int64_t> GetBucketCounts();
  std::vector<double> GetBucketBounds();
  void Submit(double sample);
  std::string Dump();
  std::optional<double> GetMin();
  std::optional<double> GetMax();
  // Returns a coarse-grained average of all values submitted previously
  std::optional<double> GetAvg();

 private:
  struct bucket {
    double upper_bound;
    int64_t count = 0;
  };
  int64_t count_ = 0;
  double avg_ = 0;
  std::vector<bucket> buckets_;
  bool sample_submitted_ = false;
  double min_ = std::numeric_limits<double>::max();
  double max_ = std::numeric_limits<double>::min();
};

struct CommStats {
  // Max bucket ~10 milliseconds.
  static constexpr double kBucketerMaxXferDuration = 10000000.0;
  // No need to maintain several sub-microsecond buckets, so
  // start at 10 us.
  static constexpr double kBucketerXferDurationScale = 10000.0;
  static constexpr double kBucketerMaxXferSize = 2097152.0;
  // Maintain buckets starting at >= 4 KiB.
  static constexpr double kBucketerXferSizeScale = 4096.0;
  // Max bucket ~10 milliseconds.
  static constexpr double kBucketMaxUnpackDuration = 10000000.0;
  // No need to maintain several sub-microsecond buckets, so
  // start at 1 us.
  static constexpr double kBucketMaxUnpackDurationScale = 1000.0;
  // Age = difference between when DXS notifies DXS client of
  // completion and when NCCL calls test on the complete request.
  static constexpr double kBucketMaxOffloadCompleteAge = 50000.0;
  static constexpr double kBucketMaxOffloadCompleteAgeScale = 100.0;
  // Max number of iovs to unpack
  static constexpr double kBucketMaxNumIovs = 2048.0;
  static constexpr double kBucketNumIovsScale = 1.0;
  // Isend/Irecv call interval bucketer. This bucketer tracks
  // the time between subsequent Isend/Irecv calls.
  // Since Irecv can sometimes fail due to missing RX
  // metadata, logic is in place to discard samples/calls after
  // the initial Irecv call that have failed due to missing metadata.
  static constexpr double kBucketerMaxXferInterval = 10000000.0;
  static constexpr double kBucketerXferIntervalScale = 1000.0;

  static constexpr int kConnectionPrintThreshold = 5000;

  const std::string gpu_pci;
  const uint32_t idx;
  const bool send;

  struct {
    uint64_t offload_scheduled = 0;
    uint64_t offload_completed = 0;
    uint64_t offload_backlog_peak = 0;

    uint64_t GetOffloadBacklog() {
      return offload_scheduled - offload_completed;
    }
    class DistributionBucketer offload_duration_bucketer{
        kBucketerMaxXferDuration, kBucketerXferDurationScale};
    class DistributionBucketer offload_size_bucketer{kBucketerMaxXferSize,
                                                     kBucketerXferSizeScale};
    class DistributionBucketer unpack_duration_bucketer{
        kBucketMaxUnpackDuration, kBucketMaxUnpackDurationScale};
    class DistributionBucketer offload_complete_age_bucketer{
        kBucketMaxOffloadCompleteAge, kBucketMaxOffloadCompleteAgeScale};
    class DistributionBucketer unpack_num_iovs_bucketer{kBucketMaxNumIovs,
                                                        kBucketNumIovsScale};
    class DistributionBucketer offload_interval_bucketer{
        kBucketerMaxXferInterval, kBucketerXferIntervalScale};
  } request;

  void Dump();
};

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_STATS_H_

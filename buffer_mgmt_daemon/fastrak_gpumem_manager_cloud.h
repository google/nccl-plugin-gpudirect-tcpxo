/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_CLOUD_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_CLOUD_H_

#include <sys/stat.h>

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/clock_interface.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_host_interface.h"
#include "json/include/nlohmann/json.hpp"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"

namespace tcpdirect {

class FastrakGpumemManagerCloud : public FastrakGpumemManagerHostInterface {
 public:
  explicit FastrakGpumemManagerCloud(
      std::unique_ptr<tensorflow::serving::net_http::HTTPServerInterface>
          http_server,
      std::unique_ptr<ClockInterface> clock_override = nullptr);
  absl::Status Setup() override;

  void SetHealthStatus(HealthStatus status, absl::string_view message) override;

  ~FastrakGpumemManagerCloud() override;

 private:
  absl::Mutex mutex_;
  HealthStatus health_status_ ABSL_GUARDED_BY(mutex_) =
      HealthStatus::kInitializing;
  absl::Time start_time_ ABSL_GUARDED_BY(mutex_) = absl::InfinitePast();
  absl::Time init_done_time_ ABSL_GUARDED_BY(mutex_) = absl::InfinitePast();
  absl::Time fatal_error_time_ ABSL_GUARDED_BY(mutex_) = absl::InfinitePast();
  std::string status_message_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<tensorflow::serving::net_http::HTTPServerInterface>
      http_server_;
  std::unique_ptr<ClockInterface> clock_;
  // Returns a json object that represents the http response body
  nlohmann::json GetHealthCheckResponseJson()
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_CLOUD_H_

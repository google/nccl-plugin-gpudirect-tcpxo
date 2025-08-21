/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/fastrak_gpumem_manager_cloud.h"

#include <sys/stat.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/clock.h"
#include "buffer_mgmt_daemon/clock_interface.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_host_interface.h"
#include "json/include/nlohmann/json.hpp"
#include "tensorflow_serving/util/net_http/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"

namespace tcpdirect {

using ::nlohmann::json;

FastrakGpumemManagerCloud::FastrakGpumemManagerCloud(
    std::unique_ptr<tensorflow::serving::net_http::HTTPServerInterface>
        http_server,
    std::unique_ptr<ClockInterface> clock_override)
    : http_server_(std::move(http_server)), clock_(std::move(clock_override)) {
  if (clock_ == nullptr) {
    clock_ = std::make_unique<Clock>();
  }
  start_time_ = clock_->TimeNow();
  if (http_server_ != nullptr) {
    http_server_->RegisterRequestHandler(
        "/health",
        [this](tensorflow::serving::net_http::ServerRequestInterface* request) {
          request->OverwriteResponseHeader("Content-Type", "application/json");
          absl::MutexLock lock(&mutex_);
          auto resp_body_json = GetHealthCheckResponseJson();
          request->WriteResponseString(resp_body_json.dump());
          switch (health_status_) {
            case HealthStatus::kInitializing:
              request->ReplyWithStatus(
                  tensorflow::serving::net_http::HTTPStatusCode::SERVICE_UNAV);
              break;
            case HealthStatus::kHealthy:
              request->ReplyWithStatus(
                  tensorflow::serving::net_http::HTTPStatusCode::OK);
              break;
            case HealthStatus::kUnhealthy:
              request->ReplyWithStatus(
                  tensorflow::serving::net_http::HTTPStatusCode::ERROR);
              break;
          }
        },
        tensorflow::serving::net_http::RequestHandlerOptions());

    http_server_->StartAcceptingRequests();
  }
}

absl::Status FastrakGpumemManagerCloud::Setup() { return absl::OkStatus(); }

void FastrakGpumemManagerCloud::SetHealthStatus(HealthStatus status,
                                                absl::string_view message) {
  absl::MutexLock lock(&mutex_);
  switch (health_status_) {
    case HealthStatus::kInitializing:
      if (status == HealthStatus::kHealthy) {
        init_done_time_ = clock_->TimeNow();
        health_status_ = status;
        status_message_ = message;
      } else if (status == HealthStatus::kUnhealthy) {
        fatal_error_time_ = clock_->TimeNow();
        health_status_ = status;
        status_message_ = message;
      }
      break;
    case HealthStatus::kHealthy:
      if (status == HealthStatus::kUnhealthy) {
        fatal_error_time_ = clock_->TimeNow();
        health_status_ = status;
        status_message_ = message;
      }
      break;
    case HealthStatus::kUnhealthy:
      // We cannot transition from unhealthy to another state.
      break;
  }
}

FastrakGpumemManagerCloud::~FastrakGpumemManagerCloud() {
  if (http_server_ != nullptr) {
    http_server_->Terminate();
    http_server_->WaitForTermination();
  }
}

json FastrakGpumemManagerCloud::GetHealthCheckResponseJson() {
  json resp_body_json;
  resp_body_json["startup_time"] = absl::FormatTime(start_time_);
  switch (health_status_) {
    case HealthStatus::kInitializing:
      resp_body_json["status"] = "initializing";
      break;
    case HealthStatus::kHealthy:
      resp_body_json["status"] = "ok";
      resp_body_json["init_done_time"] = absl::FormatTime(init_done_time_);
      break;
    case HealthStatus::kUnhealthy:
      resp_body_json["status"] = "unhealthy";
      // If error happened during init time then init_done_time will not
      // be reported in the json response, because we simply haven't finished
      // initialization before an error happened.
      if (init_done_time_ != absl::InfinitePast()) {
        resp_body_json["init_done_time"] = absl::FormatTime(init_done_time_);
      }
      resp_body_json["fatal_error_time"] = absl::FormatTime(fatal_error_time_);
      resp_body_json["error_description"] = status_message_;
      break;
  }
  resp_body_json["current_time"] = absl::FormatTime(clock_->TimeNow());
  return resp_body_json;
}

}  // namespace tcpdirect

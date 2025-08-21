/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include <sys/stat.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/buffer_manager_version.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_base.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_cloud.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_startup.h"
#include "buffer_mgmt_daemon/oss/init.h"
#include "ecclesia/lib/http/server.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"

ABSL_FLAG(int, num_nics, 8, "Number of GPU NICs attached to the VM.");
ABSL_FLAG(
    std::string, nics_to_use, "",
    "Comma-separated list of GPU NICs to use, identified by their "
    "interface name. If unset, will find NICs from PCIe topology discovery.");
ABSL_FLAG(bool, use_gpu_mem, true, "Use GPU memory for registration.");
ABSL_FLAG(int, num_gpus_to_use, 0,
          "Number of GPUs to use. If unset, will use all available GPUs.");
ABSL_FLAG(int, http_port, 7332,
          "Port number to expose HTTP /health endpoint on. Set to 0 to disable "
          "HTTP server.");
ABSL_FLAG(int, http_num_threads, 2,
          "Number of threads to use for the HTTP server.");
ABSL_FLAG(std::string, http_bind_address, "127.0.0.1",
          "IP address to bind the HTTP server to. Set to empty to bind to all "
          "addresses.");
ABSL_FLAG(absl::Duration, wait_time_before_exit, absl::Seconds(5),
          "The amount of time to wait before RXDM exits.");
ABSL_FLAG(std::string, dmabuf_import_path, "",
          "DmaBuf import path. If empty, will use the default.");

int main(int argc, char* argv[]) {
  tcpdirect::InitRxdm(argc, argv);
  // Drop RWX privileges for all Unix Domain Sockets created later on
  umask(0);

  bool use_gpu_mem = absl::GetFlag(FLAGS_use_gpu_mem);
  int num_gpus_to_use = absl::GetFlag(FLAGS_num_gpus_to_use);

  int num_nics = absl::GetFlag(FLAGS_num_nics);
  LOG(INFO) << "Starting up buffer manager, version: " << tcpdirect::kRxdmMajor
            << "." << tcpdirect::kRxdmMinor << "." << tcpdirect::kRxdmPatch;
  absl::flat_hash_set<std::string> nics_set =
      tcpdirect::GetNICsToUse(&num_nics, absl::GetFlag(FLAGS_nics_to_use));

  std::string dmabuf_import_path = absl::GetFlag(FLAGS_dmabuf_import_path);

  std::unique_ptr<tensorflow::serving::net_http::HTTPServerInterface>
      http_server = nullptr;

  // Create http server if http_port is set to a valid value.
  if (absl::GetFlag(FLAGS_http_port) > 0) {
    auto options =
        std::make_unique<tensorflow::serving::net_http::ServerOptions>();
    options->AddPort(absl::GetFlag(FLAGS_http_port));
    options->SetExecutor(std::make_unique<ecclesia::RequestExecutor>(
        absl::GetFlag(FLAGS_http_num_threads)));
    if (!absl::GetFlag(FLAGS_http_bind_address).empty()) {
      options->AddIPAddress(absl::GetFlag(FLAGS_http_bind_address));
    }
    http_server =
        tensorflow::serving::net_http::CreateEvHTTPServer(std::move(options));

    if (http_server == nullptr) {
      LOG(ERROR) << "Failed to create health status HTTP server";
      return 1;
    }
  } else {
    LOG(INFO) << "HTTP health status server is disabled via --http_port flag";
  }

  auto server = std::make_unique<tcpdirect::FasTrakGpuMemManager>(
      num_nics, use_gpu_mem, dmabuf_import_path,
      nics_set.empty() ? std::nullopt : std::make_optional(nics_set),
      std::make_unique<tcpdirect::FastrakGpumemManagerCloud>(
          std::move(http_server)),
      num_gpus_to_use == 0 ? std::nullopt
                           : std::make_optional(num_gpus_to_use));

  return tcpdirect::RunCloudGpuMemManager(
      std::move(server), absl::GetFlag(FLAGS_wait_time_before_exit));
}

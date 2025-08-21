/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/functional/bind_front.h"
#include "absl/log/log.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "buffer_mgmt_daemon/a3_gpu_rxq_configurator.h"
#include "buffer_mgmt_daemon/buffer_manager_service_interface.h"
#include "buffer_mgmt_daemon/cuda_logging.h"
#include "buffer_mgmt_daemon/fastrak_gpu_mem_importer.h"
#include "buffer_mgmt_daemon/fastrak_gpu_nic_info.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_base.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_host_interface.h"
#include "buffer_mgmt_daemon/gpu_ip_server.h"
#include "buffer_mgmt_daemon/gpu_rxq_configurator_interface.h"
#include "cuda.h"
#include "dxs/client/control-command.pb.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/dxs-client.h"

ABSL_FLAG(
    bool, shutdown_when_all_clients_exited, false,
    "Monitor the connected client and shutdown when no client is connected");

ABSL_FLAG(bool, set_up_llcm, false, "Have RxDM set up Llcm.");

ABSL_FLAG(std::string, nic_metric_directory, "",
          "Directory to store NIC metrics");

ABSL_FLAG(bool, keep_eth0, false,
          "By default removes eth0 from rxq configuration.");

ABSL_RETIRED_FLAG(bool, alsologtostderr, false,
                  "This flag is retired and should not be set.");
ABSL_RETIRED_FLAG(std::string, uid, "",
                  "This flag is retired and should not be set.");

namespace {
constexpr absl::string_view kHealthCheckFileEnvVar = "HEALTH_CHECK_LOG_FILE";

std::atomic<bool> gShouldStop = false;

void sig_handler(int signum) {
  if (signum == SIGINT || signum == SIGTERM) {
    tcpdirect::StopGpuMemManager();
  }
}

absl::Status DxsClientsHealthCheck(
    absl::Span<const tcpdirect::FasTrakGpuNicInfo> nic_infos) {
  bool healthy = true;
  std::string error_message =
      "DXS client health check failed for the following NICs: \n";
  for (const auto& nic_info : nic_infos) {
    if (!nic_info.dxs_client->HealthCheck()) {
      absl::StrAppend(
          &error_message,
          absl::StrFormat(
              "\t\tDXS: Destination: %s; Source: %s; GPU PCI addr: %s; NIC "
              "PCI addr: %s; ifname: %s; \n",
              nic_info.dxs_ip, nic_info.ip_addr, nic_info.gpu_pci_addr,
              nic_info.nic_pci_addr, nic_info.ifname));
      healthy = false;
    }
  }
  if (!healthy) {
    return absl::InternalError(error_message);
  }
  return absl::OkStatus();
}

std::string JoinPath(absl::string_view path1, absl::string_view path2) {
  if (path1.empty()) return std::string(path2);
  if (path2.empty()) return std::string(path1);
  if (path1.back() == '/') {
    if (path2.front() == '/')
      return absl::StrCat(path1, absl::ClippedSubstr(path2, 1));
  } else if (path2.front() != '/') {
    return absl::StrCat(path1, "/", path2);
  }
  return absl::StrCat(path1, path2);
}

void LogPeriodicStats(absl::string_view nic_metric_directory,
                      absl::string_view nic_ifname,
                      const dxs::PeriodicStatsUpdate& stats) {
  auto stats_file_path =
      JoinPath(nic_metric_directory, absl::StrCat(nic_ifname, "_stats"));
  std::string stats_str =
      absl::StrCat(stats.goodput_rx_bytes(), ",", stats.goodput_tx_bytes());

  // Create a temporary file to write the stats to.
  std::string tmp_file_path_str = absl::StrCat(stats_file_path, ".XXXXXX");

  // mkstemp requires a non-const c-string.
  auto tmp_file_path = std::make_unique<char[]>(tmp_file_path_str.length() + 1);
  absl::SNPrintF(tmp_file_path.get(), tmp_file_path_str.length() + 1, "%s",
                 tmp_file_path_str);

  int fd = mkstemp(tmp_file_path.get());
  if (fd == -1) {
    PLOG(ERROR) << "Error creating temporary file " << tmp_file_path.get();
    return;
  }

  FILE* fp = fdopen(fd, "w");
  if (fp == nullptr) {
    PLOG(ERROR) << "Error opening file " << tmp_file_path.get();
    close(fd);
    unlink(tmp_file_path.get());
    return;
  }
  absl::FPrintF(fp, "%s", stats_str);
  fclose(fp);

  // Rename the temporary file to the final file path.
  int result = rename(tmp_file_path.get(), stats_file_path.c_str());
  if (result != 0) {
    PLOG(ERROR) << "Failed to rename temp file " << tmp_file_path.get()
                << " to " << stats_file_path;
    unlink(tmp_file_path.get());
  }
}

absl::Status GetNumGPUs(int num_gpus_to_use, int& num_gpus) {
  num_gpus = 0;
  if (!tcpdirect::CUCallSuccess(cuDeviceGetCount(&num_gpus))) {
    return absl::InternalError("Failed to get number of GPUs via CUDA.");
  }
  if (num_gpus_to_use > num_gpus) {
    return absl::InternalError(absl::StrFormat(
        "num_gpus_to_use %d is greater than the number of GPUs %d",
        num_gpus_to_use, num_gpus));
  }
  if (num_gpus_to_use > 0) {
    LOG(INFO) << "Using " << num_gpus_to_use << " of " << num_gpus << " GPUs";
    num_gpus = num_gpus_to_use;
  }
  return absl::OkStatus();
}

void WriteRxDMHealthyLog() {
  absl::string_view log_str = "Buffer manager initialization completed.";
  absl::PrintF("%s\n", log_str);
  fflush(stdout);
  LOG(INFO) << log_str;
  char* log_filename = getenv(kHealthCheckFileEnvVar.data());
  if (!log_filename) {
    return;
  }
  std::ofstream logFile(log_filename);
  if (!logFile.is_open()) {
    LOG(ERROR) << "Error: Unable to open file " << log_filename;
    return;
  }
  logFile << log_str << std::endl;
  logFile.close();
  LOG(INFO) << "Log written successfully to " << log_filename;
}

}  // namespace

namespace tcpdirect {

void FasTrakGpuMemManager::SetAndLogError(absl::string_view error_message) {
  LOG(ERROR) << error_message;
  host_->SetHealthStatus(HealthStatus::kUnhealthy, error_message);
}

bool FasTrakGpuMemManager::CheckStatus(const absl::Status& status) {
  if (!status.ok()) {
    SetAndLogError(status.message());
    return false;
  }
  return true;
}

int FasTrakGpuMemManager::Init() {
  // Collect GPU/NIC pair configurations
  LOG(INFO) << "Collecting GPU/NIC pair configurations ...";
  std::unique_ptr<tcpdirect::GpuRxqConfiguratorInterface> gpu_rxq_configurator;
  gpu_rxq_configurator = std::make_unique<tcpdirect::A3GpuRxqConfigurator>();
  std::vector<std::unique_ptr<tcpdirect::GpuRxqConfiguration>> gpu_rxq_configs;

  gpu_rxq_configurator->GetConfigurations(&gpu_rxq_configs);

  if (!use_gpu_mem_ && !absl::GetFlag(FLAGS_keep_eth0)) {
    // Exclude eth0, otherwise BouncerBufferMgrsHealthChecks will fail.
    gpu_rxq_configs.erase(
        std::remove_if(
            gpu_rxq_configs.begin(), gpu_rxq_configs.end(),
            [](const auto& config) { return config->ifname == "eth0"; }),
        gpu_rxq_configs.end());
  }

  if (nics_to_use_.has_value()) {
    gpu_rxq_configs.erase(
        std::remove_if(gpu_rxq_configs.begin(), gpu_rxq_configs.end(),
                       [this](const auto& config) {
                         return !nics_to_use_->contains(config->ifname);
                       }),
        gpu_rxq_configs.end());
  }

  for (auto& config : gpu_rxq_configs) {
    gpu_rxq_configs_.emplace_back(std::move(*config));
  }

  // Host environment specific initialization.
  LOG(INFO) << "Starting host environment specific initialization";
  if (!CheckStatus(host_->Setup())) {
    return 1;
  }

  LOG(INFO) << "RxDM initialization finishes, will start running soon...";
  absl::FlushLogSinks();
  return 0;
}

int FasTrakGpuMemManager::Run() {
  if (use_gpu_mem_) {
    LOG(INFO) << "Initializing CUDA";
    if (!CUCallSuccess(cuInit(0))) {
      SetAndLogError("CUDA initialization failed");
      return 1;
    }

    int num_gpus = 0;
    if (!CheckStatus(GetNumGPUs(num_gpus_override_.value_or(0), num_gpus))) {
      return 1;
    }

    if (gpu_rxq_configs_.size() != static_cast<size_t>(num_gpus)) {
      SetAndLogError(absl::StrFormat(
          "Number of GPUs detected in the PCI tree %d is not equal to the "
          "actual number of GPUs reported by CUDA %d.",
          gpu_rxq_configs_.size(), num_gpus));
      return 1;
    }
  }

  // Only needed for A3-Mega machines: Enforce 1 GPU per NIC. This breaks bounce
  // buffer management but otherwise works.
  absl::flat_hash_set<std::string> nic_ips;
  std::string nic_metric_directory = absl::GetFlag(FLAGS_nic_metric_directory);
  for (auto& config : gpu_rxq_configs_) {
    auto [it, inserted] = nic_ips.insert(config.ip_addr);
    if (!inserted) continue;
    tcpdirect::FasTrakGpuNicInfo info(config);
    LOG(INFO) << "Creating DXS client: " << info.ip_addr << " " << info.dxs_ip
              << " " << info.dxs_port;

    dxs::BufferManagerOptions options = {
        .source_addr = info.ip_addr,
        .source_port = std::to_string(dxs::kBufferManagerSourcePort),
        .dest_addr_override = std::make_optional(info.dxs_ip),
        .dest_port = info.dxs_port};
    if (!nic_metric_directory.empty()) {
      LOG(INFO) << "Recording GPU traffic stats to directory: "
                << nic_metric_directory;
      options.periodic_stats_options = dxs::PeriodicStatsOptions{
          .handler = absl::bind_front(&LogPeriodicStats, nic_metric_directory,
                                      info.ifname)};
    }

    auto dxs_client = dxs::BufferManager::Create(options);
    if (!dxs_client.ok()) {
      SetAndLogError(absl::StrFormat(
          "Failed to initialize DXS client on %s with status %s", info.ip_addr,
          dxs_client.status().message()));
      StopGpuMemManager();
    }
    info.dxs_client = *std::move(dxs_client);
    nic_infos_.emplace_back(std::move(info));
  }
  if (nic_infos_.size() != num_nics_) {
    SetAndLogError(absl::StrFormat(
        "Number of NICs detected (%d) is not equal to the actual number of "
        "NICs specified (%d).",
        nic_infos_.size(), num_nics_));
    return 1;
  }

  LOG(INFO) << "Environment-related initialization completed.";

  // Initialize all interfaces needed
  if (use_gpu_mem_) {
    LOG(INFO) << "Creating GpuIpServer";
    gpu_ip_server_ = std::make_unique<GpuIpServer>(gpu_rxq_configs_);
  } else {
    LOG(INFO) << "Not using GPU mem, skipping GpuIpServer creation";
  }

  std::optional<std::string> dmabuf_import_path_opt =
      dmabuf_import_path_.empty() ? std::nullopt
                                  : std::make_optional(dmabuf_import_path_);
  gpu_mem_importer_ = std::make_unique<FasTrakGpuMemImporter>(
      nic_infos_,
      absl::GetFlag(FLAGS_shutdown_when_all_clients_exited)
          ? [] { StopGpuMemManager(); }
          : [] {},
      dmabuf_import_path_opt);

  LOG(INFO) << "Creating FasTrakGpuMemImporter";
  LOG(INFO) << "Initializing gpu_mem_importer_";
  if (!CheckStatus(gpu_mem_importer_->Initialize())) return 1;

  LOG(INFO) << "Starting gpu_mem_importer_";
  if (!CheckStatus(gpu_mem_importer_->Start())) return 1;

  // Must be last - the ability to connect to the GPU IP
  // server indicates that RxDM is fully initialized.
  if (use_gpu_mem_) {
    LOG(INFO) << "Initializing gpu_ip_server";
    if (!CheckStatus(gpu_ip_server_->Initialize())) return 1;
    LOG(INFO) << "Starting gpu_ip_server_";
    if (!CheckStatus(gpu_ip_server_->Start())) return 1;
  } else {
    LOG(INFO) << "Not using GPU mem, skipping gpu_ip_server";
  }

  WriteRxDMHealthyLog();
  host_->SetHealthStatus(HealthStatus::kHealthy, "Initialization completed.");

  // Setup global flags and handlers
  gShouldStop.store(false);
  signal(SIGINT, sig_handler);
  signal(SIGTERM, sig_handler);
  while (!gShouldStop.load()) {
    absl::SleepFor(absl::Seconds(1));
    auto dxs_health_check_status = DxsClientsHealthCheck(nic_infos_);
    if (!dxs_health_check_status.ok()) {
      LOG(ERROR) << dxs_health_check_status;
      host_->SetHealthStatus(HealthStatus::kUnhealthy,
                             dxs_health_check_status.message());
      StopGpuMemManager();
    }
  }

  // Stopping the servers.
  LOG(INFO) << "Program terminates, stopping the servers ...";
  return 0;
}

absl::flat_hash_set<std::string> GetNICsToUse(int* num_nics,
                                              std::string nics_to_use) {
  absl::flat_hash_set<std::string> nics_set;
  if (!nics_to_use.empty()) {
    LOG(INFO) << "Using NICs " << nics_to_use;
    auto nics = absl::StrSplit(nics_to_use, ',');
    nics_set.insert(nics.begin(), nics.end());
    *num_nics = nics_set.size();
  }
  return nics_set;
}

void StopGpuMemManager() { gShouldStop.store(true, std::memory_order_release); }

}  // namespace tcpdirect

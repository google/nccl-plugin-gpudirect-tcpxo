/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_BASE_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_BASE_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "buffer_mgmt_daemon/buffer_manager_service_interface.h"
#include "buffer_mgmt_daemon/fastrak_gpu_nic_info.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_host_interface.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_interface.h"
#include "buffer_mgmt_daemon/gpu_ip_server.h"
#include "buffer_mgmt_daemon/gpu_rxq_configurator_interface.h"

namespace tcpdirect {

class FasTrakGpuMemManager : public FasTrakGpuMemManagerInterface {
 public:
  // If num_gpus_override is set, it will be used instead of the actual number
  // of GPUs on the machine.
  explicit FasTrakGpuMemManager(
      bool use_gpu_mem, std::string dmabuf_import_path,
      std::optional<absl::flat_hash_set<std::string>> nics_to_use,
      std::unique_ptr<FastrakGpumemManagerHostInterface> host,
      std::optional<int> num_gpus_override = std::nullopt)
      : host_(std::move(host)),
        use_gpu_mem_(use_gpu_mem),
        dmabuf_import_path_(dmabuf_import_path),
        nics_to_use_(nics_to_use),
        num_gpus_override_(num_gpus_override) {}

  ~FasTrakGpuMemManager() override = default;

  int Init() override;
  int Run() override;

 private:
  std::vector<tcpdirect::GpuRxqConfiguration> gpu_rxq_configs_;
  std::vector<tcpdirect::FasTrakGpuNicInfo> nic_infos_;
  std::unique_ptr<BufferManagerServiceInterface> gpu_mem_importer_;
  std::unique_ptr<GpuIpServer> gpu_ip_server_;
  std::unique_ptr<FastrakGpumemManagerHostInterface> host_;
  bool use_gpu_mem_;
  std::string dmabuf_import_path_;
  std::optional<absl::flat_hash_set<std::string>> nics_to_use_;
  std::optional<int> num_gpus_override_;
  // Check the status of critical operations and set health status if we
  // encounter an error.
  bool CheckStatus(const absl::Status& status);
  // Logs a fatal error and sets the health status to unhealthy, with the
  // corresponding error message.
  void SetAndLogError(absl::string_view error_message);
};

// Stop the GPU Mem Manager.
void StopGpuMemManager();

// Obtain the set of NICs to use if optional flag is set.
absl::flat_hash_set<std::string> GetNICsToUse(std::string nics_to_use);

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_BASE_H_

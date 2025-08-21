/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/nic_client_router.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/dxs-client.h"
#include "dxs/client/oss/status_macros.h"
#include "tcpdirect_plugin/fastrak_offload/nic_client_router_interface.h"
#include "tcpdirect_plugin/fastrak_offload/params.h"

namespace fastrak {

// Removes leading and trailing whitespaces from a string.
void trimString(std::string& str) {
  if (str.empty()) {
    return;
  }
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
}

NicClientRouter::NicClientRouter(
    absl::flat_hash_map<std::string, PerNicClient> all_clients)
    : ip_addr_to_client_(std::move(all_clients)) {}

absl::StatusOr<dxs::DxsClientInterface* absl_nonnull>
NicClientRouter::GetDxsClient(absl::string_view dxs_addr) {
  absl::MutexLock lock(&mu_);
  ASSIGN_OR_RETURN(PerNicClient * nic_client, GetOrInitClient(dxs_addr));
  return nic_client->dxs_client.get();
}

absl::StatusOr<tcpdirect::BufferManagerClientInterface* absl_nonnull>
NicClientRouter::GetBufferManagerClient(absl::string_view dxs_addr) {
  absl::MutexLock lock(&mu_);
  ASSIGN_OR_RETURN(PerNicClient * nic_client, GetOrInitClient(dxs_addr));
  return nic_client->buffer_manager_client.get();
}

absl::StatusOr<NicClientRouter::PerNicClient* absl_nonnull>
NicClientRouter::GetOrInitClient(absl::string_view dxs_addr) {
  auto it = ip_addr_to_client_.find(dxs_addr);
  if (it != ip_addr_to_client_.end()) {
    return &it->second;
  }

  std::string dxs_connect_addr = dxs::kDefaultDxsAddr;

  std::string llcm_device_directory = "";
  if (kFastrakUseLlcm) {
    // `NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY` specifies the search path for LLCM
    // PCIe devices. Typically this would be sysfs, but in situations where
    // sysfs is mounted read-only, these devices may be exposed in a different
    // location.
    //
    // A trailing slash is not permitted.
    char* device_directory = getenv("NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY");
    std::string device_directory_str =
        device_directory ? std::string(device_directory) : "";
    trimString(device_directory_str);
    if (!device_directory_str.empty()) {
      llcm_device_directory = device_directory_str;
    } else {
      llcm_device_directory = dxs::kLlcmDeviceDirectory;
    }
  }

  std::string nic_addr(dxs_addr);
  ASSIGN_OR_RETURN(auto buffer_manager_client,
                   tcpdirect::BufferManagerClient::Create(nic_addr));
  ASSIGN_OR_RETURN(
      auto dxs_client,
      dxs::DxsClient::Create(std::string(dxs_addr), dxs_connect_addr,
                             dxs::kDefaultDxsPort, "0", kFastrakUseLlcm,
                             llcm_device_directory, kFastrakCloseSendOnDone));
  auto [iter, inserted] = ip_addr_to_client_.try_emplace(
      dxs_addr,
      PerNicClient{.dxs_client = std::move(dxs_client),
                   .buffer_manager_client = std::move(buffer_manager_client)});
  return &(iter->second);
}

NicClientRouterInterface*& NicClientRouterSlot() {
  static NicClientRouterInterface* nic_client_router = new NicClientRouter();
  return nic_client_router;
}

NicClientRouterInterface& GetNicClientRouter() {
  return *NicClientRouterSlot();
}

NicClientRouterInterface& TestonlyExchangeNicClientRouter(
    NicClientRouterInterface& other) {
  return *std::exchange(NicClientRouterSlot(), &other);
}

}  // namespace fastrak

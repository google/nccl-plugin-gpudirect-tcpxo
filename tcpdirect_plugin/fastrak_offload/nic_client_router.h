/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_DXS_CLIENT_ROUTER_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_DXS_CLIENT_ROUTER_H_

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "tcpdirect_plugin/fastrak_offload/nic_client_router_interface.h"

namespace fastrak {

class NicClientRouter : public NicClientRouterInterface {
 public:
  struct PerNicClient {
    absl_nonnull std::unique_ptr<dxs::DxsClientInterface> dxs_client;
    absl_nonnull std::unique_ptr<tcpdirect::BufferManagerClientInterface>
        buffer_manager_client;
  };

  NicClientRouter() = default;
  explicit NicClientRouter(
      absl::flat_hash_map<std::string, PerNicClient> clients);

  absl::StatusOr<dxs::DxsClientInterface* absl_nonnull> GetDxsClient(
      absl::string_view dxs_addr) override;

  absl::StatusOr<tcpdirect::BufferManagerClientInterface* absl_nonnull>
  GetBufferManagerClient(absl::string_view dxs_addr) override;

  void Shutdown() override {
    absl::MutexLock lock(&mu_);
    for (auto& [id, client] : ip_addr_to_client_) {
      absl::Status result =
          client.dxs_client->Shutdown(/*timeout=*/absl::Seconds(1));
      if (!result.ok()) {
        // Log if there is any error, continue the shutdown.
        LOG(ERROR) << "Failed to shutdown DXS client: " << result;
      }
    }
  }

 private:
  absl::StatusOr<PerNicClient* absl_nonnull> GetOrInitClient(
      absl::string_view dxs_addr) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, PerNicClient> ip_addr_to_client_
      ABSL_GUARDED_BY(mu_);
};

NicClientRouterInterface& GetNicClientRouter();

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_DXS_CLIENT_ROUTER_H_

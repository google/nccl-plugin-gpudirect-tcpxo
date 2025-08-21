/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_NIC_CLIENT_ROUTER_INTERFACE_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_NIC_CLIENT_ROUTER_INTERFACE_H_

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "dxs/client/dxs-client-interface.h"

namespace fastrak {

// An interface for picking a DXS client to use based on the NIC used.
// Thread safe
class NicClientRouterInterface {
 public:
  NicClientRouterInterface() = default;
  virtual ~NicClientRouterInterface() = default;
  NicClientRouterInterface(const NicClientRouterInterface&) = delete;
  NicClientRouterInterface& operator=(const NicClientRouterInterface&) = delete;

  // Returns the DXS Client for the given DXS addr,
  // or nullptr if the DXS addr is invalid.
  virtual absl::StatusOr<dxs::DxsClientInterface* absl_nonnull> GetDxsClient(
      absl::string_view dxs_addr) = 0;

  // Returns the BufferManagerClient for the given DXS addr,
  // or nullptr if the DXS addr is invalid.
  virtual absl::StatusOr<tcpdirect::BufferManagerClientInterface* absl_nonnull>
  GetBufferManagerClient(absl::string_view dxs_addr) = 0;

  // Attempt to disconnect all Dxs clients from the server.
  virtual void Shutdown() = 0;
};

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_NIC_CLIENT_ROUTER_INTERFACE_H_

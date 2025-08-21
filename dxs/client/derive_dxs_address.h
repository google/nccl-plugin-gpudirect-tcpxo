/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_DERIVE_DXS_ADDRESS_H_
#define DXS_CLIENT_DERIVE_DXS_ADDRESS_H_

#include <netinet/in.h>

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "dxs/client/control-command.h"

namespace dxs {

// Pack the IP address and port into a WireSocketAddr.
absl::StatusOr<WireSocketAddr> PackIpAddress(const std::string& dxs_addr,
                                             uint16_t port);
absl::StatusOr<std::string> UnpackIpAddress(const WireSocketAddr& addr);

WireSocketAddr FromIp(in_addr addr, uint16_t port);
WireSocketAddr FromIp(in6_addr addr, uint16_t port);

}  // namespace dxs

#endif  // DXS_CLIENT_DERIVE_DXS_ADDRESS_H_

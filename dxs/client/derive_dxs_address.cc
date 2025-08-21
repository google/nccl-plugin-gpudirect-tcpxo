/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/derive_dxs_address.h"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "dxs/client/control-command.h"

namespace dxs {

absl::StatusOr<WireSocketAddr> PackIpAddress(const std::string& dxs_addr,
                                             uint16_t port) {
  in_addr addr;
  if (inet_pton(AF_INET, dxs_addr.c_str(), &addr) == 1) {
    return FromIp(addr, port);
  }
  in6_addr addr6;
  if (inet_pton(AF_INET6, dxs_addr.c_str(), &addr6) == 1) {
    return FromIp(addr6, port);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "inet_pton() failed to convert IP address to binary: ", dxs_addr));
}

absl::StatusOr<std::string> UnpackIpAddress(const WireSocketAddr& addr) {
  if (addr.is_ipv6) {
    char ip_str[INET6_ADDRSTRLEN + 1] = {};
    if (inet_ntop(AF_INET6, &addr.addr, ip_str, sizeof(ip_str)) == nullptr) {
      return absl::InternalError(absl::StrCat(
          "Failed to convert IPv6 address to string: ", strerror(errno)));
    }
    return std::string(ip_str);
  }
  char ip_str[INET_ADDRSTRLEN + 1] = {};
  if (inet_ntop(AF_INET, &addr.addr, ip_str, sizeof(ip_str)) == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Failed to convert IPv4 address to string: ", strerror(errno)));
  }
  return std::string(ip_str);
}

WireSocketAddr FromIp(in_addr addr, uint16_t port) {
  WireSocketAddr wire{.is_ipv6 = false, .port = port};
  memcpy(wire.addr, &addr, sizeof(addr));
  return wire;
}

WireSocketAddr FromIp(in6_addr addr, uint16_t port) {
  WireSocketAddr wire{.is_ipv6 = true, .port = port};
  memcpy(wire.addr, &addr, sizeof(addr));
  return wire;
}

}  // namespace dxs

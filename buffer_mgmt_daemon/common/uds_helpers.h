/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_COMMON_UDS_HELPERS_H_
#define BUFFER_MGMT_DAEMON_COMMON_UDS_HELPERS_H_

#include <sys/un.h>

#include <string>

#include "absl/flags/declare.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tcpdirect {

// Construct a sockaddr_un for an abstract unix domain socket, or a file-based
// UDS socket in Google internal environments.
//
//
absl::StatusOr<sockaddr_un> UdsSockaddr(const std::string& path);

std::string BufOpUdsPath(absl::string_view server_ip_addr);
std::string NicIpUdsPath(absl::string_view gpu_pci_sanitized);
std::string NicMappingUdsPath();

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_COMMON_UDS_HELPERS_H_

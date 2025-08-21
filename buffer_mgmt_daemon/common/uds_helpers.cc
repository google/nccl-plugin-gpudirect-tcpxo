/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/common/uds_helpers.h"

#include <sys/socket.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

ABSL_FLAG(
    std::string, file_based_uds_path_prefix, "",
    "If not empty, use this prefix to create file-based Unix Domain "
    "Socket path (e.g. /tmp). Users need to make sure this path exists and is "
    "consistent between RxDM and workload.");
ABSL_FLAG(bool, create_uds_prefix_folder, false,
          "Create the UDS prefix folder if it doesn't exist.");

namespace tcpdirect {

namespace {

constexpr absl::string_view kFileBasedUdsPathPrefixEnvVar =
    "FILE_BASED_UDS_PATH_PREFIX";

absl::StatusOr<sockaddr_un> AbstractUdsSockaddr(absl::string_view path) {
  // 2 null bytes: abstract UDS flag and trailing null byte
  if (path.size() >= sizeof(sockaddr_un::sun_path) - 2) {
    return absl::InvalidArgumentError(
        "File path for abstract domain socket is too long.");
  }
  sockaddr_un server_addr;
  server_addr.sun_family = AF_UNIX;
  memset(server_addr.sun_path, '\0', sizeof(server_addr.sun_path));
  // First byte null for abstract UDS.
  absl::SNPrintF(server_addr.sun_path + 1, sizeof(server_addr.sun_path) - 1,
                 "%s", path);
  return server_addr;
}

absl::StatusOr<sockaddr_un> FileBasedUdsSockaddr(absl::string_view path) {
  // trailing null byte
  if (path.size() >= sizeof(sockaddr_un::sun_path) - 1) {
    return absl::InvalidArgumentError(
        "File path for abstract domain socket is too long.");
  }
  sockaddr_un server_addr;
  server_addr.sun_family = AF_UNIX;
  memset(server_addr.sun_path, '\0', sizeof(server_addr.sun_path));
  absl::SNPrintF(server_addr.sun_path, sizeof(server_addr.sun_path), "%s",
                 path);
  return server_addr;
}

// Get the file-based UDS path prefix from the flag or the environment
// variable.
std::string GetFileBasedUdsPathPrefix() {
  const std::string file_based_uds_path_prefix =
      absl::GetFlag(FLAGS_file_based_uds_path_prefix);
  if (!file_based_uds_path_prefix.empty()) {
    return file_based_uds_path_prefix;
  }

  const char* env_var = std::getenv(kFileBasedUdsPathPrefixEnvVar.data());
  return env_var == nullptr ? "" : env_var;
}

}  // namespace

absl::StatusOr<sockaddr_un> UdsSockaddr(const std::string& path) {
  if (path.empty()) {
    return absl::InvalidArgumentError("Missing file path to domain socket.");
  }

  const std::string file_based_uds_path_prefix = GetFileBasedUdsPathPrefix();
  if (!file_based_uds_path_prefix.empty()) {
    LOG(INFO) << "Using file-based UDS address specified by user: "
              << file_based_uds_path_prefix << "/" << path;
    return FileBasedUdsSockaddr(
        absl::StrCat(file_based_uds_path_prefix, "/", path));
  }

  LOG(INFO) << "Using abstract UDS address: " << path;
  return AbstractUdsSockaddr(path);
}

std::string BufOpUdsPath(absl::string_view server_ip_addr) {
  return absl::StrCat("rxdm_uds_buf_op:", server_ip_addr);
}
std::string NicIpUdsPath(absl::string_view gpu_pci_sanitized) {
  return absl::StrCat("rxdm_uds_nic_ip:", gpu_pci_sanitized);
}
std::string NicMappingUdsPath() { return "rxdm_uds_nic_mapping"; }

}  // namespace tcpdirect

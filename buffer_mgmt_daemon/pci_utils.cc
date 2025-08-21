/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/pci_utils.h"

#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <linux/limits.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/types.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace tcpdirect {

#define PCI_ID_LEN 6

constexpr char kSysfsPciDevicesPath[] = "/sys/bus/pci/devices";

int parse_pci_addr(const char* pci_addr, uint16_t* domain, uint16_t* bus,
                   uint16_t* device, uint16_t* function) {
  uint16_t tmp_domain, tmp_bus, tmp_device, tmp_function;
  if (sscanf(pci_addr, "%hx:%hx:%hx.%hx", &tmp_domain, &tmp_bus, &tmp_device,
             &tmp_function) != 4)
    return -1;
  *domain = tmp_domain;
  *bus = tmp_bus;
  *device = tmp_device;
  *function = tmp_function;
  return 0;
}

// Reads up to PCI_ID_LEN bytes of data from the specified `path`
std::string read_pci_id(absl::string_view path) {
  int fd = open(path.data(), O_RDONLY);
  if (fd < 0) {
    return "";
  }
  absl::Cleanup close_fd = [fd] { close(fd); };

  char pci_id[PCI_ID_LEN + 1] = {0};
  if (read(fd, pci_id, PCI_ID_LEN) < 0) {
    return "";
  }
  return std::string(pci_id);
}

int list_vendor_devices(const char* parent_dir_path,
                        std::vector<std::string>& candidates,
                        absl::string_view vendor_id,
                        std::optional<absl::string_view> device_id) {
  DIR* root_dir = opendir(parent_dir_path);
  if (root_dir == nullptr) {
    LOG(ERROR) << absl::StrFormat(
        "Failed to open parent directory [%s]. Error: %s", parent_dir_path,
        strerror(errno));
    return -1;
  }
  absl::Cleanup close_root_dir = [root_dir] { closedir(root_dir); };

  struct dirent* dir_entry;
  char subdir_path[PATH_MAX];
  uint16_t domain, bus, device, function;
  while ((dir_entry = readdir(root_dir)) != nullptr) {
    if (parse_pci_addr(dir_entry->d_name, &domain, &bus, &device, &function)) {
      continue;
    }
    auto vendor_str = read_pci_id(
        absl::StrFormat("%s/%s/vendor", parent_dir_path, dir_entry->d_name));
    auto device_str = read_pci_id(
        absl::StrFormat("%s/%s/device", parent_dir_path, dir_entry->d_name));
    if (vendor_str == vendor_id) {
      // Either we don't care about the device id or the device id matches
      if (!device_id.has_value() || device_str == device_id) {
        LOG(INFO) << "Match: PCI address " << dir_entry->d_name;
        candidates.emplace_back(dir_entry->d_name);
      }
    }
    snprintf(subdir_path, PATH_MAX, "%s/%s", parent_dir_path,
             dir_entry->d_name);
    // This is not a leaf node, continue our search
    list_vendor_devices(subdir_path, candidates, vendor_id, device_id);
  }
  return 0;
}

std::string get_ip_from_sockaddr(struct sockaddr* sa) {
  /* We couldn't get non-IPV4 and non-IPV6 addresses here */
  char host[HOST_MAX_LEN];
  static_assert(
      HOST_MAX_LEN >= INET6_ADDRSTRLEN && HOST_MAX_LEN >= INET_ADDRSTRLEN,
      "Need to reserve enough space for the IP address");

  switch (sa->sa_family) {
    case AF_INET: {
      struct sockaddr_in* addr_in = (struct sockaddr_in*)sa;
      if (!inet_ntop(AF_INET, &(addr_in->sin_addr), host, HOST_MAX_LEN)) {
        PLOG(ERROR) << "Failed to get IPv4 address from sockaddr";
        return "";
      } else {
        return std::string(host);
      }
    }
    case AF_INET6: {
      struct sockaddr_in6* addr_in6 = (struct sockaddr_in6*)sa;
      if (!inet_ntop(AF_INET6, &(addr_in6->sin6_addr), host, HOST_MAX_LEN)) {
        PLOG(ERROR) << "Failed to get IPv6 address from sockaddr";
        return "";
      } else {
        return std::string(host);
      }
    }
  }
  return "";
}

}  // namespace tcpdirect

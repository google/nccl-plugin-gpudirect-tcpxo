/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/a3_gpu_rxq_configurator.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <ifaddrs.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "buffer_mgmt_daemon/gpu_rxq_configurator_interface.h"
#include "buffer_mgmt_daemon/pci_utils.h"

#define MAX_HOPS 4
ABSL_FLAG(
    int, num_hops, 2,
    "Number of hops to the PCIE switch shared by the 2 GPUs and the NIC(s).");

namespace tcpdirect {

void A3GpuRxqConfigurator::DiscoverAllNics() {
  struct ifaddrs* all_ifs = nullptr;
  if (getifaddrs(&all_ifs) != 0 || all_ifs == nullptr) {
    LOG(ERROR) << "Failed to retrieve network ifs, error: " << strerror(errno);
    return;
  }
  struct ifaddrs* head = all_ifs;
  int num_hops = std::min(absl::GetFlag(FLAGS_num_hops), MAX_HOPS);
  do {
    // Skip interfaces that do not have a valid ifa_addr
    if (head->ifa_addr == nullptr) {
      continue;
    }
    // Skip non-IPV4 and non-IPV6 interfaces
    if (head->ifa_addr->sa_family != AF_INET &&
        head->ifa_addr->sa_family != AF_INET6) {
      continue;
    }
    char if_sysfs_path[PATH_MAX] = {0};
    snprintf(if_sysfs_path, PATH_MAX, "/sys/class/net/%s/device/",
             head->ifa_name);
    char if_sysfs_realpath[PATH_MAX] = {0};
    // Only pick interfaces that has an actual PCI device associated
    if (realpath(if_sysfs_path, if_sysfs_realpath) == nullptr) continue;
    if (strstr(if_sysfs_realpath, "/virtual")) {
      LOG(INFO) << head->ifa_name << " is a virtual device, skipping.";
      continue;
    }
    // We might overwrite these per NIC, but it should be fine since it should
    // be the same for all NICs on the same machine
    nic_vendor_id_ = read_pci_id(absl::StrFormat("%s/vendor", if_sysfs_path));
    nic_device_id_ = read_pci_id(absl::StrFormat("%s/device", if_sysfs_path));
    int last_char_idx = strlen(if_sysfs_realpath) - 1;
    if (if_sysfs_realpath[last_char_idx] == '/')
      if_sysfs_realpath[last_char_idx] = '\0';
    int path_length = 0;
    for (auto i = 0u; i < strlen(if_sysfs_realpath); i++) {
      if (if_sysfs_realpath[i] == '/') ++path_length;
    }
    char* pci_addr = nullptr;
    for (int i = 0; i < num_hops; i++) {
      char* slash = strrchr(if_sysfs_realpath, '/');
      /* First delimiter gives us the pci address*/
      if (i == 0) pci_addr = slash + 1;
      *slash = '\0';
    }
    uint16_t temp_domain, temp_bus, temp_device, temp_function;
    /* Not a valid PCI address */
    if (parse_pci_addr(pci_addr, &temp_domain, &temp_bus, &temp_device,
                       &temp_function))
      continue;
    std::string if_ip_addr = get_ip_from_sockaddr(head->ifa_addr);
    // Error parsing interface address.
    if (if_ip_addr.empty()) continue;
    auto [it, inserted] = netdev_pci_to_netdev_.try_emplace(
        absl::AsciiStrToLower(pci_addr), head->ifa_name, if_ip_addr);
    if (!inserted) {
      continue;
    }
    LOG(INFO) << "PCI addr for net if " << head->ifa_name << ": " << pci_addr;
    LOG(INFO) << "Root dir: " << if_sysfs_realpath;
    parent_switches_.emplace(if_sysfs_realpath);
  } while ((head = head->ifa_next) != nullptr);
  freeifaddrs(all_ifs);
}

// Sort GPUs and NICs under the same switch and pair them in ascending order
void A3GpuRxqConfigurator::GetConfigurations(
    std::vector<std::unique_ptr<GpuRxqConfiguration>>* configurations) {
  DiscoverAllNics();
  for (const auto& parent_switch : parent_switches_) {
    std::vector<std::string> gpus;
    std::vector<std::string> nics;

    if (list_vendor_devices(parent_switch.c_str(), gpus, kNvidiaVendorId,
                            kH100DeviceId) < 0) {
      LOG(WARNING) << "Failed to list GPUs under " << parent_switch;
      continue;
    }
    // Don't use hardcoded value for NICs as gVNIC and physical E2000 have
    // different PCI vendor/device ids.
    if (list_vendor_devices(parent_switch.c_str(), nics, nic_vendor_id_,
                            nic_device_id_) < 0) {
      LOG(WARNING) << "Failed to list NICs under " << parent_switch;
      continue;
    }
    if (nics.empty() || gpus.empty()) {
      continue;
    }

    // Sort the PCI BDFs in ascending order
    std::sort(gpus.begin(), gpus.end());
    std::sort(nics.begin(), nics.end());
    int gpus_per_nic = gpus.size() / nics.size();
    for (auto i = 0u; i < gpus.size(); i++) {
      int nic_idx = i / gpus_per_nic;
      auto netdev_pci = absl::AsciiStrToLower(nics[nic_idx]);
      auto it = netdev_pci_to_netdev_.find(netdev_pci);
      if (it == netdev_pci_to_netdev_.end()) {
        LOG(WARNING) << "Unknown NIC PCI: " << netdev_pci;
        continue;
      }
      auto& [netdev_name, ip_addr] = it->second;
      configurations->emplace_back(new GpuRxqConfiguration{
          absl::AsciiStrToLower(gpus[i]),
          netdev_pci,
          netdev_name,
          ip_addr,
      });
      LOG(INFO) << "GpuRxqConfigurator: Use netdev " << netdev_name
                << " for GPU PCI " << gpus[i] << ", NIC PCI " << netdev_pci
                << ", NIC IP " << ip_addr;
    }
    GpuRxqConfigurationComparator comparator;
    std::sort(configurations->begin(), configurations->end(), comparator);
  }
}
}  // namespace tcpdirect

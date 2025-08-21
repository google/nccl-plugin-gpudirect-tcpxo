/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_A3_GPU_RXQ_CONFIGURATOR_H_
#define BUFFER_MGMT_DAEMON_A3_GPU_RXQ_CONFIGURATOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "buffer_mgmt_daemon/gpu_rxq_configurator_interface.h"

namespace tcpdirect {

struct GpuRxqConfigurationComparator {
  bool operator()(const std::unique_ptr<GpuRxqConfiguration>& a,
                  const std::unique_ptr<GpuRxqConfiguration>& b) const {
    /* If it is the same netdev, compare by GPU PCI addr */
    if (a->nic_pci_addr == b->nic_pci_addr) {
      return a->gpu_pci_addr < b->gpu_pci_addr;
    }
    return a->nic_pci_addr < b->nic_pci_addr;
  }
};

class A3GpuRxqConfigurator : public GpuRxqConfiguratorInterface {
 public:
  void GetConfigurations(std::vector<std::unique_ptr<GpuRxqConfiguration>>*
                             configurations) override;

 private:
  void DiscoverAllNics();
  // Key: netdev pci, value: <netdev name, netdev ip addr>
  absl::flat_hash_map<std::string, std::pair<std::string, std::string>>
      netdev_pci_to_netdev_;
  absl::flat_hash_set<std::string> parent_switches_;
  std::string nic_vendor_id_;
  std::string nic_device_id_;
};
}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_A3_GPU_RXQ_CONFIGURATOR_H_

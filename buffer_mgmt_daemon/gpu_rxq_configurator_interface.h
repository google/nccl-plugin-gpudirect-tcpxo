/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_GPU_RXQ_CONFIGURATOR_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_GPU_RXQ_CONFIGURATOR_INTERFACE_H_

#include <memory>
#include <string>
#include <vector>

namespace tcpdirect {

struct GpuRxqConfiguration {
  std::string gpu_pci_addr;
  std::string nic_pci_addr;
  std::string ifname;
  std::string ip_addr;
  std::vector<int> rx_queue_ids;
};

class GpuRxqConfiguratorInterface {
 public:
  virtual void GetConfigurations(
      std::vector<std::unique_ptr<GpuRxqConfiguration>>* configurations) = 0;
  virtual ~GpuRxqConfiguratorInterface() = default;
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_GPU_RXQ_CONFIGURATOR_INTERFACE_H_

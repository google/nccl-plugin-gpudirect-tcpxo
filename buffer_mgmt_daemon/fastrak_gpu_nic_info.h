/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_GPU_NIC_INFO_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_GPU_NIC_INFO_H_

#include <memory>
#include <string>

#include "buffer_mgmt_daemon/gpu_rxq_configurator_interface.h"
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client.h"

namespace tcpdirect {

struct FasTrakGpuNicInfo {
  std::string gpu_pci_addr;
  std::string nic_pci_addr;
  std::string ifname;
  std::string ip_addr;
  std::string dxs_ip;
  std::string dxs_port;

  std::unique_ptr<dxs::BufferManagerInterface> dxs_client;
  explicit FasTrakGpuNicInfo(const struct GpuRxqConfiguration& config);
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_FASTRAK_GPU_NIC_INFO_H_

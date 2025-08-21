/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/fastrak_gpu_nic_info.h"

#include <arpa/inet.h>
#include <stdarg.h>
#include <stdio.h>

#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "buffer_mgmt_daemon/gpu_rxq_configurator_interface.h"
#include "dxs/client/dxs-client-types.h"

namespace tcpdirect {

FasTrakGpuNicInfo::FasTrakGpuNicInfo(const struct GpuRxqConfiguration& config) {
  this->gpu_pci_addr = config.gpu_pci_addr;
  this->nic_pci_addr = config.nic_pci_addr;
  this->ifname = config.ifname;
  this->ip_addr = config.ip_addr;

  this->dxs_ip = dxs::kDefaultDxsAddr;
  this->dxs_port = dxs::kDefaultDxsPort;
}

}  // namespace tcpdirect

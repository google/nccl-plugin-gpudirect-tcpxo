/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/common.h"

#include <fcntl.h>
#include <linux/limits.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client.h"
#include "buffer_mgmt_daemon/pci_utils.h"
#include "nccl.h"
#include "tcpdirect_plugin/fastrak_offload/utilities.h"

namespace fastrak {

int kNcclNetIfs = -1;

struct ncclSocketDev kNcclSocketDevs[MAX_IFS];

struct ncclSocketDev kNcclCtrlSocketDev;

static ncclResult_t ncclFasTrakGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  *pciPath = realpath(devicePath, nullptr);
  return ncclSuccess;
}

// Called only once during init before any
// nccl_shim instances are created.
ncclResult_t initializeNetIfs() {
  if (kNcclNetIfs == -1) {
    char names[MAX_IF_NAME_SIZE * MAX_IFS];
    union socketAddress addrs[MAX_IFS];
    char* fastrak_ifname = nullptr;
    const char* fastrak_ctrl_dev = nullptr;

    if (getenv("LOOPBACK_ONLY") != nullptr) {
      // Only use lo interface for loopback tests if env variable is set
      kNcclNetIfs =
          findInterfaces("lo", names, addrs, -1, MAX_IF_NAME_SIZE, MAX_IFS);
    } else if ((fastrak_ifname = getenv("NCCL_FASTRAK_IFNAME")) != nullptr) {
      kNcclNetIfs = findInterfaces(fastrak_ifname, names, addrs, -1,
                                   MAX_IF_NAME_SIZE, MAX_IFS);
    } else {
      kNcclNetIfs = findInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
    }
    if (kNcclNetIfs <= 0) {
      LOG(WARNING) << "No interfaces found.";
      return ncclSystemError;
    }

    if (!(fastrak_ctrl_dev = getenv("NCCL_FASTRAK_CTRL_DEV"))) {
      LOG(INFO) << "NCCL_FASTRAK_CTRL_DEV is not specified, using "
                   "eth0";
      fastrak_ctrl_dev = "eth0";
    }

    char ctrl_dev_name[MAX_IF_NAME_SIZE];
    union socketAddress addr;
    int num_ctrl_dev_found = findInterfaces(fastrak_ctrl_dev, ctrl_dev_name,
                                            &addr, -1, MAX_IF_NAME_SIZE, 1);
    if (num_ctrl_dev_found != 1) {
      LOG(WARNING) << absl::StrFormat(
          "No interfaces found for ctrl dev name %s", fastrak_ctrl_dev);
      return ncclSystemError;
    }
    strncpy(kNcclCtrlSocketDev.dev_name, ctrl_dev_name, MAX_IF_NAME_SIZE);
    memcpy(&kNcclCtrlSocketDev.addr, &addr, sizeof(union socketAddress));
    LOG(INFO) << "Using interface " << fastrak_ctrl_dev << " as ctrl dev";

    char line[2048];
    char addrline[2048];
    line[0] = '\0';
    for (int i = 0; i < kNcclNetIfs; i++) {
      strncpy(kNcclSocketDevs[i].dev_name, names + i * MAX_IF_NAME_SIZE,
              MAX_IF_NAME_SIZE);
      memcpy(&kNcclSocketDevs[i].addr, addrs + i, sizeof(union socketAddress));
      snprintf(kNcclSocketDevs[i].ip_addr, kIpAddrMaxLen, "%s",
               socketIPToString(&kNcclSocketDevs[i].addr.sa).c_str());
      if (ncclResult_t r = ncclFasTrakGetPciPath(kNcclSocketDevs[i].dev_name,
                                                 &kNcclSocketDevs[i].pci_path);
          r != ncclSuccess) {
        LOG(WARNING) << absl::StrFormat("Failed to get pci path for %s",
                                        kNcclSocketDevs[i].dev_name);
        return r;
      }
      pthread_mutex_init(&kNcclSocketDevs[i].reg_mutex, nullptr);
      kNcclSocketDevs[i].mr_cache.slots = nullptr;
      kNcclSocketDevs[i].mr_cache.capacity = 0;
      kNcclSocketDevs[i].mr_cache.population = 0;
      snprintf(line + strlen(line), 2047 - strlen(line), "[%d]%s:%s", i,
               names + i * MAX_IF_NAME_SIZE,
               socketToString(&addrs[i].sa, addrline));
    }
    std::sort(kNcclSocketDevs, kNcclSocketDevs + kNcclNetIfs,
              [](auto& a, auto& b) {
                // Use dev_name as a tiebreaker if pci_path is missing for both.
                if (a.pci_path == nullptr && b.pci_path == nullptr) {
                  return std::string(a.dev_name) < std::string(b.dev_name);
                }
                if (a.pci_path == nullptr) {
                  return true;
                }
                if (b.pci_path == nullptr) {
                  return false;
                }
                auto a_pci_bdf = std::string(strrchr(a.pci_path, '/') + 1);
                auto b_pci_bdf = std::string(strrchr(b.pci_path, '/') + 1);
                return a_pci_bdf < b_pci_bdf;
              });
    line[2047] = '\0';
    LOG(INFO) << "Using " << line;
  }
  return ncclSuccess;
}

static absl::flat_hash_map<std::string, int> GetGpuNicMapping() {
  absl::flat_hash_map<std::string, int> gpu_nic_mapping;
  auto nic_mapping_resp = tcpdirect::get_nic_mapping();
  if (!nic_mapping_resp.has_value()) {
    LOG(WARNING) << absl::StrFormat("Could not obtain NIC mappings from RxDM");
    return {};
  }
  for (const auto& [gpu_pci, nic] : nic_mapping_resp->pci_nic_map()) {
    int net_dev = -1;
    const std::string& ip_addr = nic.closest_nic_ip().Get(0);
    if (ipAddrDiscovered(ip_addr.c_str(), &net_dev) != ncclSuccess) {
      LOG(WARNING) << absl::StrFormat("IP %s is never discovered before.",
                                      ip_addr);
      continue;
    }
    gpu_nic_mapping[absl::AsciiStrToLower(gpu_pci)] = net_dev;
    LOG(LEVEL(getConnectionLogLevel()))
        << "Discovered GPU PCI " << gpu_pci << ", closest NIC is "
        << kNcclSocketDevs[net_dev].dev_name << " (address "
        << kNcclSocketDevs[net_dev].ip_addr << ", index " << net_dev << ")";
  }
  return gpu_nic_mapping;
}

absl::StatusOr<int> getClosestNetdev(std::string_view gpu_pci) {
  // absl::NoDestructor is available open source
  static absl::NoDestructor<absl::flat_hash_map<std::string, int>>
      gpu_nic_mapping(GetGpuNicMapping());
  auto it = gpu_nic_mapping->find(absl::AsciiStrToLower(gpu_pci));
  if (it == gpu_nic_mapping->end()) {
    return absl::InternalError(absl::StrFormat(
        "Could not find closest netdev for GPU PCI %s", gpu_pci));
  }
  return it->second;
}

// Checks if an IP is discovered and recorded during init
ncclResult_t ipAddrDiscovered(const char* ip_addr, int* idx) {
  for (int i = 0; i < kNcclNetIfs; i++) {
    std::string saddr = socketIPToString(&kNcclSocketDevs[i].addr.sa);
    if (!strcmp(ip_addr, saddr.c_str())) {
      *idx = i;
      return ncclSuccess;
    }
  }
  return ncclSystemError;
}

int GetSpeed(std::string_view dev_name) {
  std::string speedPath = absl::StrFormat("/sys/class/net/%s/speed", dev_name);
  int fd = open(speedPath.c_str(), O_RDONLY);
  int speed = 0;
  if (fd != -1) {
    char speedStr[] = "        ";
    if (read(fd, speedStr, sizeof(speedStr) - 1) > 0) {
      speed = strtol(speedStr, nullptr, 0);
    }
    close(fd);
  }
  if (speed <= 0) {
    LOG(INFO) << absl::StrFormat(
        "Could not get speed from %s. Defaulting to 10 Gbps.", speedPath);
    speed = 10000;
  }
  return speed;
}

static absl::flat_hash_map<std::string, uint8_t>& GetPciAddrToFastrakIdxMap() {
  static absl::NoDestructor<absl::flat_hash_map<std::string, uint8_t>>
      pci_addr_to_fastrak_idx_map{};

  static absl::once_flag once;
  absl::call_once(once, []() {
    std::vector<std::string> gpus;
    if (tcpdirect::list_vendor_devices(tcpdirect::kSysfsPciDevicesPath, gpus,
                                       tcpdirect::kNvidiaVendorId,
                                       tcpdirect::kH100DeviceId) < 0) {
      LOG(WARNING) << "Failed to list H100 GPUs under ["
                   << tcpdirect::kSysfsPciDevicesPath << "]";
      return;
    }

    // Lexicographic order matches enumeration order.
    std::sort(gpus.begin(), gpus.end());
    // Remove duplicate entries
    gpus.erase(std::unique(gpus.begin(), gpus.end()), gpus.end());

    if (!gpus.empty()) {
      LOG(INFO) << "GetPciAddrToFastrakIdxMap: Found GPUs:";
    } else {
      LOG(INFO) << "GetPciAddrToFastrakIdxMap: No GPUs found.";
    }
    for (auto idx = 0u; idx < gpus.size(); idx++) {
      const auto bdf = absl::AsciiStrToLower(gpus[idx]);
      LOG(INFO) << absl::StrFormat("\tFasTrak IDX: [%d]. PCI addr: [%s]", idx,
                                   bdf);
      pci_addr_to_fastrak_idx_map->insert({std::move(bdf), idx});
    }
  });

  return *pci_addr_to_fastrak_idx_map;
}

absl::StatusOr<uint8_t> GetFastrakIdxFromPci(absl::string_view pci_addr) {
  std::string sanitized_pci_addr = absl::AsciiStrToLower(pci_addr);
  if (!GetPciAddrToFastrakIdxMap().contains(sanitized_pci_addr)) {
    LOG(ERROR) << absl::StrFormat(
        "No FasTrak index found for PCI addr %s. Known PCI addresses and their "
        "FasTrak indices:",
        sanitized_pci_addr);
    for (const auto& [pci_addr, fastrak_idx] : GetPciAddrToFastrakIdxMap()) {
      LOG(ERROR) << "  " << pci_addr << ": " << fastrak_idx;
    }
    return absl::NotFoundError(absl::StrFormat(
        "No FasTrak index found for PCI addr %s", sanitized_pci_addr));
  }
  return GetPciAddrToFastrakIdxMap()[sanitized_pci_addr];
}

absl::StatusOr<std::string> GetPciFromFastrakIdx(uint8_t fastrak_idx) {
  for (const auto& [addr, idx] : GetPciAddrToFastrakIdxMap()) {
    if (idx == fastrak_idx) {
      return addr;
    }
  }

  LOG(ERROR) << absl::StrFormat(
      "No PCI Address found for FasTrak index %d. Known PCI addresses and "
      "their FasTrak indices:",
      fastrak_idx);
  for (const auto& [pci_addr, fastrak_idx] : GetPciAddrToFastrakIdxMap()) {
    LOG(ERROR) << "  " << pci_addr << ": " << fastrak_idx;
  }
  return absl::NotFoundError(absl::StrFormat(
      "No PCI Address found for FasTrak index %d", fastrak_idx));
}

absl::flat_hash_map<std::string, uint8_t>& TestOnlyGetPciAddrToFastrakIdxMap() {
  return GetPciAddrToFastrakIdxMap();
}

}  // namespace fastrak

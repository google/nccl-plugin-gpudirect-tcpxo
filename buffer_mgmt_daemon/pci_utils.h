/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_PCI_UTILS_H_
#define BUFFER_MGMT_DAEMON_PCI_UTILS_H_

#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
namespace tcpdirect {

// <2-4 digit domain>:<2-4 digit bus>:<2 digit device>:<1 digit function>
#define MAX_PCI_ADDR_LEN 16

#define HOST_MAX_LEN 1024

extern const char kSysfsPciDevicesPath[];

inline constexpr absl::string_view kNvidiaVendorId = "0x10de";
inline constexpr absl::string_view kH100DeviceId = "0x2330";

// Reads up to PCI_ID_LEN bytes of data from the specified `path`
std::string read_pci_id(absl::string_view path);

// List all child devices starting from a PCI device that has the
// specified vendor ID
// If device_id is specified, then only devices with the same device id
// will be listed

int list_vendor_devices(
    const char* parent_dir_path, std::vector<std::string>& candidates,
    absl::string_view vendor_id,
    std::optional<absl::string_view> device_id = std::nullopt);

int parse_pci_addr(const char* pci_addr, uint16_t* domain, uint16_t* bus,
                   uint16_t* device, uint16_t* function);

std::string get_ip_from_sockaddr(struct sockaddr* sa);

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_PCI_UTILS_H_

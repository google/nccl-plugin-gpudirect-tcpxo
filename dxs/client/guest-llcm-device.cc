/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/guest-llcm-device.h"

#include <glob.h>
#include <sys/fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dxs/client/control-command.h"
#include "dxs/client/derive_dxs_address.h"
#include "dxs/client/guest_llcm/guest_llcm.h"
#include "dxs/client/oss/htonl.h"
#include "dxs/client/oss/status_macros.h"

ABSL_FLAG(bool, dxs_guest_llcm_use_ipv6, false,
          "Use IPv6 rather than IPv4 for the llcm VIP.");

namespace dxs {
namespace {

// Llcm device info.
static constexpr uint16_t kVendorId = 0x1ae0;
static constexpr uint16_t kDeviceId = 0x0084;

// Returns the contents of 'filename'.
absl::StatusOr<std::string> GetContents(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("GetContents() failed on ", filename));
  }
  auto file_cleanup = absl::MakeCleanup([fd, filename] {
    int err = close(fd);
    if (err == EOF) {
      PLOG(ERROR) << "Failed to fclose() " << filename;
    }
  });

  struct stat stats;
  int err = fstat(fd, &stats);
  if (err != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("GetContents() failed on ", filename));
  }

  std::string contents;
  contents.resize(stats.st_size);
  size_t count = read(fd, contents.data(), stats.st_size);
  if (count < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("GetContents() failed on ", filename));
  }
  contents.resize(count);

  return contents;
}

// Returns files matching the glob pattern 'pattern'.
absl::StatusOr<std::vector<std::string>> Glob(const std::string& pattern) {
  glob_t glob_data = {};
  int err = glob(pattern.c_str(), 0, nullptr, &glob_data);
  auto glob_cleanup = absl::MakeCleanup([&glob_data] { globfree(&glob_data); });
  if (err == GLOB_NOMATCH) {
    return absl::NotFoundError(
        absl::StrCat("No files found matching ", pattern));
  } else if (err != 0) {
    return absl::InternalError(
        absl::StrCat("glob() failed with pattern ", pattern, ", err=", err));
  }

  std::vector<std::string> matching_paths;
  for (size_t i = 0; i != glob_data.gl_pathc; ++i) {
    matching_paths.emplace_back(glob_data.gl_pathv[i]);
  }
  return matching_paths;
}

// Reads and returns the contents of 'filename' as a single ASCII hexadecimal
// uint32 (e.g., "0xabcd3", "ABCD3", etc).
absl::StatusOr<uint32_t> ReadFileAsHexUint32(const std::string& filename) {
  ASSIGN_OR_RETURN(std::string contents, GetContents(filename));
  absl::StripAsciiWhitespace(&contents);
  uint32_t value;
  if (!absl::SimpleHexAtoi(contents, &value)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse file ", filename));
  }
  return value;
}

// Discovers all Llcm devices on the host and returns their BDFs.
absl::StatusOr<std::vector<std::string>> GetPcieDeviceBdfs(
    absl::string_view sysfs_pci_devices_dir) {
  const std::string device_pattern = absl::StrCat(sysfs_pci_devices_dir, "/*");
  ASSIGN_OR_RETURN(std::vector<std::string> device_paths, Glob(device_pattern));

  std::vector<std::string> device_bdfs;
  for (const auto& device_path : device_paths) {
    // Check vendor ID.
    std::string vendor_id_path = absl::StrCat(device_path, "/vendor");
    absl::StatusOr<uint32_t> vendor_id_or = ReadFileAsHexUint32(vendor_id_path);
    if (!vendor_id_or.ok()) {
      LOG(WARNING) << "Failed to read vendor ID from " << vendor_id_path << ": "
                   << vendor_id_or.status();
      continue;
    }
    if (*vendor_id_or != kVendorId) continue;

    // Check device ID.
    std::string device_id_path = absl::StrCat(device_path, "/device");
    absl::StatusOr<uint32_t> device_id_or = ReadFileAsHexUint32(device_id_path);
    if (!device_id_or.ok()) {
      LOG(WARNING) << "Failed to read device ID from " << device_id_path << ": "
                   << device_id_or.status();
      continue;
    }
    if (*device_id_or != kDeviceId) continue;

    std::vector<std::string> parts = absl::StrSplit(device_path, '/');
    std::string basename = parts.back();
    device_bdfs.emplace_back(std::move(basename));
  }
  return device_bdfs;
}

// Reads and returns the IPv4 or IPv6 address in the specified Llcm PCIe
// device's BAR resource4, which currently solely contains the address in host
// byte order.
absl::StatusOr<WireSocketAddr> ReadVipForDevice(
    absl::string_view bdf, absl::string_view sysfs_pci_devices_dir) {
  const std::string resource4_path =
      absl::StrCat(sysfs_pci_devices_dir, "/", bdf, "/resource4");
  int fd = open(resource4_path.c_str(), O_RDONLY);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Failed to open BAR4 for device ", bdf));
  }
  absl::Cleanup close_fd = [=] {
    if (close(fd) < 0) PLOG(ERROR) << "Failed to close BAR4 for device " << bdf;
  };
  // BAR4 is either a packed IPv4 address or a packed IPv6 address. A packed
  // IPv6 address is read using a uint128.
  size_t map_size = sizeof(absl::uint128);
  void* mem = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd, 0);
  if (mem == MAP_FAILED) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Failed to mmap BAR4 for device ", bdf));
  }
  absl::Cleanup unmap_mem = [=] {
    if (munmap(mem, map_size) < 0) {
      PLOG(ERROR) << "Failed to unmap BAR4 for device " << bdf;
    }
  };
  WireSocketAddr addr{};
  if (absl::GetFlag(FLAGS_dxs_guest_llcm_use_ipv6)) {
    addr.is_ipv6 = true;
    absl::uint128 ipv6_address;
    std::memcpy(&ipv6_address, mem, sizeof(ipv6_address));
    uint64_t low_net = fhtonll(absl::Uint128High64(ipv6_address));
    uint64_t high_net = fhtonll(absl::Uint128Low64(ipv6_address));
    std::memcpy(addr.addr, &low_net, sizeof(low_net));
    std::memcpy(addr.addr + 8, &high_net, sizeof(high_net));
  } else {
    addr.is_ipv6 = false;
    uint32_t ipv4_address = 0;
    std::memcpy(&ipv4_address, mem, sizeof(ipv4_address));
    ipv4_address = fntohl(ipv4_address);
    std::memcpy(addr.addr, &ipv4_address, sizeof(ipv4_address));
  }
  return addr;
}

}  // namespace

absl::Status GuestLlcmDevice::Init(WireSocketAddr local_address,
                                   absl::string_view sysfs_pci_devices_dir) {
  ASSIGN_OR_RETURN(std::string bdf,
                   FindLlcmBdfForNic(local_address, sysfs_pci_devices_dir));

  constexpr size_t kLlcmBackingSize = 8 << 20;         // 8 MiB
  constexpr size_t kReverseLlcmBackingSize = 8 << 20;  // 8 MiB
  ASSIGN_OR_RETURN(
      device_, GuestLlcm::Create(bdf, kLlcmBackingSize, kReverseLlcmBackingSize,
                                 sysfs_pci_devices_dir));
  return absl::OkStatus();
}

absl::Span<volatile uint8_t> GuestLlcmDevice::GetLocalMemory(
    uint64_t offset, uint64_t size) const {
  auto* base =
      reinterpret_cast<volatile uint8_t*>(device_->reverse_llcm().data());
  return absl::MakeSpan(base + offset, size);
}

absl::Span<volatile uint8_t> GuestLlcmDevice::GetRemoteMemory(
    uint64_t offset, uint64_t size) const {
  auto* base = reinterpret_cast<volatile uint8_t*>(device_->llcm().data());
  return absl::MakeSpan(base + offset, size);
}

absl::StatusOr<std::string> FindLlcmBdfForNic(
    WireSocketAddr address, absl::string_view sysfs_pci_devices_dir) {
  ASSIGN_OR_RETURN(std::vector<std::string> bdfs,
                   GetPcieDeviceBdfs(sysfs_pci_devices_dir));
  for (const std::string& bdf : bdfs) {
    auto vip_or = ReadVipForDevice(bdf, sysfs_pci_devices_dir);
    if (!vip_or.ok()) {
      LOG(WARNING) << "Failed to read address from device " << bdf << ": "
                   << vip_or.status();
      continue;
    }

    if (vip_or->is_ipv6) {
      ASSIGN_OR_RETURN(std::string address_str, UnpackIpAddress(address));
      ASSIGN_OR_RETURN(std::string compare_address_str,
                       UnpackIpAddress(*vip_or));
      LOG(INFO) << "IPV6: " << address_str
                << " compare ip: " << compare_address_str;
      WireSocketAddr temp_addr = address;
      std::memset(temp_addr.addr + sizeof(uint64_t), 0, sizeof(uint64_t));
      if (*vip_or == temp_addr) return bdf;
      ASSIGN_OR_RETURN(address_str, UnpackIpAddress(temp_addr));
      LOG(INFO) << "Not matching with the ip " << address_str;
    }

    if (*vip_or == address) return bdf;
    ASSIGN_OR_RETURN(std::string vip_str, UnpackIpAddress(*vip_or));
    LOG(INFO) << "Device doesn't match: " << vip_str;
  }
  ASSIGN_OR_RETURN(std::string address_str, UnpackIpAddress(address));
  return absl::NotFoundError(
      absl::StrCat("No LLCM device found for address ", address_str));
}

}  // namespace dxs

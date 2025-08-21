/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_GUEST_LLCM_DEVICE_H_
#define DXS_CLIENT_GUEST_LLCM_DEVICE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/flags/declare.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dxs/client/control-command.h"
#include "dxs/client/guest_llcm/guest_llcm.h"
#include "dxs/client/llcm-memory-interface.h"

ABSL_DECLARE_FLAG(bool, dxs_guest_llcm_use_ipv6);

namespace dxs {

class GuestLlcmDevice : public LlcmMemoryInterface {
 public:
  ~GuestLlcmDevice() override = default;

  absl::Status Init(
      WireSocketAddr local_address,
      absl::string_view sysfs_pci_devices_dir = "/sys/bus/pci/devices");

  // Returns the Reverse Llcm memory in the specified region. Subject to
  // the same size and alignment constraints as SpscMessagingQueuePair::Create.
  absl::Span<volatile uint8_t> GetLocalMemory(uint64_t offset,
                                              uint64_t size) const override;
  // Returns the Llcm memory in the specified region. Subject to the same
  // size and alignment constraints as SpscMessagingQueuePair::Create.
  absl::Span<volatile uint8_t> GetRemoteMemory(uint64_t offset,
                                               uint64_t size) const override;

 private:
  std::unique_ptr<GuestLlcm> device_;
};

// Returns the BDF of the Llcm device corresponding to the NIC specified by
// 'address'. Searches in the sysfs dir 'sysfs_pci_devices_dir'.
absl::StatusOr<std::string> FindLlcmBdfForNic(
    WireSocketAddr address, absl::string_view sysfs_pci_devices_dir);

}  // namespace dxs

#endif  // DXS_CLIENT_GUEST_LLCM_DEVICE_H_

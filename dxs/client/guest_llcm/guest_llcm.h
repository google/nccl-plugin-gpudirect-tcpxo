/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_GUEST_LLCM_GUEST_LLCM_H_
#define DXS_CLIENT_GUEST_LLCM_GUEST_LLCM_H_

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace dxs {

// Helper class to access Llcm and ReverseLlcm MMIO regions from within
// a virtual machine. There will be a PCI device emulated by Vanadium with
// vendor:device id 1ae0:0084. Its BAR0 is for Llcm and BAR2 is for
// ReverseLlcm.
class GuestLlcm {
 public:
  // The domain bdf must be in form of "0000:00:00.0".
  static absl::StatusOr<std::unique_ptr<GuestLlcm>> Create(
      absl::string_view domain_bdf, uint64_t llcm_size,
      uint64_t reverse_llcm_size,
      absl::string_view llcm_device_directory = "/sys/bus/pci/devices");

  absl::Span<uint8_t> llcm() const {
    return absl::MakeSpan(static_cast<uint8_t*>(llcm_.get()),
                          llcm_.get_deleter().size);
  }

  absl::Span<uint8_t> reverse_llcm() {
    return absl::MakeSpan(static_cast<uint8_t*>(reverse_llcm_.get()),
                          reverse_llcm_.get_deleter().size);
  }

 private:
  struct MemmapDeleter {
    void operator()(void* ptr) { munmap(ptr, size); }
    size_t size;
  };

  GuestLlcm(std::unique_ptr<void, MemmapDeleter> llcm,
            std::unique_ptr<void, MemmapDeleter> reverse_llcm)
      : llcm_(std::move(llcm)), reverse_llcm_(std::move(reverse_llcm)) {}

  std::unique_ptr<void, MemmapDeleter> llcm_;
  std::unique_ptr<void, MemmapDeleter> reverse_llcm_;
};

}  // namespace dxs

#endif  // DXS_CLIENT_GUEST_LLCM_GUEST_LLCM_H_

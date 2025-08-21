/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_LLCM_MEMORY_INTERFACE_H_
#define DXS_CLIENT_LLCM_MEMORY_INTERFACE_H_

#include <cstdint>

#include "absl/types/span.h"

namespace dxs {

// LlcmMemory interface to fetch the local and remote memory for LLCM setup.
class LlcmMemoryInterface {
 public:
  virtual ~LlcmMemoryInterface() = default;

  // Returns the Reverse Llcm memory in the specified region. Subject to
  // the same size and alignment constraints as SpscMessagingQueuePair::Create.
  virtual absl::Span<volatile uint8_t> GetLocalMemory(uint64_t offset,
                                                      uint64_t size) const = 0;
  // Returns the Llcm memory in the specified region. Subject to the same
  // size and alignment constraints as SpscMessagingQueuePair::Create.
  virtual absl::Span<volatile uint8_t> GetRemoteMemory(uint64_t offset,
                                                       uint64_t size) const = 0;
};

}  // namespace dxs

#endif  // DXS_CLIENT_LLCM_MEMORY_INTERFACE_H_

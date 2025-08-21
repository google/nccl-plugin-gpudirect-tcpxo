/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/coalesce-iovecs.h"

#include <cstdint>
#include <iterator>
#include <vector>

namespace dxs {
namespace {

bool WouldAddOverflow(uint32_t len1, uint32_t len2) {
  uint32_t result = len1 + len2;
  return result < len1;  // Can only happen if wraparound occurred.
}

}  // namespace

std::vector<iovec> CoalesceIovecs(std::vector<iovec> iovecs) {
  auto merge = iovecs.begin();
  for (auto it = std::next(merge); it != iovecs.end(); ++it) {
    if (static_cast<char*>(merge->iov_base) + merge->iov_len == it->iov_base &&
        !WouldAddOverflow(merge->iov_len, it->iov_len)) {
      merge->iov_len += it->iov_len;
    } else {
      ++merge;
      *merge = *it;
    }
  }
  iovecs.erase(++merge, iovecs.end());
  return iovecs;
}

}  // namespace dxs

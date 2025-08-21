/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_COALESCE_IOVECS_H_
#define DXS_CLIENT_COALESCE_IOVECS_H_

#include <sys/uio.h>

#include <vector>

namespace dxs {

// Coalesces the given vector of iovecs without rearrangement.
std::vector<iovec> CoalesceIovecs(std::vector<iovec> iovecs);

}  // namespace dxs

#endif  // DXS_CLIENT_COALESCE_IOVECS_H_

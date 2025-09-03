/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/macro.h"

#include <cstdint>

#include "nccl.h"
#include "nccl_common.h"

namespace fastrak {
namespace {

static void dummyDebugLog(ncclDebugLogLevel level, uint64_t flags,
                          const char* filefunc, int line, const char* fmt,
                          ...) {}

// Set only during one-time init.
ncclDebugLogger_t ncclLogFunc = dummyDebugLog;

}  // namespace

void SetNcclLogFunc(ncclDebugLogger_t logger) { ncclLogFunc = logger; }
ncclDebugLogger_t GetNcclLogFunc() { return ncclLogFunc; }

}  // namespace fastrak

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_MACRO_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_MACRO_H_

#include "absl/log/log.h"
#include "nccl_common.h"

namespace fastrak {

void SetNcclLogFunc(ncclDebugLogger_t logger);
ncclDebugLogger_t GetNcclLogFunc();

}  // namespace fastrak

// Propagate errors up
#define NCCLCHECK(call)                                                        \
  do {                                                                         \
    ncclResult_t res = call;                                                   \
    if (res != ncclSuccess) {                                                  \
      /* Print the back trace*/                                                \
      LOG(WARNING) << absl::StrFormat("%s:%d -> %d", __FILE__, __LINE__, res); \
      return res;                                                              \
    }                                                                          \
  } while (0);

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_MACRO_H_

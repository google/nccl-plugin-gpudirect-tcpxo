/*
* Copyright 2025 Google LLC
*
* Use of this source code is governed by a BSD-style
* license that can be found in the LICENSE.md file or at
* https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/cuda_logging.h"

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

namespace tcpdirect {

bool CUCallSuccess(CUresult err) {
  if (err != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* reason = nullptr;
    if (cuGetErrorName(err, &name)) {
      LOG(FATAL) << "Error: error getting error name from CU error " << err;
      return false;
    }
    if (cuGetErrorString(err, &reason)) {
      LOG(FATAL) << "Error: error getting error string from CU error " << err;
      return false;
    }
    LOG(ERROR) << absl::StrFormat("cuda error detected! name: %s; string: %s",
                                  name, reason);
    return false;
  }
  return true;
}

}  // namespace tcpdirect

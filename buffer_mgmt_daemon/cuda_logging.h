/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_CUDA_LOGGING_CU_H_
#define BUFFER_MGMT_DAEMON_CUDA_LOGGING_CU_H_

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cuda.h"

namespace tcpdirect {

#define CU_ASSERT_SUCCESS(expr)                                                \
  {{CUresult __err = (expr);                                                   \
  if (__err != CUDA_SUCCESS) {                                                 \
    const char* name = "[unknown]";                                            \
    const char* reason = "[unknown]";                                          \
    if (cuGetErrorName(__err, &name)) {                                        \
      LOG(ERROR) << "Error: error getting error name";                         \
    }                                                                          \
    if (cuGetErrorString(__err, &reason)) {                                    \
      LOG(ERROR) << "Error: error getting error string";                       \
    }                                                                          \
    LOG(FATAL) << absl::StrFormat("cuda error detected! name: %s; string: %s", \
                                  name, reason);                               \
  }                                                                            \
  }                                                                            \
  }

#define CU_ASSERT_SUCCESS_GOTO(expr, label)                                    \
  {{CUresult __err = (expr);                                                   \
  if (__err != CUDA_SUCCESS) {                                                 \
    const char* name = "[unknown]";                                            \
    const char* reason = "[unknown]";                                          \
    if (cuGetErrorName(__err, &name)) {                                        \
      LOG(ERROR) << "Error: error getting error name";                         \
    }                                                                          \
    if (cuGetErrorString(__err, &reason)) {                                    \
      LOG(ERROR) << "Error: error getting error string";                       \
    }                                                                          \
    LOG(FATAL) << absl::StrFormat("cuda error detected! name: %s; string: %s", \
                                  name, reason);                               \
    goto label;                                                                \
  }                                                                            \
  }                                                                            \
  }

bool CUCallSuccess(CUresult err);

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_CUDA_LOGGING_CU_H_

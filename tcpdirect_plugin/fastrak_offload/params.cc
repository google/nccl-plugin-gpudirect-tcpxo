/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/params.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tcpdirect_plugin/fastrak_offload/const_params.h"

namespace fastrak {
namespace params_internal {
namespace {
static void InitParam(std::string_view name, const ParamData& data) {
  std::string env_var = "NCCL_";
  absl::StrAppend(&env_var, name);
  char* str = getenv(env_var.c_str());
  int64_t tmp = 0;
  if (str && strlen(str) > 0) {
    if (absl::SimpleAtoi(str, &tmp)) {
      if (tmp < data.min_value || tmp > data.max_value) {
        LOG(INFO) << absl::StrFormat(
            "Invalid value %ld for %s - must be between %ld and %ld, using "
            "default of %ld",
            tmp, env_var.c_str(), data.min_value, data.max_value, *data.slot);
        return;
      }
      *data.slot = tmp;
      LOG(INFO) << absl::StrFormat("%s set by environment to %ld.",
                                   env_var.c_str(), *data.slot);
    } else {
      LOG(INFO) << absl::StrFormat(
          "Invalid value %s for %s, using default %ld.", str, env_var.c_str(),
          *data.slot);
    }
  } else {
    LOG(INFO) << absl::StrFormat(
        "No environment variable %s set, using default %ld.", env_var.c_str(),
        *data.slot);
  }
}
}  // namespace
}  // namespace params_internal

void InitParams() {
  for (const auto& [name, param_data] : params_internal::GetParamDataMap()) {
    params_internal::InitParam(name, param_data);
  }
}

}  // namespace fastrak

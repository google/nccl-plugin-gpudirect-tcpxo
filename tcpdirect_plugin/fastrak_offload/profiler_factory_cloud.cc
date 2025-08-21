/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include <memory>

#include "tcpdirect_plugin/fastrak_offload/profiler_factory.h"
#include "tcpdirect_plugin/fastrak_offload/profiler_noop.h"

namespace fastrak {

std::unique_ptr<ProfilerFactoryInterface> GetProfilerFactory(
    const ProfilerFactoryOptions& options) {
  return std::make_unique<NoOpProfilerFactory>();
}

}  // namespace fastrak

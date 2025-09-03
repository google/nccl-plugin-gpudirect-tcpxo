/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_INIT_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_INIT_H_

#include "absl/status/status.h"
#include "nccl.h"
#include "nccl_common.h"

namespace fastrak {

// Performs a one-time thread-safe initialization
// of core plug-in data. Subsequent calls are no-op
// and return the status of the first call.
// Global variables written by this routine need not be
// atomic as long as they're never written to again
// because all future reads come from threads that are
// children of the init thread
absl::Status PluginCoreInit(ncclDebugLogger_t logger);

// Disables initializing absl logging. Useful for tests and for binaries that
// statically link the plugin.
void DisableLogInit();

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_INIT_H_

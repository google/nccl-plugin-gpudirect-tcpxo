/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

/**
 * Includes all tuning parameters for NCCL plugin.
 */
#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PARAMS_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PARAMS_H_

#include "tcpdirect_plugin/fastrak_offload/const_params.h"

namespace fastrak {

// Initializes params from environment.
//
// getenv can only be safely used after main (so not in the constructor of a
// global singleton class), so the params are initialized early in nccl_shim
// init, so these can't be const
void InitParams();

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_PARAMS_H_

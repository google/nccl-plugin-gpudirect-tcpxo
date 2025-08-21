/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_NCCL_COMPAT_NCCL_NET_COMPAT_H_
#define TCPDIRECT_PLUGIN_NCCL_COMPAT_NCCL_NET_COMPAT_H_

#include "nccl.h"

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 26, 2)
#include "plugin/nccl_net.h"
#else
#include "nccl_net.h"
#endif

#endif  // TCPDIRECT_PLUGIN_NCCL_COMPAT_NCCL_NET_COMPAT_H_

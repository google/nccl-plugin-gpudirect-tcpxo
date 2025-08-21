/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

/**
 * Shared definitions between NCCL and NCCL plugin.
 */
#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_SHARED_DEFS_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_SHARED_DEFS_H_

#include <stdint.h>

#define NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH 16

union alignas(16) loadMeta {
  uint64_t r64[2];
  struct {
    uint32_t src_off;
    uint32_t len;
    uint64_t dst_off;
  };
};
static_assert(sizeof(union loadMeta) == 16);

/****** global memory ******/

#define NET_UNPACK_MAX_QUEUE_DEPTH 16      // MAX_REQUESTS
#define NET_UNPACK_MAX_SLICE_SIZE 4194304  // 4MB per Irecv call
#define SLICE_PAGE_SIZE 4096
#define NET_UNPACK_MAX_SLICE_PAGES \
  (NET_UNPACK_MAX_SLICE_SIZE / SLICE_PAGE_SIZE * 2)  // * 2 for slack, wasteful.

struct netUnpackMeta {
  loadMeta mem[NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH]
              [NET_UNPACK_MAX_SLICE_PAGES];
  uint64_t cnt[NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH];
};

struct unpackNetDeviceHandle {
  struct netUnpackMeta* meta;  // mapped
  void* bounce_buf;
  uint64_t head;
};

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_SHARED_DEFS_H_

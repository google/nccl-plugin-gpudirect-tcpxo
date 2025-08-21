/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_CLIENT_BOUNCE_BUFFER_HANDLE_H_
#define BUFFER_MGMT_DAEMON_CLIENT_BOUNCE_BUFFER_HANDLE_H_

#include <cstdint>

#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "dxs/client/dxs-client-types.h"

namespace tcpdirect {

constexpr uint32_t MEM_HANDLE_SIZE = 64;

struct MemFdMetadata {
  int fd = -1;
  uint64_t size = 0;
  uint64_t align = 0;
};

struct BounceBufHandle {
  ExportHandleType handle_type;
  union {
    char mem_handle[MEM_HANDLE_SIZE];
    struct MemFdMetadata mem_fd;
  };
  dxs::Reg reg_handle = dxs::kInvalidRegistration;
  BounceBufHandle() : handle_type(MEM_UNSPECIFIED) {}
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_CLIENT_BOUNCE_BUFFER_HANDLE_H_

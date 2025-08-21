/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_MTU_UTILS_H_
#define DXS_CLIENT_MTU_UTILS_H_

#include <netinet/ip6.h>
#include <netinet/udp.h>

#include <cstddef>

namespace dxs {

constexpr size_t GetUsableSizeForMtu(size_t mtu) {
  return mtu - sizeof(ip6_hdr) - sizeof(udphdr)  // when using SCTP over UDP
         - 12                                    // sizeof(sctphdr)
         - 4                                     // sizeof(sctp_chunkhdr)
         - 12;                                   // sizeof(sctp_datahdr)
}

}  // namespace dxs

#endif  // DXS_CLIENT_MTU_UTILS_H_

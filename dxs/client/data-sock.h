/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_DATA_SOCK_H_
#define DXS_CLIENT_DATA_SOCK_H_

#include <stdint.h>
#include <sys/uio.h>

#include <string>

namespace dxs {

enum class DataSockStatus : int8_t {
  kConnected = 1,
  kPendingConnect = 0,
  kInvalid = -1,
  kConnectionRefused = -2,
  kTimeout = -3,
  kUnreachable = -4,
  kUnknownError = -5,
  kInternalError = -6,
};

std::string ToString(DataSockStatus e);

template <typename Sink>
void AbslStringify(Sink& sink, DataSockStatus e) {
  sink.Append(ToString(e));
}

}  // namespace dxs

#endif  // DXS_CLIENT_DATA_SOCK_H_

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/data-sock.h"

#include <string>

#include "absl/log/log.h"

namespace dxs {

std::string ToString(DataSockStatus e) {
  switch (e) {
    case DataSockStatus::kConnected:
      return "kConnected";
      break;
    case DataSockStatus::kPendingConnect:
      return "kPendingConnect";
      break;
    case DataSockStatus::kInvalid:
      return "kInvalid";
      break;
    case DataSockStatus::kConnectionRefused:
      return "kConnectionRefused";
      break;
    case DataSockStatus::kTimeout:
      return "kTimeout";
      break;
    case DataSockStatus::kUnreachable:
      return "kUnreachable";
      break;
    case DataSockStatus::kUnknownError:
      return "kUnknownError";
      break;
    case DataSockStatus::kInternalError:
      return "kInternalError";
      break;
  }
  return "INVALID";
}

}  // namespace dxs

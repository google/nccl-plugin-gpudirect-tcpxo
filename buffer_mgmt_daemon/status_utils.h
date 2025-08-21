/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_STATUS_UTILS_H_
#define BUFFER_MGMT_DAEMON_STATUS_UTILS_H_

#include "absl/status/status.h"
#include "google/rpc/status.pb.h"

namespace tcpdirect {

void SaveStatusToProto(const absl::Status& status,
                       ::google::rpc::Status* proto_status);

absl::Status MakeStatusFromProto(const ::google::rpc::Status& proto_status);

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_STATUS_UTILS_H_

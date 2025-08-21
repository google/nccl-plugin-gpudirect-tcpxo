/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/status_utils.h"

#include "absl/status/status.h"
#include "google/rpc/status.pb.h"

namespace tcpdirect {
void SaveStatusToProto(const absl::Status& status,
                       ::google::rpc::Status* proto_status) {
  proto_status->set_code(status.raw_code());
  proto_status->set_message(status.message());
}

absl::Status MakeStatusFromProto(const ::google::rpc::Status& proto_status) {
  return absl::Status(absl::StatusCode(proto_status.code()),
                      proto_status.message());
}
}  // namespace tcpdirect

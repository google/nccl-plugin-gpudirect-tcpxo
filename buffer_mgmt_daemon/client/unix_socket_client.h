/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_CLIENT_UNIX_SOCKET_CLIENT_H_
#define BUFFER_MGMT_DAEMON_CLIENT_UNIX_SOCKET_CLIENT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "buffer_mgmt_daemon/common/unix_socket_connection.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"

namespace tcpdirect {

// UnixSocketClient implements a blocking request/response client over unix
// domain sockets. This class uses abstract sockets to avoid user namespace
// conflicts.
//
// Thread safe.
class UnixSocketClient {
 public:
  explicit UnixSocketClient(std::string path) : path_(std::move(path)) {}
  absl::Status Connect();
  absl::StatusOr<UnixSocketMessage> MakeRequest(UnixSocketMessage msg);

 private:
  const std::string path_;
  absl::Mutex mu_;
  std::unique_ptr<UnixSocketConnection> conn_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_CLIENT_UNIX_SOCKET_CLIENT_H_

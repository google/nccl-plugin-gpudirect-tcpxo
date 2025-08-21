/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/client/unix_socket_client.h"

#include <errno.h>
#include <stddef.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <memory>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/common/uds_helpers.h"
#include "buffer_mgmt_daemon/common/unix_socket_connection.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "dxs/client/oss/status_macros.h"

ABSL_FLAG(absl::Duration, uds_connect_timeout, absl::Seconds(15),
          "Timeout for UDS connections.");

namespace tcpdirect {

absl::Status UnixSocketClient::Connect() {
  ASSIGN_OR_RETURN(sockaddr_un server_addr, UdsSockaddr(path_));

  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("socket() error: %d", error_number));
  }

  timeval tv = absl::ToTimeval(absl::GetFlag(FLAGS_uds_connect_timeout));
  int err = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  if (err != 0) {
    close(fd);
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("setsockopt() error: %d", error_number));
  }

  if (connect(fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    close(fd);
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("connect() error: %d", error_number));
  }
  absl::MutexLock l(&mu_);
  conn_ = std::make_unique<UnixSocketConnection>(fd);
  return absl::OkStatus();
}

absl::StatusOr<UnixSocketMessage> UnixSocketClient::MakeRequest(
    UnixSocketMessage msg) {
  absl::MutexLock l(&mu_);
  conn_->AddMessageToSend(std::move(msg));
  while (conn_->HasPendingMessageToSend()) {
    if (!conn_->Send()) {
      break;
    }
  }
  while (!conn_->HasNewMessageToRead()) {
    if (!conn_->Receive()) {
      int error_number = errno;

      // errno can be zero if the connection was closed.
      if (error_number == 0) {
        error_number = EIO;
      }
      return absl::ErrnoToStatus(
          errno, absl::StrFormat("receive() error: %d", error_number));
    }
  }
  return conn_->ReadMessage();
}

}  // namespace tcpdirect

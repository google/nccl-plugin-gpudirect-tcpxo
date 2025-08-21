/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_COMMON_UNIX_SOCKET_CONNECTION_H_
#define BUFFER_MGMT_DAEMON_COMMON_UNIX_SOCKET_CONNECTION_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <memory>
#include <queue>
#include <string>
#include <utility>

#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"

namespace tcpdirect {

enum SendStatus {
  DONE = 0,
  IN_PROGRESS = 1,
  STOPPED = 2,
  ERROR = 3,
};

class UnixSocketConnection {
 public:
  explicit UnixSocketConnection(int fd);
  ~UnixSocketConnection();
  // return false if error occurs
  bool Receive();
  // return false if error occurs
  bool Send();
  bool HasNewMessageToRead() { return !incoming_.empty(); }
  bool HasPendingMessageToSend() { return !outgoing_.empty(); }
  void AddMessageToSend(UnixSocketMessage&& message) {
    outgoing_.emplace(std::move(message));
  }
  UnixSocketMessage ReadMessage();

 private:
  void SendText(const std::string& text, SendStatus* status, int fd = -1);
  void SendFd(int fd, SendStatus* status);
  enum State {
    LENGTH = 0,
    PAYLOAD = 1,
  };
  int fd_{-1};
  State read_state_{LENGTH};
  uint16_t read_length_;
  size_t read_offset_{0};
  std::unique_ptr<char[]> read_buffer_;
  State send_state_{LENGTH};
  uint16_t send_length_;
  uint16_t send_length_network_order_;
  size_t send_offset_{0};
  char* send_buffer_;
  std::queue<UnixSocketMessage> incoming_;
  std::queue<UnixSocketMessage> outgoing_;
  struct msghdr send_msg_;
  struct iovec send_iov_;
  char send_control_[CMSG_SPACE(sizeof(int))];
  uint16_t send_dummy_len_{0};
  struct msghdr recv_msg_;
  struct iovec recv_iov_;
  char recv_control_[CMSG_SPACE(sizeof(int))];
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_COMMON_UNIX_SOCKET_CONNECTION_H_

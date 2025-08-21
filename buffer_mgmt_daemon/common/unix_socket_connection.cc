/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/common/unix_socket_connection.h"

#include <arpa/inet.h>
#include <stdint.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <cassert>
#include <string>
#include <utility>

#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"

namespace tcpdirect {

UnixSocketConnection::UnixSocketConnection(int fd) : fd_(fd) {
  read_length_ = sizeof(uint16_t);
  read_buffer_.reset(new char[read_length_]);
}

UnixSocketConnection::~UnixSocketConnection() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

bool UnixSocketConnection::Receive() {
  memset(&recv_msg_, 0, sizeof(recv_msg_));

  // Setup IO vector
  recv_iov_.iov_base = read_buffer_.get() + read_offset_;
  recv_iov_.iov_len = read_length_ - read_offset_;
  recv_msg_.msg_iov = &recv_iov_;
  recv_msg_.msg_iovlen = 1;
  recv_msg_.msg_control = &recv_control_;
  recv_msg_.msg_controllen = sizeof(recv_control_);

  int bytes_read = recvmsg(fd_, &recv_msg_, 0);
  int fd = -1;

  // Getting 0 from blocking recvmsg could only mean that the connection
  // is closed by remote peer.
  if (bytes_read <= 0) return false;

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&recv_msg_);

  if (cmsg != nullptr && cmsg->cmsg_len == CMSG_LEN(sizeof(int)) &&
      cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
    fd = *((int*)CMSG_DATA(cmsg));
  }

  read_offset_ += bytes_read;

  if (read_offset_ == read_length_) {
    switch (read_state_) {
      case LENGTH: {
        read_length_ = ntohs(*((uint16_t*)read_buffer_.get()));
        /* No payload, only fds */
        if (read_length_ == 0) {
          read_length_ = sizeof(uint16_t);
          UnixSocketMessage message;
          if (fd > 0) message.set_fd(fd);
          incoming_.emplace(std::move(message));
          break;
        }
        // Ignore and close the fd when the current message is LENGTH
        if (fd > 0) {
          close(fd);
        }
        read_state_ = PAYLOAD;
        break;
      }
      case PAYLOAD: {
        std::string payload;
        payload.reserve(read_length_);
        for (int i = 0; i < read_length_; ++i) {
          payload.push_back(read_buffer_[i]);
        }
        UnixSocketMessage message;
        if (fd > 0) message.set_fd(fd);
        message.set_text(std::move(payload));
        incoming_.emplace(std::move(message));
        read_length_ = sizeof(uint16_t);
        read_state_ = LENGTH;
        break;
      }
      default:
        assert(false && "bad read_state_");
    }
    read_buffer_.reset(new char[read_length_]);
    read_offset_ = 0;
  }
  return true;
}

void UnixSocketConnection::SendFd(int fd, SendStatus* status) {
  memset(&send_msg_, 0, sizeof(send_msg_));
  send_msg_.msg_control = &send_control_;
  send_msg_.msg_controllen = sizeof(send_control_);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&send_msg_);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  *((int*)CMSG_DATA(cmsg)) = fd;

  send_iov_.iov_base = &send_dummy_len_;
  send_iov_.iov_len = sizeof(send_dummy_len_);
  send_msg_.msg_iov = &send_iov_;
  send_msg_.msg_iovlen = 1;

  if (sendmsg(fd_, &send_msg_, 0) < 0) {
    *status = ERROR;
  }
  *status = DONE;
}

void UnixSocketConnection::SendText(const std::string& text, SendStatus* status,
                                    int fd) {
  memset(&send_msg_, 0, sizeof(send_msg_));
  if (send_state_ == LENGTH && send_offset_ == 0) {
    send_length_ = sizeof(uint16_t);
    send_length_network_order_ = htons((uint16_t)text.size());
    send_buffer_ = (char*)&send_length_network_order_;
  }
  // Setup IO vector
  send_iov_.iov_base = send_buffer_ + send_offset_;
  send_iov_.iov_len = send_length_ - send_offset_;
  send_msg_.msg_iov = &send_iov_;
  send_msg_.msg_iovlen = 1;

  if (fd > 0) {
    send_msg_.msg_control = &send_control_;
    send_msg_.msg_controllen = sizeof(send_control_);
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&send_msg_);
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    *((int*)CMSG_DATA(cmsg)) = fd;
  }

  int bytes_sent = send_length_ == 0 ? 0 : sendmsg(fd_, &send_msg_, 0);

  if (bytes_sent < 0) {
    *status = ERROR;
    return;
  }
  send_offset_ += bytes_sent;
  if (send_offset_ != send_length_) {
    // Send operation is not finished, break the loop here and wait for the
    // next chance.
    *status = STOPPED;
    return;
  }
  // Send operation completes.  Move forward to next state or next message.
  switch (send_state_) {
    case LENGTH: {
      send_length_ = (uint16_t)text.size();
      send_buffer_ = (char*)text.data();
      send_state_ = PAYLOAD;
      *status = IN_PROGRESS;
      break;
    }
    case PAYLOAD: {
      send_length_ = sizeof(uint16_t);
      send_buffer_ = (char*)&send_length_network_order_;
      send_state_ = LENGTH;
      *status = DONE;
      break;
    }
    default:
      assert(false && "bad send_state_");
  }
  send_offset_ = 0;
}

bool UnixSocketConnection::Send() {
  while (!outgoing_.empty()) {
    UnixSocketMessage& outmsg = outgoing_.front();

    SendStatus status = ERROR;

    if (outmsg.has_text()) {
      if (outmsg.has_fd() && outmsg.fd() > 0)
        SendText(outmsg.text(), &status, outmsg.fd());
      else
        SendText(outmsg.text(), &status);
    } else if (outmsg.has_fd()) {
      SendFd(outmsg.fd(), &status);
    }

    if (status == ERROR) {
      return false;
    } else if (status == DONE) {
      outgoing_.pop();
    } else if (status == STOPPED) {
      return true;
    }
  }
  return true;
}

UnixSocketMessage UnixSocketConnection::ReadMessage() {
  if (incoming_.empty()) {
    return {};
  }
  UnixSocketMessage ret = std::move(incoming_.front());
  incoming_.pop();
  return ret;
}

}  // namespace tcpdirect

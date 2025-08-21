/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/unix_socket_server.h"

#include <arpa/inet.h>
#include <errno.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "buffer_mgmt_daemon/common/uds_helpers.h"
#include "buffer_mgmt_daemon/common/unix_socket_connection.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "dxs/client/oss/status_macros.h"
#include "dxs/client/thread-shim.h"

namespace tcpdirect {

namespace {
constexpr int kDefaultBacklog{128};

}  // namespace

UnixSocketServer::UnixSocketServer(std::string path,
                                   ServiceFunc service_handler,
                                   CleanupFunc cleanup_handler)
    : path_(std::move(path)),
      service_handler_(std::move(service_handler)),
      cleanup_handler_(std::move(cleanup_handler)) {
  LOG(INFO) << "Creating UnixSocketServer to listen to path: " << path_;
}

UnixSocketServer::~UnixSocketServer() { Stop(); }

absl::Status UnixSocketServer::Start() {
  LOG(INFO) << "Starting UnixSocketServer to listen to path: " << path_;
  ASSIGN_OR_RETURN(sockaddr_un server_addr, UdsSockaddr(path_));

  // Remove existing socket for file-based UDS.
  if (server_addr.sun_path[0] != 0 && unlink(server_addr.sun_path)) {
    if (errno != ENOENT) {
      return absl::ErrnoToStatus(errno,
                                 absl::StrFormat("unlink() error: %d", errno));
    }
  }

  if (service_handler_ == nullptr) {
    return absl::InvalidArgumentError("Missing service handler.");
  }

  listener_socket_ = socket(AF_UNIX, SOCK_STREAM, 0);

  if (listener_socket_ < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("socket() error: %d", error_number));
  }

  if (bind(listener_socket_, (struct sockaddr*)&server_addr,
           sizeof(server_addr)) != 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("bind() error: %d", error_number));
  }

  if (listen(listener_socket_, kDefaultBacklog) != 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("listen() error: %d", error_number));
  }

  running_.store(true);

  epoll_fd_ = epoll_create1(/*flags=*/0);
  if (epoll_fd_ < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("epoll_create1() error: %d", error_number));
  }

  if (RegisterEvents(listener_socket_, EPOLLIN) < 0) {
    int error_number = errno;
    return absl::ErrnoToStatus(
        errno, absl::StrFormat("epoll_ctl() error: %d", error_number));
  }

  // Internal //thread library does not like characters other than
  // A-z,a-z,0-9,_,- in the thread name. Replacing ':' and '.' with '_' in
  // server path to retain thread identification by server path.
  std::string thread_server_path =
      absl::StrReplaceAll(path_, {{":", "_"}, {".", "_"}});

  event_thread_ =
      dxs::NewThreadShim([this] { EventLoop(); }, thread_server_path);

  return absl::OkStatus();
}

void UnixSocketServer::Stop() {
  running_.store(false);

  // Trigger event thread join.
  event_thread_.reset();

  // Remove all connected clients, closing all existing UDS connections
  connected_clients_.clear();
  if (listener_socket_ >= 0) {
    close(listener_socket_);
  }
  LOG(INFO) << "Stopping UnixSocketServer that listens to path: " << path_;
}

int UnixSocketServer::RegisterEvents(int fd, uint32_t events) {
  struct epoll_event event;
  event.events = events;
  event.data.fd = fd;
  return epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &event);
}

int UnixSocketServer::UnregisterFd(int fd) {
  return epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
}

void UnixSocketServer::EventLoop() {
  while (running_) {
    std::vector<struct epoll_event> events(connected_clients_.size() + 1);
    // We are using level-triggered epoll event distribution.
    int nevents = epoll_wait(epoll_fd_, events.data(), events.size(),
                             std::chrono::milliseconds(100).count());
    // EINTR does not indicate any fatal errors.
    if (nevents < 0 && errno != EINTR) {
      LOG(ERROR) << "epoll_wait() error: " << errno;
      continue;
    }
    for (int i = 0; i < nevents; ++i) {
      const struct epoll_event& event = events[i];
      if (event.data.fd == listener_socket_) {
        HandleListener(event.events);
      } else {
        HandleClient(event.data.fd, event.events);
      }
    }
  }
}

void UnixSocketServer::HandleListener(uint32_t events) {
  if (events & EPOLLERR) {
    int error_number = errno;
    LOG(ERROR) << "Listener socket error, errno: " << error_number;
    running_.store(false);
    return;
  }
  if (events & EPOLLIN) {
    struct sockaddr_un peer_addr;
    unsigned int peer_addr_len = 0;
    LOG(INFO) << "Received EPOLLIN, accepting socket...";
    int socket = accept4(listener_socket_, (struct sockaddr*)&peer_addr,
                         &peer_addr_len, 0);
    if (socket < 0) {
      LOG(ERROR) << "accept4() error: " << static_cast<int>(errno);
      return;
    }
    LOG(INFO) << "Accepted socket: " << socket;
    LOG(INFO) << "Creating socket handler for accepted clients...";
    connected_clients_[socket] = std::make_unique<UnixSocketConnection>(socket);
    RegisterEvents(socket, EPOLLIN);
  }
}

void UnixSocketServer::HandleClient(int client, uint32_t events) {
  UnixSocketConnection& connection = *connected_clients_[client];
  bool fin{false};
  if (events & EPOLLIN) {
    if (connection.Receive()) {
      if (connection.HasNewMessageToRead()) {
        UnixSocketMessage response;
        service_handler_(client, connection.ReadMessage(), &response, &fin);
        connection.AddMessageToSend(std::move(response));
        if (!connection.Send()) {
          fin = true;
        }
      }
    } else {
      fin = true;
    }
  }
  if ((events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) || fin) {
    if (events & EPOLLERR) {
      LOG(ERROR) << "Client socket on EPOLLERR";
    }
    if (cleanup_handler_) {
      cleanup_handler_(client);
    }

    RemoveClient(client);
    return;
  }
}

void UnixSocketServer::RemoveClient(int client_socket) {
  UnregisterFd(client_socket);
  connected_clients_.erase(client_socket);
}

}  // namespace tcpdirect

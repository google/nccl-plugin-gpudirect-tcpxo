/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_UNIX_SOCKET_SERVER_H_
#define BUFFER_MGMT_DAEMON_UNIX_SOCKET_SERVER_H_

#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/un.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "buffer_mgmt_daemon/common/unix_socket_connection.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "dxs/client/thread-shim.h"

namespace tcpdirect {

// UnixSocketServer implements a request/response server over unix domain
// sockets. This class uses abstract sockets to avoid user namespace conflicts.
//
class UnixSocketServer {
  using ServiceFunc =
      std::function<void(int, UnixSocketMessage&&, UnixSocketMessage*, bool*)>;
  using CleanupFunc = std::function<void(int)>;

 public:
  UnixSocketServer(std::string path, ServiceFunc service_handler,
                   CleanupFunc cleanup_handler = nullptr);
  ~UnixSocketServer();
  absl::Status Start();
  void Stop();

 private:
  int RegisterEvents(int fd, uint32_t events);
  int UnregisterFd(int fd);
  void EventLoop();
  void HandleListener(uint32_t events);
  void HandleClient(int client_socket, uint32_t events);
  void RemoveClient(int client_socket);

  std::string path_;
  ServiceFunc service_handler_{nullptr};
  CleanupFunc cleanup_handler_{nullptr};

  std::atomic<bool> running_{false};
  std::unique_ptr<dxs::ThreadShim> event_thread_{nullptr};
  int listener_socket_{-1};
  int epoll_fd_{-1};
  absl::flat_hash_map<int, std::unique_ptr<UnixSocketConnection>>
      connected_clients_;
};
}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_UNIX_SOCKET_SERVER_H_

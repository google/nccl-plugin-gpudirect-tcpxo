/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef NETWORK_SYSTEM_CALL_H__
#define NETWORK_SYSTEM_CALL_H__

#include <netdb.h>
#include <stddef.h>
#include <sys/socket.h>
#include <unistd.h>

#include "net_util/network_system_call_interface.h"

namespace net_util {

class NetworkSystemCall : public NetworkSystemCallInterface {
 public:
  NetworkSystemCall() = default;
  NetworkSystemCall(const NetworkSystemCall&) = delete;
  NetworkSystemCall(NetworkSystemCall&&) = delete;

  // Accept - accept a connection on a socket.
  // 'man 2 accept'
  int Accept(int socket_file_descriptor, struct sockaddr* address,
             socklen_t* address_length) override {
    return accept(socket_file_descriptor, address, address_length);
  }

  // Bind - bind a name to a socket.
  // 'man 2 bind'
  int Bind(int socket_file_descriptor, const struct sockaddr* my_address,
           socklen_t address_length) override {
    return bind(socket_file_descriptor, my_address, address_length);
  }

  // Close - close a file descriptor.
  // 'man 2 close'
  int Close(int file_descriptor) override { return close(file_descriptor); }

  // Shutdown - shut down part of a full-duplex connection.
  // 'man 2 shutdown'
  int Shutdown(int socket_file_descriptor, int how) override {
    return shutdown(socket_file_descriptor, how);
  }

  // GetAddressInformation, FreeAddressInformation - network address
  // and service translation.
  // 'man 3 getaddrinfo'
  void FreeAddressInformation(struct addrinfo* result) const override {
    freeaddrinfo(result);
  }
  int GetAddressInformation(const char* node, const char* service,
                            const struct addrinfo* hints,
                            struct addrinfo** result) override {
    return getaddrinfo(node, service, hints, result);
  }

  // GetSocketOption - get options on socket.
  // 'man 2 getsockopt'
  int GetSocketOption(int file_descriptor, int level, int option_name,
                      void* option_value, socklen_t* option_length) override {
    return getsockopt(file_descriptor, level, option_name, option_value,
                      option_length);
  }

  // Listen - listen for connections on a socket
  // 'man 2 listen'
  int Listen(int socket_file_descriptor, int backlog) override {
    return listen(socket_file_descriptor, backlog);
  }

  // Read - read from a file descriptor.
  // 'man 2 read'
  ssize_t Read(int file_descriptor, void* buffer, size_t count) override {
    return read(file_descriptor, buffer, count);
  }

  // SetSocketOption - get and set options on sockets.
  // 'man 2 setsockopt'
  int SetSocketOption(int file_descriptor, int level, int option_name,
                      const void* option_value,
                      socklen_t option_length) override {
    return setsockopt(file_descriptor, level, option_name, option_value,
                      option_length);
  }

  // Socket - create an endpoint for communication.
  int Socket(int domain, int type, int protocol) override {
    return socket(domain, type, protocol);
  }

  // Socketpair - create an unnamed pair of connected sockets.
  int Socketpair(int domain, int type, int protocol, int sv[2]) override {
    return socketpair(domain, type, protocol, sv);
  }

  // Write - write to a file descriptor
  // 'man 2 write'
  ssize_t Write(int file_descriptor, const void* buffer,
                size_t count) override {
    return write(file_descriptor, buffer, count);
  }

  // ReceiveFrom - receive a message from a socket
  // 'man 2 recvfrom'
  ssize_t ReceiveFrom(int socket_descriptor, void* buffer, size_t max_bytes,
                      int flags, struct sockaddr* from_addr,
                      socklen_t* from_length) override {
    return recvfrom(socket_descriptor, buffer, max_bytes, flags, from_addr,
                    from_length);
  }

  // Receive - receive a message from a socket
  // 'man 2 recv'
  ssize_t Receive(int socket_descriptor, void* buffer, size_t max_bytes,
                  int flags) override {
    return recv(socket_descriptor, buffer, max_bytes, flags);
  }

  // ReceiveMsg - receive a message from a socket
  // 'man 2 recvmsg'
  ssize_t ReceiveMsg(int socket_descriptor, struct msghdr* msg,
                     int flags) override {
    return recvmsg(socket_descriptor, msg, flags);
  }

  // SendTo - send a message on a socket
  // 'man 2 sendto'
  ssize_t SendTo(int socket_descriptor, const void* buffer, size_t num_bytes,
                 int flags, const struct sockaddr* to_addr,
                 socklen_t to_length) override {
    return sendto(socket_descriptor, buffer, num_bytes, flags, to_addr,
                  to_length);
  }

  ssize_t SendMsg(int socket_descriptor, const struct msghdr* msg,
                  int flags) override {
    return sendmsg(socket_descriptor, msg, flags);
  }

  // Connect - initiate a connection on a socket
  // 'man 2 connect'
  int Connect(int socket_descriptor, const struct sockaddr* serv_addr,
              socklen_t addrlen) override {
    return connect(socket_descriptor, serv_addr, addrlen);
  }

  // Indicates which of the specified file descriptors is ready for reading
  // (`read_fds`), writing (`write_fds`), or has an error (`except_fds`). If the
  // specified condition is false for all of them, select() blocks, up to the
  // specified `timeout`. See `man 2 select` for details.
  int Select(int nfds, fd_set* read_fds, fd_set* write_fds, fd_set* except_fds,
             struct timeval* timeout) override {
    return select(nfds, read_fds, write_fds, except_fds, timeout);
  }
};

}  // namespace net_util

#endif  // NETWORK_SYSTEM_CALL_H__

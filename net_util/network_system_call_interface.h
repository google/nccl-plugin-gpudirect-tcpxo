/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef NETWORK_SYSTEM_CALL_INTERFACE_H__
#define NETWORK_SYSTEM_CALL_INTERFACE_H__

#include <netdb.h>
#include <stddef.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>

namespace net_util {

class NetworkSystemCallInterface {
 public:
  // This type is neither copyable nor movable.
  NetworkSystemCallInterface(const NetworkSystemCallInterface&) = delete;
  NetworkSystemCallInterface& operator=(const NetworkSystemCallInterface&) =
      delete;

  virtual ~NetworkSystemCallInterface() = default;

  // Accept - accept a connection on a socket.
  // 'man 2 accept'
  virtual int Accept(int socket_file_descriptor, struct sockaddr* address,
                     socklen_t* address_length) = 0;

  // Bind - bind a name to a socket.
  // 'man 2 bind'
  virtual int Bind(int socket_file_descriptor,
                   const struct sockaddr* my_address,
                   socklen_t address_length) = 0;

  // Close - close a file descriptor.
  // 'man 2 close'
  virtual int Close(int file_descriptor) = 0;

  // Shutdown - shut down part of a full-duplex connection.
  // 'man 2 shutdown'
  virtual int Shutdown(int socket_file_descriptor, int how) = 0;

  // GetAddressInformation, FreeAddressInformation - network address
  // and service translation.
  // 'man 3 getaddrinfo'
  virtual void FreeAddressInformation(struct addrinfo* result) const = 0;
  virtual int GetAddressInformation(const char* node, const char* service,
                                    const struct addrinfo* hints,
                                    struct addrinfo** result) = 0;

  // GetSocketOption - get options on sockets.
  // 'man 2 getsockopt'
  virtual int GetSocketOption(int file_descriptor, int level, int option_name,
                              void* option_value, socklen_t* option_length) = 0;

  // Listen - listen for connections on a socket
  // 'man 2 listen'
  virtual int Listen(int socket_file_descriptor, int backlog) = 0;

  // Read - read from a file descriptor.
  // 'man 2 read'
  virtual ssize_t Read(int file_descriptor, void* buffer, size_t count) = 0;

  // SetSocketOption - get and set options on sockets.
  // 'man 2 setsockopt'
  virtual int SetSocketOption(int file_descriptor, int level, int option_name,
                              const void* option_value,
                              socklen_t option_length) = 0;

  // Socket - create an endpoint for communication.
  virtual int Socket(int domain, int type, int protocol) = 0;

  // Socketpair - create an unnamed pair of connected sockets.
  // 'man 2 socketpair'
  virtual int Socketpair(int domain, int type, int protocol, int sv[2]) = 0;

  // Write - write to a file descriptor
  // 'man 2 write'
  virtual ssize_t Write(int file_descriptor, const void* buffer,
                        size_t count) = 0;

  // ReceiveFrom - receive a message from a socket
  // 'man 2 recvfrom'
  virtual ssize_t ReceiveFrom(int socket_descriptor, void* buffer,
                              size_t max_bytes, int flags,
                              struct sockaddr* from_addr,
                              socklen_t* from_length) = 0;

  // Receive - receive a message from a socket
  // 'man 2 recv'
  virtual ssize_t Receive(int socket_descriptor, void* buffer, size_t max_bytes,
                          int flags) = 0;

  // ReceiveMsg - receive a message from a socket
  // 'man 2 recvmsg'
  virtual ssize_t ReceiveMsg(int socket_descriptor, struct msghdr* msg,
                             int flags) = 0;

  // SendTo - send a message on a socket
  // 'man 2 sendto'
  virtual ssize_t SendTo(int socket_descriptor, const void* buffer,
                         size_t num_bytes, int flags,
                         const struct sockaddr* to_addr,
                         socklen_t to_length) = 0;

  // SendMsg - send a message on a socket
  // 'man 2 sendmsg'
  virtual ssize_t SendMsg(int socket_descriptor, const struct msghdr* msg,
                          int flags) = 0;

  // Connect - initiate a connection on a socket
  // 'man 2 connect'
  virtual int Connect(int socket_descriptor, const struct sockaddr* serv_addr,
                      socklen_t addrlen) = 0;

  // Indicates which of the specified file descriptors is ready for reading
  // (`read_fds`), writing (`write_fds`), or has an error (`except_fds`). If the
  // specified condition is false for all of them, select() blocks, up to the
  // specified `timeout`. See `man 2 select` for details.
  virtual int Select(int nfds, fd_set* read_fds, fd_set* write_fds,
                     fd_set* except_fds, struct timeval* timeout) = 0;

 protected:
  NetworkSystemCallInterface() = default;
};

}  // namespace net_util

#endif  // NETWORK_SYSTEM_CALL_INTERFACE_H__

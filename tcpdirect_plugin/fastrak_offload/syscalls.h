/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_SYSCALLS_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_SYSCALLS_H_

#include <ifaddrs.h>

#include <cerrno>
#include <string_view>

#include "absl/status/status.h"
#include "net_util/network_system_call_interface.h"

namespace fastrak {

// Extensions of NetSysCalls for fastrak-specific syscalls.
class SysCallExtensionsInterface {
 public:
  SysCallExtensionsInterface() = default;
  virtual ~SysCallExtensionsInterface() = default;
  SysCallExtensionsInterface(const SysCallExtensionsInterface&) = delete;
  SysCallExtensionsInterface& operator=(const SysCallExtensionsInterface&) =
      delete;

  // Creates a linked list of structures describing the network interfaces of
  // the local system, and stores the address of the first item of the list in
  // *ifap.
  virtual int GetIfaddrs(ifaddrs** ifap) = 0;

  // Frees the memory allocation of `ifap` from `GetIfaddrs`.
  virtual void FreeIfaddrs(ifaddrs* ifa) = 0;
  // Gets the current address to which the socket `sockfd` is bound.
  // 'man 2 getsockname'
  virtual int GetSockname(int sockfd, sockaddr* addr, socklen_t* addrlen) = 0;
};

net_util::NetworkSystemCallInterface& NetSysCall();
SysCallExtensionsInterface& SysCallExtensions();

inline absl::Status SysCallResultToStatus(int result,
                                          std::string_view message) {
  if (result < 0) return absl::ErrnoToStatus(errno, message);
  return absl::OkStatus();
}

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_SYSCALLS_H_

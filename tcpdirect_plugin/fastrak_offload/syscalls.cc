/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/syscalls.h"

#include <utility>

#include "net_util/network_system_call.h"
#include "net_util/network_system_call_interface.h"

namespace fastrak {
namespace {

class SysCallExtensionsImpl : public SysCallExtensionsInterface {
 public:
  // Creates a linked list of structures describing the network interfaces of
  // the local system, and stores the address of the first item of the list in
  // *ifap. See `man 2 getifaddrs` for details.
  int GetIfaddrs(ifaddrs** ifap) override { return getifaddrs(ifap); }

  // Frees the memory allocation of `ifap` from `GetIfaddrs`. See `man 2
  // freeifaddrs` for details.
  void FreeIfaddrs(ifaddrs* ifa) override { freeifaddrs(ifa); }

  // Gets the current address to which the socket `sockfd` is bound.
  // 'man 2 getsockname'
  int GetSockname(int sockfd, sockaddr* addr, socklen_t* addrlen) override {
    return getsockname(sockfd, addr, addrlen);
  }
};

net_util::NetworkSystemCallInterface*& NetSysCallSlot() {
  static net_util::NetworkSystemCallInterface* slot =
      new net_util::NetworkSystemCall();
  return slot;
}

SysCallExtensionsInterface*& SysCallExtensionsSlot() {
  static SysCallExtensionsInterface* slot = new SysCallExtensionsImpl();
  return slot;
}

}  // namespace

net_util::NetworkSystemCallInterface& NetSysCall() { return *NetSysCallSlot(); }
net_util::NetworkSystemCallInterface& TestonlyExchangeNetSysCall(
    net_util::NetworkSystemCallInterface& other) {
  return *std::exchange(NetSysCallSlot(), &other);
}

SysCallExtensionsInterface& SysCallExtensions() {
  return *SysCallExtensionsSlot();
}
SysCallExtensionsInterface& TestonlyExchangeSysCallExtensions(
    SysCallExtensionsInterface& other) {
  return *std::exchange(SysCallExtensionsSlot(), &other);
}

}  // namespace fastrak

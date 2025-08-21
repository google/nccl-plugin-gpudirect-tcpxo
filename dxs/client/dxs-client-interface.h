/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_DXS_CLIENT_INTERFACE_H_
#define DXS_CLIENT_DXS_CLIENT_INTERFACE_H_

#include <stdint.h>
#include <sys/uio.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dxs/client/base-interface.h"
#include "dxs/client/control-command.h"
#include "dxs/client/control-command.pb.h"
#include "dxs/client/dxs-client-types.h"
#include "dxs/client/monotonic-timestamp.h"

namespace dxs {

class OpInterface : public BaseInterface {
 public:
  // Gets the op id of the op.
  virtual OpId GetOpId() const = 0;

  // Gets the op completion timestamp in the
  // CLOCK_MONOTONIC domain.
  // Only valid if Test()==1
  virtual MonotonicTs GetCompletionTime() const = 0;
};

// An outstanding read operation on a data socket.
// Must not outlive the client which created it.
// Thread compatible, usually only accessed from a single thread.
class RecvOpInterface : public OpInterface {
 public:
  // Tests if this has completed. Returns nullopt if not complete. If
  // successful, returns iovecs.
  virtual std::optional<absl::StatusOr<std::vector<iovec>>> Test() = 0;

  // Closes resources associated with the op.
  // Returns 0 on success or < 0 on error.
  virtual absl::Status Release() = 0;
};

// An outstanding server side linearized read operation on a data socket.
// Must not outlive the client which created it.
// Thread compatible, usually only accessed from a single thread.
class LinearizedRecvOpInterface : public OpInterface {
 public:
  // Tests if this has completed. Returns nullopt if not complete. If
  // successful, returns size.
  virtual std::optional<absl::StatusOr<uint64_t>> Test() = 0;
};

// An outstanding send operation on a data socket.
// Must not outlive the client which created it.
// Thread compatible, usually only accessed from a single thread.
class SendOpInterface : public OpInterface {
 public:
  // Tests if this has completed. Returns nullopt if not complete.
  virtual std::optional<absl::Status> Test() = 0;
};

class DataSocketInterface : public BaseInterface {
 public:
  // Returns nullopt if not yet connected, else the connect status.
  virtual std::optional<absl::Status> SocketReady() = 0;
};

class ConnectedSocketInterface : public DataSocketInterface {
 public:
  // Returns the address of the peer.
  virtual WireSocketAddr Peer() const = 0;
};

// An open data socket for sending data.
// Must not outlive the client which created it.
// Thread compatible, usually only accessed from a single thread.
class SendSocketInterface : public ConnectedSocketInterface {
 public:
  // Returns the dxs address this is writing to.
  virtual std::string Address() const = 0;

  // Sends data. Must not be called until SocketReady returns kConnected.
  virtual absl::StatusOr<std::unique_ptr<SendOpInterface>> Send(
      uint64_t offset, uint64_t size, Reg reg_handle) = 0;
};

// An open data socket for receiving data.
// Must not outlive the client which created it.
// Thread compatible, usually only accessed from a single thread.
class RecvSocketInterface : public ConnectedSocketInterface {
 public:
  // Receives data which is linearized by the server side.
  virtual absl::StatusOr<std::unique_ptr<LinearizedRecvOpInterface>>
  RecvLinearized(uint64_t offset, uint64_t size, Reg reg_handle) = 0;
};

class ListenBaseInterface : public DataSocketInterface {
 public:
  // Returns the port dxs is listening on. Returns -1 before the socket is
  // ready.
  virtual int Port() const = 0;

  // Returns the dxs address this is listening on.
  virtual std::string Address() const = 0;
};

// A socket listening for connections.
// Must not outlive the client which created it.
// Thread compatible, usually only accessed from a single thread.
class ListenSocketInterface : public ListenBaseInterface {
 public:
  // Accept a new connection on a listening socket. 'data_sock' is not valid
  // until SocketReady() returns OkStatus. Returns nullptr if there is no
  // incoming connection.
  virtual absl::StatusOr<absl_nullable std::unique_ptr<RecvSocketInterface>>
  Accept() = 0;
};

// A client for managing DXS connections.
class DxsClientInterface {
 public:
  virtual ~DxsClientInterface() = default;

  virtual absl::Status Shutdown(absl::Duration timeout) = 0;

  /*** Listen Socket API ***/

  // Listens for new connections. 'listen_sock' is not valid until
  // SocketReady(listen_sock) returns 1. Do not call Accept(listen_sock, ...)
  // before then.
  virtual absl::StatusOr<std::unique_ptr<ListenSocketInterface>> Listen() = 0;

  /*** Data Socket API ***/

  // Connects to the specified destination address and port. 'data_sock' is not
  // valid until SocketReady(data_sock) returns 1.
  virtual absl::StatusOr<std::unique_ptr<SendSocketInterface>> Connect(
      std::string addr, uint16_t port) = 0;

  /*** Misc ***/
  // Pings the server and returns the RTT on success.
  virtual absl::StatusOr<absl::Duration> Ping() = 0;

  virtual absl::string_view GetServerBuildId() = 0;
  virtual uint64_t GetServerVersion() = 0;
};

// An internal interface for DXS client for socket usage. Thread safe.
class SocketRegistryInterface {
 public:
  virtual ~SocketRegistryInterface() = default;

  // Register a SendOp that will be updated with future control messages.
  virtual std::unique_ptr<SendOpInterface> RegisterSendOp(OpId id) = 0;

  // Register a LinearizedRecvOp that will be updated with future control
  // messages.
  virtual std::unique_ptr<LinearizedRecvOpInterface> RegisterLinearizedRecvOp(
      DataSocketHandle handle, OpId id, uint64_t size) = 0;

  // Register a RecvSocket that will be updated with future control messages.
  virtual std::unique_ptr<RecvSocketInterface> RegisterRecvSocket(
      DataSocketHandle handle, WireSocketAddr peer) = 0;
};

// A client for managing Buffers on GPUs directly. Thread safe.
class BufferManagerInterface {
 public:
  virtual ~BufferManagerInterface() = default;

  virtual absl::StatusOr<Reg> RegBuffer(absl::Span<const iovec> gpas) = 0;
  virtual absl::Status DeregBuffer(Reg reg_handle) = 0;

  virtual bool HealthCheck() const = 0;

  virtual absl::string_view GetServerBuildId() = 0;
  virtual uint64_t GetServerVersion() = 0;
};

}  // namespace dxs

#endif  // DXS_CLIENT_DXS_CLIENT_INTERFACE_H_

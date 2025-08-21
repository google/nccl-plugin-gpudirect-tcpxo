/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_WIRE_VERSION_H_
#define DXS_CLIENT_WIRE_VERSION_H_

#include <cstdint>

namespace dxs {

// DXS clients and engines must maintain control channel wire compatibility.
// This file describes the version and constraints. kWireVersion should be
// updated any time that the client<->engine protocol changes. If a change is
// made that is not backwards compatible, kMinimimWireVersion must be updated.

// Current version of the wire protocol.
static constexpr uint64_t kWireVersion = 7;

// Minimum acceptable version for this client or engine.
static constexpr uint64_t kMinimumWireVersion = 1;

// This is the wire version that support for AcceptAck messages was introduced.
static constexpr uint64_t kAcceptAckVersion = 4;

// This is the wire version that support for SubscribeToPeriodicStats messages
// was introduced.
static constexpr uint64_t kSubscribeToPeriodicStatsVersion = 5;

// This is the wire version that support for receiving kPing & respond with
// kPong from client.
static constexpr uint64_t kHandlePingVersion = 6;

// This is the minimum wire version that supports sending peer socket
// address via IncomingConnectionMessageV2.
static constexpr uint64_t kPeerSocketAddressVersion = 7;

}  // namespace dxs

#endif  // DXS_CLIENT_WIRE_VERSION_H_

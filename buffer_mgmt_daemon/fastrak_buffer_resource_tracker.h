/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_BUFFER_RESOURCE_TRACKER_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_BUFFER_RESOURCE_TRACKER_H_

#include <cstdint>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "dxs/client/dxs-client-types.h"

namespace tcpdirect {

class FastrakBufferResourceTracker {
 public:
  using RegHandle = uint64_t;
  using DmabufId = uint64_t;

  FastrakBufferResourceTracker() = default;
  FastrakBufferResourceTracker(const FastrakBufferResourceTracker&) = delete;
  FastrakBufferResourceTracker& operator=(const FastrakBufferResourceTracker&) =
      delete;
  ~FastrakBufferResourceTracker();

  // Track a buffer for the given client.
  absl::Status TrackBuffer(int client, int dmabuf_fd, DmabufId dmabuf_id,
                           dxs::Reg reg_handle);

  // Get the dmabuf id of the registered buffer
  absl::StatusOr<DmabufId> GetDmabufId(int client, dxs::Reg reg_handle);

  // Get reg handles of all buffers of the given client. Return an error if the
  // client is not found.
  absl::StatusOr<std::vector<dxs::Reg>> GetRegHandles(int client);

  // Return all clients registered with the resource tracker
  std::vector<int> GetClients();

  // Register the client with the resource tracker. Operations will fail if the
  // corresponding client is not registered. This function call is idempotent.
  // Registering an existed client has no side effect.
  void RegisterClient(int client);

  // Erase client when the current socket connection is not active.
  void UnregisterClient(int client);

  // Untrack the given buffer from the client. If all buffers of this client are
  // untracked, the client will be automatically deregistered.
  absl::Status UntrackBuffer(int client, dxs::Reg reg_handle);

 private:
  struct BufferInfo {
    int dmabuf_fd;
    DmabufId dmabuf_id;
  };
  absl::Mutex mutex_;
  absl::flat_hash_map<int, absl::flat_hash_map<RegHandle, BufferInfo>>
      client_reg_handles_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_FASTRAK_BUFFER_RESOURCE_TRACKER_H_

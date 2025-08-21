/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/fastrak_buffer_resource_tracker.h"

#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "dxs/client/dxs-client-types.h"

namespace tcpdirect {

FastrakBufferResourceTracker::~FastrakBufferResourceTracker() {
  absl::MutexLock lock(&mutex_);
  for (const auto& [client, reg_dmabufs] : client_reg_handles_) {
    for (const auto& [reg_handle, dmabuf_info] : reg_dmabufs) {
      if (close(dmabuf_info.dmabuf_fd) != 0) {
        LOG(ERROR) << "Failed to close dmabuf fd " << dmabuf_info.dmabuf_fd
                   << ", errno = " << errno;
      }
    }
  }
}

absl::Status FastrakBufferResourceTracker::TrackBuffer(int client,
                                                       int dmabuf_fd,
                                                       DmabufId dmabuf_id,
                                                       dxs::Reg reg_handle) {
  absl::MutexLock lock(&mutex_);
  if (!client_reg_handles_.contains(client)) {
    return absl::InternalError(
        absl::StrFormat("Client %d not found in resource tracker", client));
  }
  auto& reg_dmabufs = client_reg_handles_[client];
  if (reg_dmabufs.contains(reg_handle)) {
    return absl::InternalError("Reg handle already exists in resource tracker");
  }
  reg_dmabufs[reg_handle] = BufferInfo{dmabuf_fd, dmabuf_id};
  VLOG(1) << "Client: " << client << ", tracked buffer: " << dmabuf_fd
          << ", dmabuf id: " << dmabuf_id << ", reg handle: " << reg_handle;
  return absl::OkStatus();
}

absl::StatusOr<FastrakBufferResourceTracker::DmabufId>
FastrakBufferResourceTracker::GetDmabufId(int client, dxs::Reg reg_handle) {
  absl::MutexLock lock(&mutex_);
  if (!client_reg_handles_.contains(client)) {
    return absl::InternalError(
        absl::StrFormat("Client %d not found in fetching dma buffer id for %d",
                        client, reg_handle));
  }
  auto& reg_dmabufs = client_reg_handles_[client];
  if (!reg_dmabufs.contains(reg_handle)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid registration handle: %" PRIu64 ".", reg_handle));
  }
  return reg_dmabufs[reg_handle].dmabuf_id;
}

absl::StatusOr<std::vector<dxs::Reg>>
FastrakBufferResourceTracker::GetRegHandles(int client) {
  absl::MutexLock lock(&mutex_);
  if (!client_reg_handles_.contains(client)) {
    return absl::NotFoundError(
        absl::StrFormat("Client %d not found in resource tracker", client));
  }

  std::vector<dxs::Reg> reg_handles;
  auto& reg_dmabufs = client_reg_handles_[client];

  for (const auto& entry : reg_dmabufs) {
    reg_handles.push_back(entry.first);
  }
  return reg_handles;
}

std::vector<int> FastrakBufferResourceTracker::GetClients() {
  absl::MutexLock lock(&mutex_);
  std::vector<int> clients;
  for (const auto& entry : client_reg_handles_) {
    clients.push_back(entry.first);
  }
  return clients;
}

void FastrakBufferResourceTracker::RegisterClient(int client) {
  absl::MutexLock lock(&mutex_);
  if (!client_reg_handles_.contains(client)) {
    client_reg_handles_[client] = {};
  }
}

void FastrakBufferResourceTracker::UnregisterClient(int client) {
  absl::MutexLock lock(&mutex_);
  client_reg_handles_.erase(client);
}

absl::Status FastrakBufferResourceTracker::UntrackBuffer(int client,
                                                         dxs::Reg reg_handle) {
  absl::MutexLock lock(&mutex_);
  if (!client_reg_handles_.contains(client)) {
    return absl::InternalError(absl::StrFormat(
        "Client %d not found in untracking reg buffer %d", client, reg_handle));
  }
  auto& reg_dmabufs = client_reg_handles_[client];
  if (!reg_dmabufs.contains(reg_handle)) {
    return absl::InternalError(
        absl::StrFormat("Reg handle %d not found in client %d's reg buffers",
                        reg_handle, client));
  }
  auto& dma_buffer_info = reg_dmabufs[reg_handle];

  VLOG(1) << "Client: " << client
          << ", untracked buffer: " << dma_buffer_info.dmabuf_fd
          << ", dmabuf id: " << dma_buffer_info.dmabuf_id
          << ", reg handle: " << reg_handle;

  // Close the fd
  if (close(dma_buffer_info.dmabuf_fd) != 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to close dmabuf fd %d, errno = %d",
                        dma_buffer_info.dmabuf_fd, errno));
  }

  reg_dmabufs.erase(reg_handle);

  return absl::OkStatus();
}
}  // namespace tcpdirect

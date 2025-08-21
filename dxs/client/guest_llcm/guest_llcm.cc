/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dxs/client/guest_llcm/guest_llcm.h"

#include <sys/fcntl.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "dxs/client/oss/status_macros.h"

namespace dxs {

namespace {
absl::StatusOr<off_t> GetFileSize(int fd) {
  struct stat statbuf{};
  int err = fstat(fd, &statbuf);
  if (err) {
    return absl::ErrnoToStatus(errno, "Failed to get file stat");
  }
  return statbuf.st_size;
}
}  // namespace

absl::StatusOr<std::unique_ptr<GuestLlcm>> GuestLlcm::Create(
    absl::string_view domain_bdf, uint64_t llcm_size,
    uint64_t reverse_llcm_size, absl::string_view llcm_device_directory) {
  std::string resource0_path =
      absl::StrJoin({llcm_device_directory, domain_bdf, "resource0_wc"}, "/");
  std::string resource2_path =
      absl::StrJoin({llcm_device_directory, domain_bdf, "resource2_wc"}, "/");
  int fd_0 = open(resource0_path.c_str(), O_RDWR);
  if (fd_0 < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Failed to open resource0: ", resource0_path));
  }
  absl::Cleanup cleanup_0 = [fd_0] { close(fd_0); };

  int fd_2 = open(resource2_path.c_str(), O_RDWR);
  if (fd_2 < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Failed to open resource2: ", resource2_path));
  }
  absl::Cleanup cleanup_2 = [fd_2] { close(fd_2); };

  ASSIGN_OR_RETURN(off_t resource0_size, GetFileSize(fd_0));
  if (std::cmp_less(resource0_size, llcm_size)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "resource0 too small, want %llu, got %lld", llcm_size, resource0_size));
  }
  ASSIGN_OR_RETURN(off_t resource2_size, GetFileSize(fd_2));
  if (std::cmp_less(resource2_size, reverse_llcm_size)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("resource2 too small, want %llu, got %lld",
                        reverse_llcm_size, resource2_size));
  }

  void* llcm = mmap(nullptr, llcm_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd_0, /*offset=*/0);
  if (llcm == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "Failed to mmap Llcm");
  }
  std::unique_ptr<void, MemmapDeleter> llcm_unique(llcm);
  void* reverse_llcm =
      mmap(nullptr, reverse_llcm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_2,
           /*offset=*/0);
  if (reverse_llcm == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "Failed to mmap Reverse Llcm");
  }
  std::unique_ptr<void, MemmapDeleter> reverse_llcm_unique(reverse_llcm);

  // The file descriptors are not moved and will be closed. The mmaped memory
  // should stay valid even the fd is closed.
  return absl::WrapUnique(
      new GuestLlcm(std::move(llcm_unique), std::move(reverse_llcm_unique)));
}

}  // namespace dxs

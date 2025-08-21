/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include <ng-log/logging.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "base/init_google.h"
#include "base/strtoint.h"
#include "dxs/client/guest_llcm/guest_llcm.h"
#include "dxs/client/oss/barrier.h"
#include "dxs/client/oss/mmio.h"
#include "dxs/client/oss/status_macros.h"

// Usage: ./guest_llcm_tool -uid= -logtostderr <cmd1> <cmd2> ...
// cmd: <a|r>:<1|2|4|8>:<offset>[=val]
//      e.g. read uint64 from offset 0x400 of llcm: a:8:0x400
//      e.g. write uint64 0x55aa to offset 8 of llcm: a:8:8=0x55aa

ABSL_FLAG(std::string, dom_bdf, "0000:00:07.0", "domain:bus:device.function");
ABSL_FLAG(std::string, llcm_device_directory, "/dev/llcm_devices",
          "The llcm device directory to mount the llcm device in "
          "workload container.");
ABSL_FLAG(uint64_t, llcm_size, 4096, "");
ABSL_FLAG(uint64_t, reverse_llcm_size, 4096, "");

namespace dxs {

namespace {
struct Operation {
  bool is_llcm;
  int bytes;  // 1,2,4,8
  uint64_t offset;
  std::optional<uint64_t> new_value;
};

absl::StatusOr<uint64_t> ParseInt(const std::string& str) {
  char* end_ptr = nullptr;
  uint64_t val = strto64(str.c_str(), &end_ptr, 0);
  if (end_ptr != str.data() + str.size()) {
    return absl::InvalidArgumentError(absl::StrCat("failed to parse: ", str));
  }
  return val;
}

}  // namespace

absl::Status Run(int argc, char* argv[]) {
  // Open devices
  ASSIGN_OR_RETURN(
      auto guest_llcm,
      GuestLlcm::Create(absl::GetFlag(FLAGS_dom_bdf),
                        absl::GetFlag(FLAGS_llcm_size),
                        absl::GetFlag(FLAGS_reverse_llcm_size),
                        absl::GetFlag(FLAGS_llcm_device_directory)));
  absl::Span<uint8_t> llcm = guest_llcm->llcm();
  absl::Span<uint8_t> reverse_llcm = guest_llcm->reverse_llcm();
  LOG(INFO) << absl::StrFormat("Llcm: addr=%p, len=%llu", llcm.data(),
                               llcm.size());
  LOG(INFO) << absl::StrFormat("ReverseLlcm: addr=%p, len=%llu",
                               reverse_llcm.data(), reverse_llcm.size());

  // Parse and verify all the commands
  std::vector<Operation> operations;
  for (int i = 1; i < argc; ++i) {
    std::string cmd = argv[i];
    LOG(INFO) << absl::StrFormat("cmd #%d is %s", i, cmd);

    std::vector<std::string> parts = absl::StrSplit(cmd, ':');
    if (parts.size() != 3) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid command: ", cmd));
    }

    std::vector<std::string> offset_val = absl::StrSplit(parts[2], '=');
    if (offset_val.size() != 1 && offset_val.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid command: ", cmd));
    }

    bool is_llcm;
    int bytes;
    uint64_t offset;
    std::optional<uint64_t> new_value = std::nullopt;

    if (parts[0] == "a") {
      is_llcm = true;
    } else if (parts[0] == "r") {
      is_llcm = false;
    } else {
      return absl::InvalidArgumentError(absl::StrCat("Invalid command: ", cmd));
    }

    ASSIGN_OR_RETURN(bytes, ParseInt(parts[1]));
    if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid command: ", cmd));
    }

    ASSIGN_OR_RETURN(offset, ParseInt(offset_val[0]));
    uint64_t region_size = is_llcm ? llcm.size() : reverse_llcm.size();
    if (offset + bytes > region_size) {
      return absl::OutOfRangeError(absl::StrCat("Invalid command: ", cmd));
    }

    if (offset_val.size() == 2) {
      ASSIGN_OR_RETURN(new_value, ParseInt(offset_val[1]));
    }

    operations.emplace_back(is_llcm, bytes, offset, new_value);
  }

  if (operations.empty()) {
    LOG(INFO) << "No cmd specified";
    return absl::OkStatus();
  }

  // Execute commands
  for (const Operation& op : operations) {
    uint8_t* addr =
        (op.is_llcm ? llcm.data() : reverse_llcm.data()) + op.offset;
    std::string msg_region = op.is_llcm ? "llcm" : "reverse_llcm";
    std::string msg_type = op.bytes == 1   ? "uint8_t"
                           : op.bytes == 2 ? "uint16_t"
                           : op.bytes == 4 ? "uint32_t"
                                           : "uint64_t";

    if (op.new_value.has_value()) {
      if (op.bytes == 1) {
        platforms_util::MmioWriteRelease8(addr, *op.new_value);
      } else if (op.bytes == 2) {
        platforms_util::MmioWriteRelease16(reinterpret_cast<uint16_t*>(addr),
                                           *op.new_value);
      } else if (op.bytes == 4) {
        platforms_util::MmioWriteRelease32(reinterpret_cast<uint32_t*>(addr),
                                           *op.new_value);
      } else if (op.bytes == 8) {
        platforms_util::MmioWriteRelease64(reinterpret_cast<uint64_t*>(addr),
                                           *op.new_value);
      } else {
        return absl::InternalError("invalid bytes");
      }
      platforms_util::MmioWriteBarrier();
      LOG(INFO) << absl::StrFormat("*(%s*)(%s+%#x) <= %#x", msg_type,
                                   msg_region, op.offset, *op.new_value);
    } else {
      uint64_t val;
      if (op.bytes == 1) {
        val = platforms_util::MmioReadAcquire8(addr);
      } else if (op.bytes == 2) {
        val = platforms_util::MmioReadAcquire16(
            reinterpret_cast<uint16_t*>(addr));
      } else if (op.bytes == 4) {
        val = platforms_util::MmioReadAcquire32(
            reinterpret_cast<uint32_t*>(addr));
      } else if (op.bytes == 8) {
        val = platforms_util::MmioReadAcquire64(
            reinterpret_cast<uint64_t*>(addr));
      } else {
        return absl::InternalError("invalid bytes");
      }
      LOG(INFO) << absl::StrFormat("*(%s*)(%s+%#x) == %#x", msg_type,
                                   msg_region, op.offset, val);
    }
  }

  return absl::OkStatus();
}
}  // namespace dxs

int main(int argc, char* argv[]) {
  nglog::InitializeLogging(argv[0], &argc, &argv, true);
  absl::Status status = dxs::Run(argc, argv);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }
  return 0;
}

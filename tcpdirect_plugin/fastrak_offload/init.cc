/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "tcpdirect_plugin/fastrak_offload/init.h"

#include <unistd.h>

#include <atomic>
#include <sstream>
#include <string>

#include "absl/base/call_once.h"
#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client.h"
#include "nccl.h"
#include "nccl_common.h"
#include "tcpdirect_plugin/fastrak_offload/common.h"
#include "tcpdirect_plugin/fastrak_offload/fastrak_plugin_version.h"
#include "tcpdirect_plugin/fastrak_offload/macro.h"
#include "tcpdirect_plugin/fastrak_offload/params.h"

namespace fastrak {
namespace {

absl::once_flag once;
absl::StatusCode init_state = absl::StatusCode::kOk;

class AbslLogSink : public absl::LogSink {
  void Send(const absl::LogEntry& entry) override {
    ncclDebugLogLevel level = entry.log_severity() == absl::LogSeverity::kInfo
                                  ? NCCL_LOG_INFO
                                  : NCCL_LOG_WARN;
    // For warnings, log to all subsystems.
    ncclDebugLogSubSys sub_sys =
        entry.log_severity() == absl::LogSeverity::kInfo ? NCCL_NET : NCCL_ALL;
    // The NCCL logger uses a fixed 1024 byte buffer which can result
    // in truncation when large multi-line logs are printed (stats dump),
    // so print lines separately.
    std::istringstream stream(absl::StrFormat(
        "NET/FasTrak %s: %s", absl::FormatTime(entry.timestamp()),
        entry.text_message()));
    std::string line;
    while (std::getline(stream, line)) {
      if (line.size() > 512) {
        line.resize(512);  // Leave space for NCCL-added prefixes.
        line.append(" (line truncated)");  // Brings it to a little over 512 len
      }
      // The NCCL logger is a C style printf-like function that expects
      // null terminated strings and appends its own new line.
      GetNcclLogFunc()(level, sub_sys,
                       std::string(entry.source_filename()).c_str(),
                       entry.source_line(), line.c_str());
    }
  }
};

absl::Status WaitForRxDM() {
  // Do not wait for RxDM if no timeout is specified.
  if (kFastrakRxDMInitTimeout == 0) return absl::OkStatus();

  absl::Status status = tcpdirect::rxdm_running();
  for (int i = 0; i < kFastrakRxDMInitTimeout; ++i) {
    if (status.ok()) {
      LOG(INFO) << "RxDM ready";
      return status;
    }
    absl::SleepFor(absl::Seconds(1));
    LOG(INFO) << absl::StrFormat(
        "RxDM not ready after %d seconds (status: %s), retrying... (timeout "
        "%d s)",
        i, status.ToString(), kFastrakRxDMInitTimeout);
    status = tcpdirect::rxdm_running();
  }

  char hostname[HOST_NAME_MAX + 1] = {0};
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    absl::SNPrintF(hostname, sizeof(hostname), "<unknown>");
  }
  hostname[HOST_NAME_MAX] = '\0';
  LOG(WARNING) << absl::StrFormat(
      "Timeout: RxDM not ready on %s after %d seconds: %s", hostname,
      kFastrakRxDMInitTimeout, status.ToString());

  return status;
}

std::atomic_flag disable_log_init;

void InitOnce(ncclDebugLogger_t logger) {
  // Logger must always be set first because other
  // initializations may attempt to print logs.
  SetNcclLogFunc(logger);

  if (!disable_log_init.test_and_set()) {
    // Align with DXS: use PST time (falls in the time zone America/Los_Angeles)
    absl::TimeZone tz = absl::LocalTimeZone();
    LOG(INFO) << "Current time zone: " << tz.name();
    absl::InitializeLog();

    // All logs should only go out of the sink to the NCCL logger.
    // NCCL has its own mechanism for choosing to direct to stdout or not.
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);

    absl::AddLogSink(new AbslLogSink());
  }

  LOG(INFO) << "Initializing network FasTrak, version: " << kFastrakPluginMajor
            << "." << kFastrakPluginMinor << "." << kFastrakPluginPatch;

  InitParams();

  if (!WaitForRxDM().ok()) {
    init_state = absl::StatusCode::kDeadlineExceeded;
    return;
  }

  init_state = initializeNetIfs() == ncclSuccess ? absl::StatusCode::kOk
                                                 : absl::StatusCode::kInternal;
}

}  // namespace

void DisableLogInit() { disable_log_init.test_and_set(); }

absl::Status PluginCoreInit(ncclDebugLogger_t logger) {
  absl::call_once(once, InitOnce, logger);
  return absl::Status(init_state, "");
}

}  // namespace fastrak

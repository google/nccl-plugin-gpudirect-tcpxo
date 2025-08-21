/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include <ng-log/logging.h>
#include <stdio.h>
#include <stdlib.h>

#include "absl/base/log_severity.h"
#include "absl/base/no_destructor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
namespace tcpdirect {

constexpr int kLogFileSizeMB = 10;

class InitLogSink : public absl::LogSink {
  void Send(const absl::LogEntry& entry) override {
    if (entry.log_severity() == absl::LogSeverity::kFatal) {
      LOG(FATAL) << entry.text_message();
    } else if (entry.log_severity() == absl::LogSeverity::kError) {
      LOG(ERROR) << entry.text_message();
    } else if (entry.log_severity() == absl::LogSeverity::kWarning) {
      LOG(WARNING) << entry.text_message();
    } else if (entry.log_severity() == absl::LogSeverity::kInfo) {
      LOG(INFO) << entry.text_message();
    }
  }

  void Flush() override { nglog::FlushLogFiles(nglog::NGLOG_INFO); }
};

void InitRxdm(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  FLAGS_max_log_size = kLogFileSizeMB;  // in MB
  nglog::InitializeLogging(argv[0]);
  static absl::NoDestructor<InitLogSink> init_log_sink;
  absl::AddLogSink(init_log_sink.get());
  absl::InitializeLog();
}

}  // namespace tcpdirect

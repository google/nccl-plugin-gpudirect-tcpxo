/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "buffer_mgmt_daemon/fastrak_gpumem_manager_startup.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "buffer_mgmt_daemon/clock.h"
#include "buffer_mgmt_daemon/clock_interface.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_interface.h"

namespace tcpdirect {

int RunCloudGpuMemManager(std::unique_ptr<FasTrakGpuMemManagerInterface> server,
                          absl::Duration wait_time_before_exit,
                          std::unique_ptr<ClockInterface> clock_override) {
  auto clock = clock_override == nullptr ? std::make_unique<Clock>()
                                         : std::move(clock_override);
  if (server == nullptr) {
    clock->SleepFor(wait_time_before_exit);
    LOG(FATAL) << "No GPU Mem Manager created, exiting.";
    return 1;
  }

  auto init_status = server->Init();
  if (init_status != 0) {
    clock->SleepFor(wait_time_before_exit);
    LOG(ERROR) << "Initialization failed with result:" << init_status;
    return init_status;
  }

  auto run_status = server->Run();
  // Sleep for a while before exiting, regardless of whether Run() succeeded or
  // not.
  clock->SleepFor(wait_time_before_exit);
  if (run_status != 0) {
    LOG(ERROR) << "Exiting with result:" << run_status;
    return run_status;
  }
  return 0;
}

}  // namespace tcpdirect

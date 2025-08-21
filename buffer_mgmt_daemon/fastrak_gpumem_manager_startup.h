/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_STARTUP_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_STARTUP_H_

#include <memory>

#include "absl/time/time.h"
#include "buffer_mgmt_daemon/clock_interface.h"
#include "buffer_mgmt_daemon/fastrak_gpumem_manager_interface.h"

namespace tcpdirect {

int RunCloudGpuMemManager(
    std::unique_ptr<FasTrakGpuMemManagerInterface> server,
    absl::Duration wait_time_before_exit,
    std::unique_ptr<ClockInterface> clock_override = nullptr);

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_STARTUP_H_

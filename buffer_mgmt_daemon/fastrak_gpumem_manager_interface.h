/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_INTERFACE_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_INTERFACE_H_

namespace tcpdirect {
// The interface for the FasTrak GPU Mem Manager.
class FasTrakGpuMemManagerInterface {
 public:
  virtual ~FasTrakGpuMemManagerInterface() = default;
  virtual int Init() = 0;
  virtual int Run() = 0;
};
}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_FASTRAK_GPUMEM_MANAGER_INTERFACE_H_

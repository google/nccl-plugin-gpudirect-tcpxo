/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_GPU_IP_SERVER_H_
#define BUFFER_MGMT_DAEMON_GPU_IP_SERVER_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "buffer_mgmt_daemon/buffer_manager_service_interface.h"
#include "buffer_mgmt_daemon/gpu_rxq_configurator_interface.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "buffer_mgmt_daemon/unix_socket_server.h"

namespace tcpdirect {

using tcpdirect::GpuRxqConfiguration;

/* Runs UDS servers for GPU-to-NIC IP lookup */
class GpuIpServer : public BufferManagerServiceInterface {
 public:
  explicit GpuIpServer(
      absl::Span<const tcpdirect::GpuRxqConfiguration> configs);
  absl::Status Initialize() override;
  absl::Status Start() override;
  ~GpuIpServer() override;

 private:
  const std::vector<GpuRxqConfiguration> configs_;
  std::vector<std::unique_ptr<tcpdirect::UnixSocketServer>> nic_services_;
};

}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_GPU_IP_SERVER_H_

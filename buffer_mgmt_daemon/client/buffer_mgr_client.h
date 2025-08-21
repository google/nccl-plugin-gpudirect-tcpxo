/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_CLIENT_BUFFER_MGR_CLIENT_H_
#define BUFFER_MGMT_DAEMON_CLIENT_BUFFER_MGR_CLIENT_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "buffer_mgmt_daemon/client/buffer_mgr_client-interface.h"
#include "buffer_mgmt_daemon/client/unix_socket_client.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "dxs/client/dxs-client-types.h"

namespace tcpdirect {

class BufferManagerClient : public BufferManagerClientInterface {
 public:
  static absl::StatusOr<std::unique_ptr<BufferManagerClientInterface>> Create(
      absl::string_view server_ip_addr);

  absl::StatusOr<dxs::Reg> RegBuf(int fd, size_t size) override;
  absl::Status DeregBuf(dxs::Reg handle) override;

 private:
  BufferManagerClient(absl::string_view server_ip_addr,
                      std::unique_ptr<UnixSocketClient> buf_op_client);
  absl::StatusOr<BufferOpResp> ParseBufferOpResp(
      const UnixSocketMessage& ret_msg);
  const std::string server_ip_addr_;
  std::unique_ptr<UnixSocketClient> buf_op_client_;
};

// Empty string on error, IPV4/V6 address of the corresponding NIC on success
std::string get_nic_ip_by_gpu_pci(absl::string_view gpu_pci);

// Returns the full mapping of GPU to closest NICs
std::optional<GetNicMappingResp> get_nic_mapping();

// Returns ok status if RxDM is running and ready to accept connections.
// This call establishes a connection, so should not be used in performance
// critical scenarios.
absl::Status rxdm_running();

}  // namespace tcpdirect
#endif  // BUFFER_MGMT_DAEMON_CLIENT_BUFFER_MGR_CLIENT_H_

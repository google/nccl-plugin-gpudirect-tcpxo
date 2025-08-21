/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_FASTRAK_GPU_MEM_IMPORTER_H_
#define BUFFER_MGMT_DAEMON_FASTRAK_GPU_MEM_IMPORTER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "buffer_mgmt_daemon/addr_translator_interface.h"
#include "buffer_mgmt_daemon/buffer_manager_service_interface.h"
#include "buffer_mgmt_daemon/fastrak_buffer_resource_tracker.h"
#include "buffer_mgmt_daemon/fastrak_gpu_nic_info.h"
#include "buffer_mgmt_daemon/proto/tcpfastrak_buffer_mgmt_message.pb.h"
#include "buffer_mgmt_daemon/unix_socket_server.h"
#include "dxs/client/dxs-client-interface.h"

namespace tcpdirect {
class FasTrakGpuMemImporter : public BufferManagerServiceInterface {
 public:
  struct NicBinding {
    std::string ip_addr;
    std::string nic_pci_addr;
    std::unique_ptr<AddrTranslatorInterface> addr_translator;
    dxs::BufferManagerInterface* dxs_client;
    std::unique_ptr<FastrakBufferResourceTracker> resource_tracker;
  };
  explicit FasTrakGpuMemImporter(
      absl::Span<const tcpdirect::FasTrakGpuNicInfo> nic_infos,
      std::function<void()> all_clients_exited_callback = nullptr,
      std::optional<std::string> dmabuf_import_path = std::nullopt);
  explicit FasTrakGpuMemImporter(std::vector<NicBinding>& bindings);

  // This type is neither copyable nor movable.
  FasTrakGpuMemImporter(const FasTrakGpuMemImporter&) = delete;
  FasTrakGpuMemImporter& operator=(const FasTrakGpuMemImporter&) = delete;

  absl::Status Initialize() override;
  absl::Status Start() override;
  ~FasTrakGpuMemImporter() override;

 private:
  absl::Status InitializeServers();
  // Service handler for client buffer reg/dereg requests.
  static void HandleRequest(int client, NicBinding& binding,
                            tcpdirect::UnixSocketMessage&& request,
                            tcpdirect::UnixSocketMessage* response, bool* fin);
  static absl::Status HandleRegBuffer(int client, NicBinding& binding,
                                      tcpdirect::UnixSocketMessage& request,
                                      tcpdirect::BufferOpResp* resp, bool* fin);
  static absl::Status HandleDeregBuffer(int client, NicBinding& binding,
                                        tcpdirect::BufferOpReq& request,
                                        tcpdirect::BufferOpResp* resp,
                                        bool* fin);
  static void CleanUp(int client, NicBinding& binding);

  // Keep tracks new clients and left clients
  void AddConnectedClient(int client, NicBinding& binding);
  void RemoveConnectedClient(int client, NicBinding& binding);

  struct NicService {
    NicBinding binding;
    std::unique_ptr<tcpdirect::UnixSocketServer> unix_socket_server;
  };
  std::vector<NicService> nic_services_;
  absl::Mutex mutex_;
  absl::flat_hash_set<int> connected_clients_ ABSL_GUARDED_BY(mutex_);
  std::function<void()> all_clients_exited_callback_;
};

}  // namespace tcpdirect
#endif  // BUFFER_MGMT_DAEMON_FASTRAK_GPU_MEM_IMPORTER_H_

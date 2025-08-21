/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DMABUF_BRIDGE_DMABUF_IMPORTER_H_
#define DMABUF_BRIDGE_DMABUF_IMPORTER_H_

#if __cplusplus
extern "C" {
#endif

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/ioctl.h>
#include <stdbool.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

struct dmabuf_importer_ctx;

struct dmabuf_importer_buf {
  struct dmabuf_importer_ctx* ctx;
  int attach_handle;
  unsigned int num_iovecs;
};

struct dmabuf_importer_map_req {
  int dmabuf_fd;
  int pci_dev_domain;
  int pci_dev_bus;
  int pci_dev_device;
  int pci_dev_func;
  int attach_handle;
};

int dmabuf_importer_create_ctx(const char* dmabuf_import_path,
                               struct dmabuf_importer_ctx** ctx);
int dmabuf_importer_map(struct dmabuf_importer_ctx* ctx,
                        struct dmabuf_importer_map_req* map_req,
                        struct dmabuf_importer_buf* buf);
int dmabuf_importer_unmap(struct dmabuf_importer_buf* buf);
// iovecs must be at least buf->num_iovecs long. Returns true on success.
bool dmabuf_importer_get_iovecs(struct dmabuf_importer_buf* buf,
                                struct iovec* iovecs);
void dmabuf_importer_destroy_ctx(struct dmabuf_importer_ctx* ctx);
#if __cplusplus
}
#endif

#endif  // DMABUF_BRIDGE_DMABUF_IMPORTER_H_

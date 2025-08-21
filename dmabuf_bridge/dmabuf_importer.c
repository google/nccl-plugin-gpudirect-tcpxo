/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include "dmabuf_bridge/dmabuf_importer.h"

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

#include "dmabuf_bridge/dmabuf_import_helper.h"

static const char dmabuf_import_dev_path[] = "/dev/dmabuf_import_helper";

struct dmabuf_importer_ctx {
  int dev_fd;
};

int dmabuf_importer_create_ctx(const char* dmabuf_import_path,
                               struct dmabuf_importer_ctx** ctx) {
  // The path might be different depending on environment, give caller
  // opportunity to provide path.
  if (!dmabuf_import_path) {
    dmabuf_import_path = dmabuf_import_dev_path;
  }

  int dev_fd = open(dmabuf_import_path, O_RDWR);
  if (dev_fd < 0) {
    return -1;
  }

  *ctx = malloc(sizeof(struct dmabuf_importer_ctx));
  if (!ctx) {
    close(dev_fd);
    errno = ENOMEM;
    return -1;
  }
  (*ctx)->dev_fd = dev_fd;
  return 0;
}

int dmabuf_importer_map(struct dmabuf_importer_ctx* ctx,
                        struct dmabuf_importer_map_req* map_req,
                        struct dmabuf_importer_buf* buf) {
  int ret;
  struct import_helper_map param = {
      .attach_handle = map_req->attach_handle,
      .fd = map_req->dmabuf_fd,
      .iovecs_count = 0,
  };

  param.dbdf[0] = map_req->pci_dev_domain;
  param.dbdf[1] = map_req->pci_dev_bus;
  param.dbdf[2] = map_req->pci_dev_device;
  param.dbdf[3] = map_req->pci_dev_func;

  ret = ioctl(ctx->dev_fd, IMPORT_HELPER_MAP, &param);
  if (ret < 0) {
    return -1;
  }

  buf->ctx = ctx;
  buf->attach_handle = param.attach_handle;
  buf->num_iovecs = param.iovecs_count;
  return 0;
}

int dmabuf_importer_unmap(struct dmabuf_importer_buf* buf) {
  int ret;

  struct import_helper_unmap param = {.attach_handle = buf->attach_handle};

  ret = ioctl(buf->ctx->dev_fd, IMPORT_HELPER_UNMAP, &param);
  if (ret < 0) return -1;

  return 0;
}

bool dmabuf_importer_get_iovecs(struct dmabuf_importer_buf* buf,
                                struct iovec* iovecs) {
  uint32_t offset = 0;
  while (offset < buf->num_iovecs) {
    struct import_helper_get_iovecs param = {
        .attach_handle = buf->attach_handle,
        .offset = offset,
    };
    if (ioctl(buf->ctx->dev_fd, IMPORT_HELPER_GET_IOVECS, &param) < 0) {
      return false;
    }
    uint32_t size = param.num_valid_iovecs;
    assert(size + offset <= buf->num_iovecs);
    memcpy(iovecs + offset, &param.iovecs, size * sizeof(struct iovec));
    offset += size;
    assert(offset <= buf->num_iovecs);
  }
  return true;
}

void dmabuf_importer_destroy_ctx(struct dmabuf_importer_ctx* ctx) {
  close(ctx->dev_fd);
  free(ctx);
}

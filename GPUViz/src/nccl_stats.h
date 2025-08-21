/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef GPUVIZ_SRC_NCCL_STATS_H_
#define GPUVIZ_SRC_NCCL_STATS_H_

#include <arpa/inet.h>
#include <stdint.h>

#include "tcpdirect_plugin/nccl_compat/nccl_net_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  EntityNVLConnection,
  EntityPCIConnection,
  EntityTCPConnection,
  EntityRDMAConnection,
  EntityProfilerPluginConnection
} ncclStatsConnectionType;

typedef enum { NetPlugin, CollNetPlugin, ProfilerPlugin } ncclStatsPluginType;

typedef union {
  uintptr_t net_plugin;
  uintptr_t coll_net_plugin;
} ncclStatsPlugin;

typedef struct {
} ncclStatsNVLConnection;

typedef struct {
} ncclStatsPCIConnection;

typedef struct {
  struct sockaddr_storage local_endpoint;
  struct sockaddr_storage remote_endpoint;
} ncclStatsTCPConnection;

typedef struct {
  struct sockaddr_storage local_endpoint;
  uint32_t local_qpn;
  struct sockaddr_storage remote_endpoint;
  uint32_t remote_qpn;
} ncclStatsRDMAConnection;

typedef struct {
  struct sockaddr_storage local_endpoint;
  uint32_t local_rank;
  uint64_t local_comm_hash;
  uint32_t remote_or_root_rank;
  const char* collective_type;       // NULL if not known/applicable.
  const char* collective_algorithm;  // NULL if not known/applicable.
  const char* description;           // NULL if not known/applicable.
} ncclStatsProfilerPluginConnection;

typedef union {
  ncclStatsNVLConnection nvl_conn;
  ncclStatsPCIConnection pci_conn;
  ncclStatsTCPConnection tcp_conn;
  ncclStatsRDMAConnection rdma_conn;
  ncclStatsProfilerPluginConnection profiler_conn;
} ncclStatsConnection;

typedef struct {
  ncclStatsConnectionType conn_type;
  ncclStatsPluginType nccl_plugin_type;
  const char* nccl_plugin_name;
  ncclStatsPlugin nccl_plugin;
  const char* gpu_pci_addr;
  ncclStatsConnection connection;
} ncclStatsConnectionIdentifier;

typedef enum {
  ConnectionCloseLocalTerminate,   // Local terminated the connection due to end
                                   // of use
  ConnectionCloseRemoteTerminate,  // Remote terminated the connection due to
                                   // end of use
  ConnectionCloseLocalError,  // Connection closed due to error detected locally
  ConnectionCloseRemoteError,
  ConnectionCloseLocalTimeout,
  ConnectionCloseRemoteTimeout
} ncclStatsConnectionCloseType;

typedef enum {
  LatencySoftware,
  LatencyNetHW,
  LatencyRecvReady
} ncclStatsLatencyType;

typedef struct {
  ncclStatsLatencyType latency_type;
  uint64_t latency_in_nanoseconds;
} ncclStatsLatencyMeasurement;

typedef enum {
  OperationTypeChunkSend,
  OperationTypeChunkRecv,
  OperationTypeCollective,
  OperationTypeMsgSend,
  OperationTypeMsgRecv
} ncclStatsOpType;

typedef struct {
  ncclStatsOpType type;
  uint64_t collective_id;
  uint64_t op_id;
  uint64_t op_sz;  // msg size
  uint64_t op_start_time;
  uint32_t num_measurements;
  const ncclStatsLatencyMeasurement* measurements;
} ncclStatsOperationMetric;

typedef enum {
  SendLatencySWIdx = 0,
  RecvLatencySWIdx = 1,
  SendLatencyNetHWIdx = 2,
  RecvLatencyNetHWIdx = 3,
  RecvReadyLatencyIdx = 4,
  SendMessageSizeIdx = 5,
  RecvMessageSizeIdx = 6,
} ncclStatsDistributionIdx;

typedef enum {
  SendLatencySW = 1 << SendLatencySWIdx,
  RecvLatencySW = 1 << RecvLatencySWIdx,
  SendLatencyNetHW = 1 << SendLatencyNetHWIdx,
  RecvLatencyNetHW = 1 << RecvLatencyNetHWIdx,
  RecvReadyLatency = 1 << RecvReadyLatencyIdx,
  SendMessageSize = 1 << SendMessageSizeIdx,
  RecvMessageSize = 1 << RecvMessageSizeIdx,
  TotalDistributionType = 7,
} ncclStatsDistributionType;

typedef struct {
  // Name of the statistics collector
  const char* name;
  // Initialize a statistics object, input is a bitmap of distribution type to
  // collect, output is a handle to the created object
  ncclResult_t (*init)(ncclDebugLogger_t logFunction,
                       uint64_t distributionCollectorBitmap,
                       uintptr_t* statsGlobalHandle /* Out */);
  // Destroy a statistics object. Implicitly destroys all connections created
  // for the object to track
  ncclResult_t (*destroy)(uintptr_t statsGlobalHandle);
  // Notifies the statistics object about a new connection, output is a handle
  // to the statistics tracking object for this connection.
  ncclResult_t (*addConnection)(
      uintptr_t statsGlobalHandle,
      const ncclStatsConnectionIdentifier* connectionIdentifier,
      uintptr_t* statsConnectionHandle);
  // Indicates that a connection was closed, with a reason description
  ncclResult_t (*deleteConnection)(uintptr_t statsConnectionHandle,
                                   ncclStatsConnectionCloseType closeType,
                                   const char* verboseReason);
  // Notifies the statistics object about a measurement of a transaction over a
  // specific connection, must not be called from multiple threads for the same
  // connection handle at the same time (caller responsible for synchronization)
  ncclResult_t (*notifyOperationMeasurement)(
      uintptr_t statsConnectionHandle,
      const ncclStatsOperationMetric* measurement);
} ncclStatsPlugin_v1_t;

typedef ncclStatsPlugin_v1_t ncclStatsPlugin_t;

#ifdef __cplusplus
}
#endif

#endif  // GPUVIZ_SRC_NCCL_STATS_H_

/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

// All FasTrak plug-in environment variable params
// are defined here (and only here).
//
// The syntax is as follows:
// Global symbol name, env var name, default value, min value, max value
//
// Once defined here, a global int64_t reference representing
// the environment variable state as it was during initialization
// will be available throughout the plug-in and can be accessed
// without any performance penalty. If the environment changes
// during runtime, the value will not be updated (which is how
// it was in the original implementation as well).

#ifndef TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_CONST_PARAMS_H_
#define TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_CONST_PARAMS_H_

#include <cstdint>
#include <string_view>

#include "absl/container/flat_hash_map.h"

namespace fastrak {
namespace params_internal {

struct ParamData {
  int64_t min_value;
  int64_t max_value;
  int64_t* slot;
};

inline absl::flat_hash_map<std::string_view, ParamData>& GetParamDataMap() {
  static auto* map = new absl::flat_hash_map<std::string_view, ParamData>;
  return *map;
}

struct ParamRegistrar {
  ParamRegistrar(std::string_view name, int64_t min_value, int64_t max_value,
                 int64_t* slot) {
    GetParamDataMap()[name] = {min_value, max_value, slot};
  }
};

}  // namespace params_internal

#define NCCL_CONST_PARAM(name, env_var_name, default_value, min_value,   \
                         max_value)                                      \
  namespace params_internal {                                            \
  inline int64_t impl_##name = default_value;                            \
  inline params_internal::ParamRegistrar registrar_##name(env_var_name,  \
                                                          min_value,     \
                                                          max_value,     \
                                                          &impl_##name); \
  }                                                                      \
  inline const int64_t& name = params_internal::impl_##name;

// Global enable/disable flag for plug-in.
NCCL_CONST_PARAM(kEnableFasTrak, "FASTRAK_ENABLE", 1, 0, 1);

// Timeout threshold to finish the data transfer request from isend/irecv.
// Timeout enforces when the Test() shows in progress data transport
// after the timeout threshold.
// The plugin won't timeout if the NCCL invokes Test() after the timeout
// threshold and the data transport has finished.
// If set to 0, or FastrakPluginDisableTimekeeping is set,
// then no timeout is enforced.
NCCL_CONST_PARAM(kFastrakDataTransferTimeoutMs,
                 "FASTRAK_DATA_TRANSFER_TIMEOUT_MS", 2 * 60 * 60'000, 0,
                 86400000);  // 2 hours

// Initial threshold after which a warning message will be printed if a
// request is still pending. Each event causes this timeout to get
// multiplied by 2 to avoid polluting the log.
NCCL_CONST_PARAM(kFastrakDataTransferSlownessMs,
                 "FASTRAK_DATA_TRANSFER_SLOWNESS_MS", 5 * 60'000, 1, 86400000);

// Timeout threshold to finish the plugin connection.
// If set to 0, or FastrakPluginDisableTimekeeping is set,
// then no timeout is enforced.
NCCL_CONST_PARAM(kFastrakPluginConnectTimeoutMs,
                 "FASTRAK_PLUGIN_CONNECT_TIMEOUT_MS", 5 * 60'000, 0,
                 86400000);  // 5 Minutes
NCCL_CONST_PARAM(kFastrakPluginAcceptTimeoutMs,
                 "FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS", 15 * 60'000, 0,
                 86400000);  // 15 Minutes

// If enabled, logs will be printed even in the performance-critical
// code paths, provided NCCL_DEBUG is set to INFO.
// If disabled, then only the initialization and connection related
// logs are printed (including the distribution metrics on comm close).
NCCL_CONST_PARAM(kEnableHotpathLogging, "FASTRAK_ENABLE_HOTPATH_LOGGING", 0, 0,
                 1);

// Number of flows per NCCL-level connection.
constexpr int64_t kFastrakMaxNumFlows = 8;
NCCL_CONST_PARAM(kFastrakNumFlows, "FASTRAK_NUM_FLOWS", 2, 1,
                 kFastrakMaxNumFlows);

// If enabled, enforces net devices used on both ends of a NCCL connection
// to be on the same rail / within the same subnet. Enabled by default.
NCCL_CONST_PARAM(kNicRailAligned, "FASTRAK_RAIL_ALIGNED", 1, 0, 1);

// If disabled and Snap is used currently, Send sockets created by DXS clients
// will not be closed by DXS when they are done. Disabled by default.
NCCL_CONST_PARAM(kFastrakCloseSendOnDone, "FASTRAK_CLOSE_SEND_ON_DONE", 0, 0,
                 1);

// If enabled, the plugin caches memory regions that are already registered
// with DXS. If disabled, every piece of memory provided by NCCL will be
// considered as new and registered with DXS, regardless of whether it has
// already been registered before. Enabled by default.
NCCL_CONST_PARAM(kFastrakCacheMr, "FASTRAK_CACHE_MR", 1, 0, 1);

// If enabled, requests LLCM for FasTrak control path communication. Disabled by
// default.
NCCL_CONST_PARAM(kFastrakUseLlcm, "FASTRAK_USE_LLCM", 0, 0, 1);

NCCL_CONST_PARAM(kFastrakDumpCommStats, "FASTRAK_DUMP_COMM_STATS", 1, 0, 1);

// If enabled, sends out the stats metrics. Disabled by default.
NCCL_CONST_PARAM(kFastrakEnableSendStats, "FASTRAK_ENABLE_SEND_STATS", 0, 0, 1);

// Defines the interval to send stats after every N number of offload requests.
NCCL_CONST_PARAM(kFastrakSendStatsInterval, "FASTRAK_SEND_STATS_INTERVAL",
                 1000000, 1, 1000000000);

// Defines the maximum time (in seconds) to wait for RxDM to come
// online during init. If set to 0, then no pre-shim init check is performed.
NCCL_CONST_PARAM(kFastrakRxDMInitTimeout, "FASTRAK_RXDM_INIT_TIMEOUT", 30, 0,
                 86400);

// If enabled, fastrak does not require every GPU to find a closest NIC. Thereby
// allowing a subset of GPUs to communicate with the network through a subset of
// all available NICs. Note this requires setting other NCCL params such as
// NCCL_FASTRAK_IFNAME, NCCL_ALGO, NCCL_NET_GDR_LEVEL correctly.
NCCL_CONST_PARAM(kFastrakAllowNicSubset, "FASTRAK_ALLOW_NIC_SUBSET", 0, 0, 1);

// Manages telemetry exporting.
// 0: Disabled
// 1: GPUViz Write On Local Disk Only
// 3: GPUViz Upload Only
// 4: GPUViz Write On Local Disk and Upload
enum class TelemetryMode {
  kDisabled = 0,
  kGPUVizWriteOnLocalDiskOnly = 1,
  kGPUVizUploadOnly = 3,
  kGPUVizWriteOnLocalDiskAndUpload = 4
};
NCCL_CONST_PARAM(kFastrakPluginTelemetryMode, "NET_PLUGIN_TELEMETRY_MODE", 0, 0,
                 4);

// The DXS Prober requires users to scrape the connection info from the
// logs. This flag enables logging of the connection info without needing to
// set NCCL_DEBUG to INFO.
NCCL_CONST_PARAM(kFastrakLogConnectionInfo, "FASTRAK_LOG_CONNECTION_INFO", 0, 0,
                 1);

#undef NCCL_CONST_PARAM

}  // namespace fastrak

#endif  // TCPDIRECT_PLUGIN_FASTRAK_OFFLOAD_CONST_PARAMS_H_

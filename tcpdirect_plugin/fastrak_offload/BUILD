load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# BUILD file for TCPFasTrak NCCL interface - offload
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

# Dual-licensed, using the least restrictive
licenses(["notice"])

cc_library(
    name = "defs",
    hdrs = [
        "shared_defs.h",
    ],
)

cc_library(
    name = "macro",
    srcs = ["macro.cc"],
    hdrs = ["macro.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@nccl//:plugin_lib",
    ],
)

cc_library(
    name = "stats",
    srcs = ["stats.cc"],
    hdrs = ["stats.h"],
    deps = [
        ":nccl_params",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "fastrak_plugin_version",
    hdrs = ["fastrak_plugin_version.h"],
)

cc_library(
    name = "init",
    srcs = ["init.cc"],
    hdrs = ["init.h"],
    deps = [
        ":fastrak_common",
        ":fastrak_plugin_version",
        ":macro",
        ":nccl_params",
        "//buffer_mgmt_daemon/client:buffer_mgr_client",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@nccl//:plugin_lib",
    ],
)

cc_library(
    name = "request",
    hdrs = ["request.h"],
    deps = [
        ":profiler_interface",
        ":stats",
        "//dxs/client:dxs-client-interface",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "fastrak_common",
    srcs = ["common.cc"],
    hdrs = [
        "common.h",
        "utilities.h",
    ],
    deps = [
        ":macro",
        ":nccl_params",
        ":profiler_factory",
        ":profiler_interface",
        ":stats",
        ":syscalls",
        "//buffer_mgmt_daemon:pci_utils",
        "//buffer_mgmt_daemon/client:buffer_mgr_client",
        "//buffer_mgmt_daemon/client:buffer_mgr_client-interface",
        "//dxs/client:derive_dxs_address",
        "//dxs/client:dxs-client-interface",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "//nccl_cuda:cuda_defs",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@nccl//:plugin_lib",
    ],
)

cc_library(
    name = "nccl_params",
    srcs = [
        "const_params.h",
        "params.cc",
    ],
    hdrs = [
        "params.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "nic_client_router_interface",
    hdrs = [
        "nic_client_router_interface.h",
    ],
    deps = [
        "//buffer_mgmt_daemon/client:buffer_mgr_client-interface",
        "//dxs/client:dxs-client-interface",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "nic_client_router",
    srcs = [
        "nic_client_router.cc",
    ],
    hdrs = [
        "nic_client_router.h",
    ],
    deps = [
        ":nccl_params",
        ":nic_client_router_interface",
        "//buffer_mgmt_daemon/client:buffer_mgr_client",
        "//buffer_mgmt_daemon/client:buffer_mgr_client-interface",
        "//dxs/client:dxs-client",
        "//dxs/client:dxs-client-interface",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "nccl_shim",
    srcs = ["nccl_shim.cc"],
    hdrs = ["nccl_shim.h"],
    deps = [
        ":fastrak_common",
        ":nccl_params",
        ":nic_client_router",
        ":profiler_factory",
        ":profiler_interface",
        ":request",
        ":stats",
        ":syscalls",
        "//buffer_mgmt_daemon/client:buffer_mgr_client-interface",
        "//dxs/client:control-command",
        "//dxs/client:derive_dxs_address",
        "//dxs/client:dxs-client-interface",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "//nccl_cuda:cuda_common",
        "//nccl_cuda:cuda_defs",
        "//tcpdirect_plugin/nccl_compat:nccl_net_compat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "fastrak_plugin",
    srcs = [
        "fastrak_plugin.cc",
    ],
    linkopts = ["-Wl,--export-dynamic-symbol=ncclNetPlugin_v7"],
    deps = [
        ":fastrak_common",
        ":init",
        ":nccl_shim",
        ":nic_client_router",
        ":profiler_factory_gpuviz",
        ":request",
        "//dxs/client/oss:status_macros",
        "//nccl_cuda:cuda_common",
        "//nccl_cuda:cuda_defs",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@nccl//:plugin_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "syscalls",
    srcs = ["syscalls.cc"],
    hdrs = ["syscalls.h"],
    deps = [
        "//net_util:network_system_call_h",
        "//net_util:network_system_call_interface_h",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "profiler_interface",
    hdrs = ["profiler_interface.h"],
    deps = [
        "//dxs/client:base-interface",
        "@com_google_absl//absl/base:nullability",
    ],
)

cc_library(
    name = "profiler_noop",
    hdrs = ["profiler_noop.h"],
    deps = [
        ":fastrak_common",
        ":profiler_factory",
        ":profiler_interface",
        "@com_google_absl//absl/base:nullability",
    ],
)

cc_library(
    name = "profiler_gpuviz",
    srcs = ["profiler_gpuviz.cc"],
    hdrs = [
        "profiler_gpuviz.h",
    ],
    deps = [
        ":profiler_interface",
        "//GPUViz:gpuviz_lib",
        "//dxs/client/oss:status_macros",
        "//tcpdirect_plugin/nccl_compat:nccl_net_compat",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nccl//:plugin_lib",
    ],
)

cc_library(
    name = "profiler_factory",
    hdrs = ["profiler_factory.h"],
    deps = [
        ":profiler_interface",
        "//nccl_cuda:cuda_defs",
    ],
)

cc_library(
    name = "profiler_factory_cloud",
    srcs = ["profiler_factory_cloud.cc"],
    hdrs = ["profiler_factory.h"],
    deps = [
        ":profiler_factory",
        ":profiler_interface",
        ":profiler_noop",
        "//nccl_cuda:cuda_defs",
    ],
)

cc_library(
    name = "profiler_factory_gpuviz",
    srcs = ["profiler_factory_gpuviz.cc"],
    hdrs = [
        "profiler_factory.h",
        "profiler_factory_gpuviz.h",
    ],
    deps = [
        ":fastrak_common",
        ":macro",
        ":nccl_params",
        ":profiler_factory",
        ":profiler_gpuviz",
        ":profiler_interface",
        ":profiler_noop",
        "//GPUViz:gpuviz_lib",
        "//nccl_cuda:cuda_defs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@nccl//:plugin_lib",
    ],
)

cc_binary(
    name = "libnccl-net.so",
    features = ["-enable_relr"],
    linkshared = True,
    linkstatic = True,
    deps = [
        ":fastrak_plugin",
    ],
)

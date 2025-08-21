load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pci_utils",
    srcs = ["pci_utils.cc"],
    hdrs = ["pci_utils.h"],
    deps = [
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cuda_library(
    name = "cuda_logging",
    srcs = ["cuda_logging.cu"],
    hdrs = ["cuda_logging.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
        "@cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "gpu_rxq_configurator_interface",
    hdrs = ["gpu_rxq_configurator_interface.h"],
)

cc_library(
    name = "a3_gpu_rxq_configurator",
    srcs = ["a3_gpu_rxq_configurator.cc"],
    hdrs = ["a3_gpu_rxq_configurator.h"],
    deps = [
        ":gpu_rxq_configurator_interface",
        ":pci_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "addr_translator_interface",
    hdrs = ["addr_translator_interface.h"],
    deps = ["@com_google_absl//absl/status:statusor"],
)

cc_library(
    name = "fastrak_addr_translator",
    srcs = ["fastrak_addr_translator.cc"],
    hdrs = [
        "fastrak_addr_translator.h",
    ],
    deps = [
        ":addr_translator_interface",
        ":pci_utils",
        "//dmabuf_bridge:dmabuf_importer",
        "//dxs/client:coalesce-iovecs",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "gpu_ip_server",
    srcs = ["gpu_ip_server.cc"],
    hdrs = [
        "buffer_manager_service_interface.h",
        "gpu_ip_server.h",
    ],
    deps = [
        ":gpu_rxq_configurator_interface",
        ":status_utils",
        ":unix_socket_server",
        "//buffer_mgmt_daemon/common:uds_helpers",
        "//buffer_mgmt_daemon/proto:tcpfastrak_buffer_mgmt_message_cc_proto",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@googleapis//google/rpc:status_cc_proto",
    ],
)

cc_library(
    name = "fastrak_gpu_nic_info",
    srcs = ["fastrak_gpu_nic_info.cc"],
    hdrs = [
        "fastrak_gpu_nic_info.h",
        "gpu_rxq_configurator_interface.h",
    ],
    deps = [
        "//dxs/client:dxs-client",
        "//dxs/client:dxs-client-interface",
        "//dxs/client:dxs-client-types",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "fastrak_gpu_mem_importer",
    srcs = ["fastrak_gpu_mem_importer.cc"],
    hdrs = [
        "buffer_manager_service_interface.h",
        "fastrak_gpu_mem_importer.h",
    ],
    deps = [
        ":addr_translator_interface",
        ":fastrak_addr_translator",
        ":fastrak_buffer_resource_tracker",
        ":fastrak_gpu_nic_info",
        ":status_utils",
        ":unix_socket_server",
        "//buffer_mgmt_daemon/common:uds_helpers",
        "//buffer_mgmt_daemon/proto:tcpfastrak_buffer_mgmt_message_cc_proto",
        "//dxs/client:dxs-client-interface",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@googleapis//google/rpc:status_cc_proto",
    ],
)

cc_library(
    name = "unix_socket_server",
    srcs = ["unix_socket_server.cc"],
    hdrs = ["unix_socket_server.h"],
    deps = [
        "//buffer_mgmt_daemon/common:uds_helpers",
        "//buffer_mgmt_daemon/common:unix_socket_connection",
        "//buffer_mgmt_daemon/proto:tcpfastrak_buffer_mgmt_message_cc_proto",
        "//dxs/client:thread-shim",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "fastrak_gpumem_manager_interface",
    hdrs = ["fastrak_gpumem_manager_interface.h"],
)

cc_library(
    name = "fastrak_gpumem_manager_base_lib",
    srcs = ["fastrak_gpumem_manager.cc"],
    hdrs = ["fastrak_gpumem_manager_base.h"],
    deps = [
        ":a3_gpu_rxq_configurator",
        ":cuda_logging",
        ":fastrak_gpu_mem_importer",
        ":fastrak_gpu_nic_info",
        ":fastrak_gpumem_manager_host_interface",
        ":fastrak_gpumem_manager_interface",
        ":gpu_ip_server",
        ":gpu_rxq_configurator_interface",
        "//dxs/client:control_command_cc_proto",
        "//dxs/client:dxs-client",
        "//dxs/client:dxs-client-types",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@cuda_headers",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "buffer_manager_version",
    hdrs = ["buffer_manager_version.h"],
)

cc_library(
    name = "fastrak_gpumem_manager_startup",
    srcs = ["fastrak_gpumem_manager_startup.cc"],
    hdrs = ["fastrak_gpumem_manager_startup.h"],
    deps = [
        ":clock",
        ":clock_interface",
        ":fastrak_gpumem_manager_interface",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "fastrak_gpumem_manager_lib",
    srcs = [
        "main_cloud.cc",
    ],
    deps = [
        ":buffer_manager_version",
        ":fastrak_gpumem_manager_base_lib",
        ":fastrak_gpumem_manager_cloud",
        ":fastrak_gpumem_manager_startup",
        "//buffer_mgmt_daemon/oss:init",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:flags",
        "@com_google_absl//absl/time",
        "@com_google_tensorflow_serving//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@ecclesia//ecclesia/lib/http:server",
        "@local_cuda//:cuda",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_binary(
    name = "fastrak_gpumem_manager",
    linkstatic = True,
    deps = [
        ":fastrak_gpumem_manager_lib",
    ],
)

cc_binary(
    name = "mtest_fastrak_gpumem_manager",
    features = ["-enable_relr"],
    linkopts = ["-Wl,--dynamic-linker=/lib64/ld-linux-x86-64.so.2"],
    linkstatic = True,
    deps = [
        ":fastrak_gpumem_manager_lib",
        ":google_pthread_signal_safe_key_create",
    ],
)

cc_library(
    name = "clock_interface",
    hdrs = ["clock_interface.h"],
    deps = ["@com_google_absl//absl/time"],
)

cc_library(
    name = "clock",
    hdrs = ["clock.h"],
    deps = [
        ":clock_interface",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "fastrak_gpumem_manager_cloud",
    srcs = ["fastrak_gpumem_manager_cloud.cc"],
    hdrs = ["fastrak_gpumem_manager_cloud.h"],
    deps = [
        ":clock",
        ":clock_interface",
        ":fastrak_gpumem_manager_host_interface",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_tensorflow_serving//tensorflow_serving/util/net_http/public:shared_files",
        "@com_google_tensorflow_serving//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@json//json",
    ],
)

cc_library(
    name = "status_utils",
    srcs = ["status_utils.cc"],
    hdrs = ["status_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@googleapis//google/rpc:status_cc_proto",
    ],
)

cc_library(
    name = "fastrak_buffer_resource_tracker",
    srcs = ["fastrak_buffer_resource_tracker.cc"],
    hdrs = ["fastrak_buffer_resource_tracker.h"],
    deps = [
        "//dxs/client:dxs-client-types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "fastrak_gpumem_manager_host_interface",
    hdrs = ["fastrak_gpumem_manager_host_interface.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

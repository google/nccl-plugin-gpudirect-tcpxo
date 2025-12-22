load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "uds_helpers",
    srcs = ["uds_helpers.cc"],
    hdrs = ["uds_helpers.h"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "unix_socket_connection",
    srcs = ["unix_socket_connection.cc"],
    hdrs = ["unix_socket_connection.h"],
    deps = [
        "//buffer_mgmt_daemon/proto:tcpfastrak_buffer_mgmt_message_cc_proto",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "nvidia_mem_share_interface",
    hdrs = ["nvidia_mem_share_interface.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@cuda_headers",
    ],
)

cuda_library(
    name = "nvidia_tcpxo_mem_share",
    hdrs = ["nvidia_tcpxo_mem_share.h"],
    deps = [
        ":nvidia_mem_share_interface",
        "//buffer_mgmt_daemon:cuda_logging",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@cuda_headers",
    ],
)

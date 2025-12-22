load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "unix_socket_client",
    srcs = ["unix_socket_client.cc"],
    hdrs = ["unix_socket_client.h"],
    deps = [
        "//buffer_mgmt_daemon/common:uds_helpers",
        "//buffer_mgmt_daemon/common:unix_socket_connection",
        "//buffer_mgmt_daemon/proto:tcpfastrak_buffer_mgmt_message_cc_proto",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "buffer_mgr_client-interface",
    hdrs = [
        "buffer_mgr_client-interface.h",
    ],
    deps = [
        "//dxs/client:dxs-client-types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "buffer_mgr_client",
    srcs = ["buffer_mgr_client.cc"],
    hdrs = [
        "buffer_mgr_client.h",
    ],
    deps = [
        ":buffer_mgr_client-interface",
        ":unix_socket_client",
        "//buffer_mgmt_daemon/common:uds_helpers",
        "//buffer_mgmt_daemon/proto:tcpfastrak_buffer_mgmt_message_cc_proto",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@googleapis//google/rpc:code_cc_proto",
        "@googleapis//google/rpc:status_cc_proto",
    ],
)

cc_library(
    name = "bounce_buffer_handle",
    hdrs = [
        "bounce_buffer_handle.h",
    ],
    deps = [
        "//buffer_mgmt_daemon/proto:tcpfastrak_buffer_mgmt_message_cc_proto",
        "//dxs/client:dxs-client-types",
    ],
)

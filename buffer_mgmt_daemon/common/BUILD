load("@rules_cc//cc:defs.bzl", "cc_library")

# Common libraries for tcpdirect and tcpfastrak.

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

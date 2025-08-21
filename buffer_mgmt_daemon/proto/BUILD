load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "tcpfastrak_buffer_mgmt_message_proto",
    srcs = ["tcpfastrak_buffer_mgmt_message.proto"],
    deps = ["@googleapis//google/rpc:status_proto"],
)

cc_proto_library(
    name = "tcpfastrak_buffer_mgmt_message_cc_proto",
    deps = [":tcpfastrak_buffer_mgmt_message_proto"],
)

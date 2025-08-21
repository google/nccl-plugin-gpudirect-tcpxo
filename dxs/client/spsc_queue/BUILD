# Single-producer single-consumer queue intended for use with Llcm or Reverse Llcm.
# The queue is suitable for use with an MMIO region provided by Llcm.
# See design doc

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "spsc_queue_pair_proto",
    srcs = ["spsc_queue_pair.proto"],
)

cc_proto_library(
    name = "spsc_queue_pair_cc_proto",
    deps = [":spsc_queue_pair_proto"],
)

cc_library(
    name = "spsc_queue_pair",
    srcs = [
        "spsc_queue_pair.cc",
    ],
    hdrs = [
        "spsc_queue_pair.h",
    ],
    deps = [
        ":spsc_queue_pair_cc_proto",
        "//dxs/client/oss:barrier",
        "//dxs/client/oss:mmio",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "spsc_messaging_queue_pair",
    srcs = ["spsc_messaging_queue_pair.cc"],
    hdrs = ["spsc_messaging_queue_pair.h"],
    deps = [
        ":spsc_queue_pair",
        ":spsc_queue_pair_cc_proto",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

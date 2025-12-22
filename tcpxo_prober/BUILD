load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "connection",
    srcs = ["src/connection.cc"],
    hdrs = ["src/connection.h"],
    deps = [
        "//buffer_mgmt_daemon/client:buffer_mgr_client-interface",
        "//cuda_helpers:gpu_mem_helper_interface",
        "//cuda_helpers:gpu_mem_validator_interface",
        "//dxs/client:dxs-client-interface",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "nic-manager_lib",
    srcs = ["src/nic-manager.cc"],
    hdrs = ["src/nic-manager.h"],
    deps = [
        ":connection",
        ":connection_manager_cc_grpc_proto",
        ":connection_manager_cc_proto",
        ":prober_cc_proto",
        "//buffer_mgmt_daemon/client:buffer_mgr_client",
        "//buffer_mgmt_daemon/client:buffer_mgr_client-interface",
        "//cuda_helpers",
        "//cuda_helpers:gpu_mem_helper_interface",
        "//cuda_helpers:gpu_mem_validator_interface",
        "//dxs/client:dxs-client",
        "//dxs/client:dxs-client-interface",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@grpc//:grpc++",
    ],
)

cc_library(
    name = "agent_lib",
    srcs = ["src/agent.cc"],
    hdrs = ["src/agent.h"],
    deps = [
        ":connection_manager_cc_grpc_proto",
        ":connection_manager_cc_proto",
        ":nic-manager_lib",
        ":prober_cc_grpc_proto",
        ":prober_cc_proto",
        "//dxs/client:dxs-client",
        "//dxs/client:dxs-client-types",
        "//dxs/client/oss:status_macros",
        "@boost.asio",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@googleapis//google/rpc:code_cc_proto",
        "@grpc//:grpc++",
    ],
)

cc_binary(
    name = "prober",
    srcs = ["src/main.cc"],
    features = ["-enable_relr"],
    linkopts = ["-l:libstdc++.a"],
    linkstatic = True,
    deps = [
        ":agent_lib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@grpc//:grpc++",
    ],
)

py_binary(
    name = "load_test",
    srcs = ["scripts/load_test.py"],
    deps = [
        "//net/grpc/python:grpc",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
    ],
)

proto_library(
    name = "prober_proto",
    srcs = ["src/prober.proto"],
)

cc_proto_library(
    name = "prober_cc_proto",
    deps = [":prober_proto"],
)

cc_grpc_library(
    name = "prober_cc_grpc_proto",
    srcs = [":prober_proto"],
    grpc_only = True,
    deps = [":prober_cc_proto"],
)

proto_library(
    name = "connection_manager_proto",
    srcs = ["src/connection-manager.proto"],
)

cc_proto_library(
    name = "connection_manager_cc_proto",
    deps = [":connection_manager_proto"],
)

cc_grpc_library(
    name = "connection_manager_cc_grpc_proto",
    srcs = [":connection_manager_proto"],
    grpc_only = True,
    deps = [":connection_manager_cc_proto"],
)

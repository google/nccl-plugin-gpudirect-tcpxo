load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "buffer-manager-client-mock",
    testonly = 1,
    hdrs = ["buffer-manager-client_mock.h"],
    deps = [
        ":dxs-client-types",
        "//buffer_mgmt_daemon/client:bounce_buffer_handle",
        "//buffer_mgmt_daemon/client:buffer_mgr_client-interface",
        "//testing/base/public:gunit_for_library_testonly",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "llcm-memory-interface",
    hdrs = ["llcm-memory-interface.h"],
    deps = ["@com_google_absl//absl/types:span"],
)

cc_library(
    name = "guest-llcm-device",
    srcs = ["guest-llcm-device.cc"],
    hdrs = ["guest-llcm-device.h"],
    deps = [
        ":control-command",
        ":derive_dxs_address",
        ":llcm-memory-interface",
        "//dxs/client/guest_llcm",
        "//dxs/client/oss:htonl",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "guest-clock",
    srcs = [],
    hdrs = ["guest-clock.h"],
    deps = [
        "//dxs:clock-interface",
    ],
)

cc_library(
    name = "thread-shim",
    hdrs = ["thread-shim.h"],
    deps = [
        ":thread-shim-external",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "thread-shim-external",
    srcs = [
        "thread-shim.h",
        "thread-shim-external.cc",
    ],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "wire-version",
    hdrs = ["wire-version.h"],
)

cc_library(
    name = "control-command",
    srcs = ["control-command.cc"],
    hdrs = ["control-command.h"],
    deps = [
        ":control_command_cc_proto",
        ":data-sock",
        ":dxs-client-types",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@googleapis//google/rpc:status_cc_proto",
        "@protobuf//:any_cc_proto",
    ],
)

cc_library(
    name = "control-message-handler-interface",
    hdrs = ["control-message-handler-interface.h"],
    deps = [
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@protobuf",
    ],
)

cc_library(
    name = "data-sock",
    srcs = ["data-sock.cc"],
    hdrs = ["data-sock.h"],
)

cc_library(
    name = "dxs-client",
    srcs = ["dxs-client.cc"],
    hdrs = ["dxs-client.h"],
    deps = [
        ":atomic-future",
        ":control-command",
        ":control-message-handler-interface",
        ":control_command_cc_proto",
        ":data-sock",
        ":derive_dxs_address",
        ":dxs-client-interface",
        ":dxs-client-types",
        ":guest-llcm-device",
        ":linearized-recv-op",
        ":listen-socket",
        ":llcm-handler",
        ":llcm-memory-interface",
        ":make-unique-with-cleanup",
        ":monotonic-timestamp",
        ":mtu_utils",
        ":recv-socket",
        ":relaxed-atomic",
        ":sctp-handler",
        ":send-op",
        ":send-socket",
        ":sequence-number",
        ":wire-version",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dxs-client-interface",
    hdrs = [
        "dxs-client-interface.h",
    ],
    deps = [
        ":base-interface",
        ":control-command",
        ":control_command_cc_proto",
        ":dxs-client-types",
        ":monotonic-timestamp",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "llcm-handler",
    srcs = ["llcm-handler.cc"],
    hdrs = ["llcm-handler.h"],
    deps = [
        ":control-command",
        ":control-message-handler-interface",
        "//dxs/client/oss:status_macros",
        "//dxs/client/spsc_queue:spsc_messaging_queue_pair",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sctp-handler",
    srcs = ["sctp-handler.cc"],
    hdrs = ["sctp-handler.h"],
    deps = [
        ":control-message-handler-interface",
        ":guest-clock",
        ":thread-shim",
        "//dxs:sctp-timeout-queue",
        "//webrtc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sequence-number",
    hdrs = ["sequence-number.h"],
)

cc_library(
    name = "control-message-handler-mocks",
    testonly = True,
    hdrs = ["control-message-handler-mocks.h"],
    deps = [
        ":control-message-handler-interface",
        "//testing/base/public:gunit_for_library_testonly",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "relaxed-atomic",
    hdrs = ["relaxed-atomic.h"],
)

cc_library(
    name = "send-op",
    hdrs = ["send-op.h"],
    deps = [
        ":control-command",
        ":control-message-handler-interface",
        ":dxs-client-interface",
        ":dxs-client-types",
        ":monotonic-timestamp",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "make-unique-with-cleanup",
    hdrs = ["make-unique-with-cleanup.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "listen-socket",
    hdrs = ["listen-socket.h"],
    deps = [
        ":control-command",
        ":control-message-handler-interface",
        ":dxs-client-interface",
        ":dxs-client-types",
        ":relaxed-atomic",
        ":sequence-number",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "send-socket",
    hdrs = ["send-socket.h"],
    deps = [
        ":control-command",
        ":control-message-handler-interface",
        ":data-sock",
        ":dxs-client-interface",
        ":dxs-client-types",
        ":relaxed-atomic",
        ":sequence-number",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "recv-socket",
    hdrs = ["recv-socket.h"],
    deps = [
        ":control-command",
        ":control-message-handler-interface",
        ":data-sock",
        ":dxs-client-interface",
        ":dxs-client-types",
        ":relaxed-atomic",
        ":sequence-number",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "dxs-client-types",
    hdrs = ["dxs-client-types.h"],
)

cc_library(
    name = "monotonic-timestamp",
    hdrs = ["monotonic-timestamp.h"],
    deps = ["@com_google_absl//absl/time"],
)

cc_library(
    name = "linearized-recv-op",
    hdrs = ["linearized-recv-op.h"],
    deps = [
        ":control-command",
        ":control-message-handler-interface",
        ":dxs-client-interface",
        ":dxs-client-types",
        ":monotonic-timestamp",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "derive_dxs_address",
    srcs = ["derive_dxs_address.cc"],
    hdrs = ["derive_dxs_address.h"],
    deps = [
        ":control-command",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "mtu_utils",
    hdrs = ["mtu_utils.h"],
)

cc_library(
    name = "coalesce-iovecs",
    srcs = ["coalesce-iovecs.cc"],
    hdrs = ["coalesce-iovecs.h"],
)

cc_library(
    name = "base-interface",
    hdrs = ["base-interface.h"],
)

cc_library(
    name = "atomic-future",
    hdrs = ["atomic-future.h"],
    deps = ["@com_google_absl//absl/log:check"],
)

cc_library(
    name = "mock_dxs_client",
    testonly = True,
    hdrs = ["mock_dxs_client.h"],
    deps = [
        ":control-command",
        ":dxs-client-interface",
        ":dxs-client-types",
        ":monotonic-timestamp",
        "//testing/base/public:gunit_for_library_testonly",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

proto_library(
    name = "control_command_proto",
    srcs = ["control-command.proto"],
)

cc_proto_library(
    name = "control_command_cc_proto",
    deps = [":control_command_proto"],
)

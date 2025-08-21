# OSS-only replacements for internal libraries. Only included in OSS builds. Do not expose
# visibility for internal builds.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mmio",
    hdrs = ["mmio.h"],
    deps = [":barrier"],
)

cc_library(
    name = "barrier",
    hdrs = ["barrier.h"],
)

cc_binary(
    name = "status_macros_build_test",
    testonly = 1,
    srcs = ["status_macros_build_test.cc"],
    deps = [
        ":status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "htonl",
    hdrs = ["htonl.h"],
    deps = [
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/numeric:bits",
    ],
)

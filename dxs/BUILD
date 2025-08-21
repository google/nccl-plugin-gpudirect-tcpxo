load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "clock-interface",
    srcs = [],
    hdrs = ["clock-interface.h"],
)

cc_library(
    name = "sctp-timeout-queue",
    srcs = ["sctp-timeout-queue.cc"],
    hdrs = ["sctp-timeout-queue.h"],
    deps = [
        ":clock-interface",
        "//webrtc",
        "@com_google_absl//absl/log:check",
        "@ortools//ortools/base:adjustable_priority_queue",
    ],
)

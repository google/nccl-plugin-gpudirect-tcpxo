load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gpuviz_lib",
    hdrs = ["src/nccl_stats.h"],
    deps = ["@nccl//:plugin_lib"],
)

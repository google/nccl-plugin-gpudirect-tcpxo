load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl_net_compat",
    hdrs = ["nccl_net_compat.h"],
    deps = [
        "@nccl//:plugin_lib",
    ],
)

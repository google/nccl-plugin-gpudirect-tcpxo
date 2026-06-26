#CUDA-related components for NCCL FasTrak plugin

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_defs",
    hdrs = ["cuda_defs.h"],
    deps = ["@cuda_headers"],
)

cc_library(
    name = "cuda_common",
    srcs = ["cuda_common.cc"],
    hdrs = ["cuda_common.h"],
    deps = [
        ":cuda_defs",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@cuda_headers",
        "@local_cuda//:cuda",
        "@nccl//:plugin_lib",
    ],
)

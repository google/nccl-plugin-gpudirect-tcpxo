load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "cuda_helpers",
    srcs = ["cuda_helpers.cu"],
    hdrs = ["cuda_helpers.h"],
    deps = [
        ":gpu_mem_helper_interface",
        ":gpu_mem_validator_interface",
        "//buffer_mgmt_daemon:cuda_logging",
        "//buffer_mgmt_daemon/client:bounce_buffer_handle",
        "//buffer_mgmt_daemon/client:buffer_mgr_client",
        "//buffer_mgmt_daemon/common:nvidia_mem_share_interface",
        "//buffer_mgmt_daemon/common:nvidia_tcpxo_mem_share",
        "//dxs/client/oss:status_macros",
        "//tcpdirect_plugin/fastrak_offload:defs",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@cuda_headers",
        "@local_cuda//:cuda",
        "@nccl//:plugin_lib",
    ],
)

cc_library(
    name = "gpu_mem_helper_interface",
    hdrs = ["gpu_mem_helper_interface.h"],
    deps = [
        "//buffer_mgmt_daemon/client:bounce_buffer_handle",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "gpu_mem_validator_interface",
    hdrs = ["gpu_mem_validator_interface.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

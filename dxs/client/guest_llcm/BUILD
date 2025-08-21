load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "guest_llcm",
    srcs = ["guest_llcm.cc"],
    hdrs = ["guest_llcm.h"],
    deps = [
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "guest_llcm_tool",
    srcs = ["guest_llcm_tool.cc"],
    linkopts = ["-Wl,--dynamic-linker=/lib64/ld-linux-x86-64.so.2,--strip-all"],
    linkstatic = True,
    deps = [
        ":guest_llcm",
        "//base",
        "//base:strtoint",
        "//dxs/client/oss:barrier",
        "//dxs/client/oss:mmio",
        "//dxs/client/oss:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

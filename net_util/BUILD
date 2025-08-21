load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "network_system_call_interface_h",
    hdrs = ["network_system_call_interface.h"],
)

cc_library(
    name = "network_system_call_h",
    hdrs = ["network_system_call.h"],
    deps = [":network_system_call_interface_h"],
)

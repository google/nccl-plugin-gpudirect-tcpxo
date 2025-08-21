# BUILD rules for DmaBuf Bridge

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "dmabuf_importer",
    srcs = [
        "dmabuf_importer.c",
    ],
    hdrs = [
        "dmabuf_import_helper.h",
        "dmabuf_importer.h",
    ],
    linkstatic = 1,
    alwayslink = 1,
)

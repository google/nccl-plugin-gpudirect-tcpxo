# This BUILD file is written for use with the git repository at
# https://github.com/NVIDIA/nccl.
# It does not and is not intended to work with packaged/built releases of NCCL
# (those ending in .txz, .rpm, .deb, etc)

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],
)

license(
    name = "license",
    package_name = "nccl",
    license_kinds = ["@rules_license//licenses/spdx:BSD-3-Clause"],
)

licenses(["notice"])

# This will be present in the top-level directory of the NCCL installation.
# See the most recent version at https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt
exports_files(["LICENSE.txt"])

genrule(
    name = "nccl_h",
    srcs = glob(["**"]),
    outs = ["src/include/nccl.h"],
    cmd = " && ".join([
        'pushd "`dirname $(location Makefile)`"',
        'export WORKDIR="`pwd`"',
        "pushd src",
        'make "$$WORKDIR/build/include/nccl.h"',
        "popd",
        "popd",
        'cp -L "$$WORKDIR/build/include/nccl.h" "$@"',
    ]),
)

cc_library(
    name = "plugin_lib",
    hdrs = ["src/include/nccl.h"] + glob(["src/include/**/*.h"]),
    strip_include_prefix = "src/include",
    deps = [
        "@cuda_headers",
    ],
)

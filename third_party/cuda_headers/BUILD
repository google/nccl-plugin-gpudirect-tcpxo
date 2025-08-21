load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_license//rules:license.bzl", "license")
load("@rules_license//rules:license_kind.bzl", "license_kind")

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
    package_name = "cuda",
)

license_kind(
    name = "EULA",
    url = "https://docs.nvidia.com/cuda/eula/index.html",
)

licenses(["notice"])

# This will be present in the top-level directory of the CUDA installation.
# See the most recent version at https://docs.nvidia.com/cuda/eula/index.html
exports_files(["EULA.txt"])

cc_library(
    name = "cuda_headers",
    hdrs = glob(["include/**"]),
    includes = ["include"],
)

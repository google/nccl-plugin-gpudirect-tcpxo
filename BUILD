load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [
        ":license",
    ],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "fastrak-guest",
    copyright_notice = "Copyright © 2025 FasTrak team. All rights reserved.",
    license_kinds = [
        "@rules_license//licenses/spdx:BSD-3-Clause",
    ],
)

exports_files(["LICENSE"])

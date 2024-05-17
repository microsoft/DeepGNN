# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

filegroup(
    name = "all_srcs",
    srcs = glob(
        include = ["**"],
        exclude = ["*.bazel"],
    ),
)

configure_make(
    name = "openssl",
    configure_command = "config",
    configure_in_place = True,
    env = select({
        "@platforms//os:macos": {
            "AR": "",
            "PERL": "$$EXT_BUILD_ROOT$$/$(PERL)",
        },
        "//conditions:default": {
            "PERL": "$$EXT_BUILD_ROOT$$/$(PERL)",
        },
    }),
    lib_name = "openssl",
    lib_source = ":all_srcs",
    out_static_libs = [
        "libssl.a",
        "libcrypto.a",
    ],
    out_lib_dir = select({
        "@platforms//os:macos": "../copy_openssl/openssl/lib",
        "//conditions:default": "../copy_openssl/openssl/lib64",
    }),
    targets = ["install_sw"],
    toolchains = ["@rules_perl//:current_toolchain"],
    visibility = ["//visibility:public"],
)

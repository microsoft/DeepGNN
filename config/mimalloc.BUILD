# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

# TODO(alsamylk): use mimalloc on windows and mac.
filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debug_build",
    constraint_values = ["@platforms//os:linux"],
    values = {
        "compilation_mode": "dbg",
    },
)

config_setting(
    name = "optimized_build",
    constraint_values = ["@platforms//os:linux"],
    values = {
        "compilation_mode": "opt",
    },
)

cmake(
    name = "mimalloc",
    cache_entries = {
        "CMAKE_C_FLAGS": "-fPIC",
        "MI_OVERRIDE": "ON",
        "MI_USE_CXX": "ON",
        "MI_BUILD_STATIC": "ON",
        "MI_BUILD_TESTS": "OFF",
        "MI_INSTALL_TOPLEVEL": "ON",
        "CMAKE_INSTALL_LIBDIR": "libdir",
    },
    out_lib_dir = "libdir",
    lib_source = "@mimalloc//:all",
    out_static_libs = select({
        ":optimized_build": ["libmimalloc.a"],
        ":debug_build": ["libmimalloc-debug.a"],
    }),
    visibility = ["//visibility:public"],
)

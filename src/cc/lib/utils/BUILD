# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_cc//cc:defs.bzl", "cc_library")
load("//config:variables.bzl", "CXX_OPTS", "PLATFORM_DEFINES")

cc_library(
    name = "utils",
    hdrs = [
        "threadpool.h",
    ],
    copts = CXX_OPTS,
    linkopts = select({
        "@platforms//os:linux": ["-ldl"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
    ],
)

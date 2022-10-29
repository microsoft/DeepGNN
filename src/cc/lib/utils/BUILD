# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_cc//cc:defs.bzl", "cc_library")
load("//config:variables.bzl", "CXX_OPTS", "PLATFORM_DEFINES")

cc_library(
    name = "utils",
    srcs = [
        "thread_pool.cc",
    ],
    hdrs = [
        "thread_pool.h",
    ],
    copts = CXX_OPTS,
    linkopts = select({
        "@platforms//os:linux": ["-ldl"],
        "//conditions:default": [],
    }),
    # ERROR macro is defined in glog and windows.h
    defines = PLATFORM_DEFINES + ["GLOG_NO_ABBREVIATED_SEVERITIES"],
    features = ["fully_static_link"],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        "@boost//:asio",
    ],
)

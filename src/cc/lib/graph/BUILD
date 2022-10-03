# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_cc//cc:defs.bzl", "cc_library")
load("//config:variables.bzl", "CXX_OPTS", "PLATFORM_DEFINES")

cc_library(
    name = "graph",
    srcs = [
        "graph.cc",
        "locator.cc",
        "metadata.cc",
        "partition.cc",
        "sampler.cc",
        "hdfs_wrap.cc",
    ],
    hdrs = [
        "graph.h",
        "locator.h",
        "metadata.h",
        "partition.h",
        "sampler.h",
        "storage.h",
        "hdfs_wrap.h",
        "types.h",
        "xoroshiro.h",
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
        "@boost//:random",
        "@boost//:asio",
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

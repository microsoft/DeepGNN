# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_cc//cc:defs.bzl", "cc_test")
load("//config:variables.bzl", "CXX_OPTS")
load("//config:variables.bzl", "PLATFORM_DEFINES")

cc_library(
    name = "mocks",
    srcs = ["mocks.cc"],
    hdrs = ["mocks.h"],
    copts = CXX_OPTS,
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = ["//src/cc/lib/graph"],
)

cc_test(
    name = "native_tests",
    srcs = [
        "graph_test.cc",
        "mocks.cc",
        "mocks.h",
    ],
    copts = CXX_OPTS,
    defines = PLATFORM_DEFINES,
    linkopts = ["-lm"],
    deps = [
        ":mocks",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "distributed_tests",
    srcs = [
        "distributed_test.cc",
    ],
    copts = CXX_OPTS,
    defines = PLATFORM_DEFINES,
    linkopts = ["-lm"],
    deps = [
        ":mocks",
        "//src/cc/lib/distributed:grpc",
        "//src/cc/lib/graph",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "hdfs_tests",
    srcs = [
        "hdfs_test.cc",
    ],
    copts = CXX_OPTS,
    defines = PLATFORM_DEFINES,
    deps = [
        ":mocks",
        "//src/cc/lib/graph",
        "@googletest//:gtest_main",
        "@hadoop//:hadoop",
        "@hadoop//:hadoop_include",
    ],
    data = ["core-site.xml",],
    target_compatible_with = ["@bazel_tools//platforms:linux"],
)

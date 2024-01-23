# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//config:variables.bzl", "CXX_OPTS")

cc_binary(
    name = "sampler_benchmark",
    srcs = ["sampler_benchmark.cc"],
    copts = CXX_OPTS,
    linkopts = ["-lm"],
    deps = select({
        "//conditions:default": [
            "//src/cc/lib/graph",
            "@boost//:random",
            "@com_google_benchmark//:benchmark",
        ],
        "@platforms//os:linux": [
            "@mimalloc//:mimalloc",  # mimalloc should go first to ensure malloc is overridden everywhere
            "//src/cc/lib/graph",
            "@com_google_benchmark//:benchmark",
            "@boost//:random",
        ],
    }),
)

cc_binary(
    name = "search_benchmark",
    srcs = ["search_benchmark.cc"],
    copts = CXX_OPTS,
    linkopts = ["-lm"],
    deps = select({
        "//conditions:default": [
            "@com_google_benchmark//:benchmark",
        ],
        "@platforms//os:linux": [
            "@mimalloc//:mimalloc",  # mimalloc should go first to ensure malloc is overridden everywhere
            "@com_google_benchmark//:benchmark",
        ],
    }),
)

cc_binary(
    name = "neighbor_sampler_benchmark",
    srcs = ["neighbor_sampler_benchmark.cc"],
    copts = CXX_OPTS,
    linkopts = ["-lm"],
    deps = select({
        "//conditions:default": [
            "//src/cc/lib/graph",
            "@boost//:random",
            "@com_google_benchmark//:benchmark",
        ],
        "@platforms//os:linux": [
            "@mimalloc//:mimalloc",  # mimalloc should go first to ensure malloc is overridden everywhere
            "//src/cc/lib/graph",
            "@boost//:random",
            "@com_google_benchmark//:benchmark",
        ],
    }),
)

cc_binary(
    name = "grpc_benchmark",
    srcs = [
        "grpc_benchmark.cc",
    ],
    copts = CXX_OPTS,
    linkopts = [
        "-lm",
    ],
    deps = select({
        "//conditions:default": [
            "//src/cc/lib/distributed:grpc",
            "//src/cc/lib/graph",
            "//src/cc/tests:mocks",
            "@com_github_grpc_grpc//:grpc++",
            "@com_google_benchmark//:benchmark",
        ],
        "@platforms//os:linux": [
            "@mimalloc//:mimalloc",  # mimalloc should go first to ensure malloc is overridden everywhere
            "//src/cc/lib/distributed:grpc",
            "//src/cc/lib/graph",
            "//src/cc/tests:mocks",
            "@com_github_grpc_grpc//:grpc++",
            "@com_google_benchmark//:benchmark",
            "@boost//:random",
        ],
    }),
)

cc_binary(
    name = "partition_benchmark",
    srcs = ["partition_benchmark.cc"],
    copts = CXX_OPTS,
    linkopts = ["-lm"],
    deps = [
        "//src/cc/lib/graph",
        "//src/cc/tests:mocks",
        "@boost//:random",
        "@com_google_benchmark//:benchmark",
    ],
)

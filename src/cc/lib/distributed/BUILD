# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("//config:variables.bzl", "CXX_OPTS")

proto_library(
    name = "service_proto",
    srcs = ["service.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "service_cc_proto",
    deps = [":service_proto"],
)

cc_grpc_library(
    name = "service_cc_grpc",
    srcs = [":service_proto"],
    grpc_only = True,
    deps = [":service_cc_proto"],
)

windows_deps = [
        "//src/cc/lib/graph",
        # Order matters on windows. We want to use latest abseil instead of transitive from grpc.
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@boost//:random",
        ":service_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_benchmark//:benchmark",
        "@glog//:glog"
]

# Windows builds use a system provided openssl defined in .bazelrc file.
macos_linux_deps = windows_deps + ["@openssl"]

cc_library(
    name = "grpc",
    srcs = [
        "call_data.cc",
        "client.cc",
        "graph_engine.cc",
        "graph_sampler.cc",
        "server.cc",
    ],
    hdrs = [
        "call_data.h",
        "client.h",
        "graph_engine.h",
        "graph_sampler.h",
        "server.h",
    ],
    copts = CXX_OPTS,
    features = ["fully_static_link"],
    linkstatic = True,
    visibility = ["//visibility:public"],
    # ERROR macro is defined in glog and windows.h
    defines = ["GLOG_NO_ABBREVIATED_SEVERITIES", "OPENSSL_IS_BORINGSSL"],
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": macos_linux_deps,
    }),
)

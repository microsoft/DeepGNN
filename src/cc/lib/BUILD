# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//config:variables.bzl", "CXX_OPTS")

cc_binary(
    name = "wrapper",
    srcs = [
        "py_graph.cc",
        "py_graph.h",
        "py_server.cc",
    ],
    copts = CXX_OPTS,
    defines = ["GLOG_NO_ABBREVIATED_SEVERITIES"],
    # ERROR macro is defined in glog and windows.h
    linkopts = select({
        "@platforms//os:windows": [""],
        # Explicitly list functions we want to export.
        "@platforms//os:macos": [
            "-Wl,-exported_symbols_list,$(location version-script.darwin.lds)",
        ],
        "@platforms//os:linux": [
            "-Wl,--version-script=$(location version-script.linux.lds)",
            "-fwhole-program",
        ],
    }),
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@platforms//os:windows": [
            "//src/cc/lib/distributed:grpc",
            "//src/cc/lib/graph",
            "@com_github_google_glog//:glog",
            "@snark_com_google_absl//absl/container:flat_hash_map",
            "@boost//:random",
        ],
        "@platforms//os:macos": [
            "version-script.darwin.lds",
            "//src/cc/lib/distributed:grpc",
            "//src/cc/lib/graph",
            "@com_github_google_glog//:glog",
            "@snark_com_google_absl//absl/container:flat_hash_map",
            "@boost//:random",
        ],
        "@platforms//os:linux": [
            "@mimalloc//:mimalloc",  # mimalloc should go first to ensure malloc is overridden everywhere
            "version-script.linux.lds",
            "//src/cc/lib/distributed:grpc",
            "//src/cc/lib/graph",
            "@com_github_google_glog//:glog",
            "@snark_com_google_absl//absl/container:flat_hash_map",
            "@boost//:random",
        ],
    }),
)

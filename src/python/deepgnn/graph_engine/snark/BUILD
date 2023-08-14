# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_library")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

# out atrtibute is not configurable, hence the need for three rules
copy_file(
    name = "library_as_data_windows",
    src = "//src/cc/lib:wrapper",
    out = ":wrapper.dll",
)

copy_file(
    name = "library_as_data_macos",
    src = "//src/cc/lib:wrapper",
    out = ":libwrapper.dylib",
)

copy_file(
    name = "library_as_data_linux",
    src = "//src/cc/lib:wrapper",
    out = ":libwrapper.so",
)

py_library(
    name = "graph_engine_snark",
    srcs = [
        "_downloader.py",
        "_lib.py",
        "alias.py",
        "client.py",
        "convert.py",
        "meta_merger.py",
        "converter/__init__.py",
        "converter/process.py",
        "converter/writers.py",
        "converter/options.py",
        "decoders.py",
        "dispatcher.py",
        "distributed.py",
        "local.py",
        "meta.py",
        "server.py",
    ],
    data = select({
        "@platforms//os:windows": [":wrapper.dll"],
        "@platforms//os:macos": [":libwrapper.dylib"],
        "@platforms//os:linux": [":libwrapper.so"],
    }),
    deps = [
        "//src/python/deepgnn/graph_engine:graph_engine",
    ],
    visibility = ["//visibility:public"],
)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "graph_engine_backends",
    srcs = [
        "common.py",
        "options.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/python/deepgnn/graph_engine:graph_engine",
    ],
)

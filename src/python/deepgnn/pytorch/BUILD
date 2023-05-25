# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "deepgnn_pytorch",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine:graph_engine_exports",
        "//src/python/deepgnn:deepgnn",
    ],
    imports = ["src/python", "src/python/deepgnn", "src/python/deepgnn/pytorch"],
    visibility = ["//visibility:public"],
)

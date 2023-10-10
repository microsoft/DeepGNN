# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "deepgnn_tf",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//src/python/deepgnn:deepgnn",
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine:graph_engine_exports",
    ],
    visibility = ["//visibility:public"],
    deprecation = "This target is deprecated",
)

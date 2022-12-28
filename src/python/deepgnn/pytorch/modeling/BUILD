# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "deepgnn_pytorch_modeling",
    srcs = [
        "__init__.py",
        "base_model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/python/deepgnn/pytorch/encoding:deepgnn_pytorch_encoding",
    ],
)

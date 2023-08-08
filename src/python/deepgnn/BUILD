# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

py_library(
    name = "deepgnn",
    srcs = [
        "__init__.py",
        "arg_types.py",
        "log_consts.py",
        "logging_utils.py",
        "train_types.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_library(
    name = "deepgnn_exports",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
)

py_package(
    name = "deepgnn_ge_package",
    packages = ["src.python.deepgnn.graph_engine", "src.python.deepgnn"],
    deps = [
        ":deepgnn_exports",
        ":deepgnn",
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/graph_engine:graph_engine_exports",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
    ],
    visibility = ["//visibility:public"],
)

py_wheel(
    name = "deepgnn_ge_wheel",
    distribution = "deepgnn-ge",
    python_tag = "py3",
    strip_path_prefixes = [
        "src/python",
    ],
    version = "0.0.1",
    deps = [
        ":deepgnn_ge_package",
    ],
    visibility = ["//visibility:public"],
)

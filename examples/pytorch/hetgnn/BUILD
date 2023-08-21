# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "main",
    srcs = [
        "__init__.py",
        "evaluation.py",
        "graph.py",
        "main.py",
        "model.py",
        "sampler.py",
    ],
    deps = [
        "//src/python/deepgnn:deepgnn_ge_wheel_library",
        requirement("numpy"),
        requirement("scikit-learn"),
        requirement("fsspec"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("torch"),
        # requirement("torch-cluster"),
        # requirement("torch-scatter"),
        # requirement("torch-sparse"),
        requirement("torch_geometric"),
        requirement("tenacity"),
    ],
)

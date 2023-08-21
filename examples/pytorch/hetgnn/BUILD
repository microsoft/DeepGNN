# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")
load("@jvolkman_rules_pycross//pycross:defs.bzl", "pycross_wheel_library")

pycross_wheel_library(
    name = "deepgnn_ge_wheel_library",
    wheel = "//src/python/deepgnn:deepgnn_ge_wheel",
)

py_binary(
    name = "main",
    srcs = [
        "evaluation.py",
        "graph.py",
        "main.py",
        "model.py",
        "sampler.py",
        "__init__.py",
    ],
    deps = [
        ":deepgnn_ge_wheel_library",
        requirement("numpy"),
        requirement("scikit-learn"),
        requirement("fsspec"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("torch"),
        requirement("torch-cluster"),
        requirement("torch-scatter"),
        requirement("torch-sparse"),
        requirement("torch_geometric"),
        requirement("tenacity"),
    ],
)

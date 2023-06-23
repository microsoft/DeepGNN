# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "main",
    srcs = [
        "graph.py",
        "main.py",
        "model.py",
    ],
    deps = [
        "//src/python:deepgnn",
        "//src/python/deepgnn/graph_engine",
        "//src/python/deepgnn/graph_engine:graph_engine_exports",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine/backends:graph_engine_backends",
        requirement("numpy"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("fsspec"),
        requirement("transformers"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("tenacity"),
    ],
)

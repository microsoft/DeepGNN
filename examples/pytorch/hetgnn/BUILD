# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "main",
    srcs = [
        "evaluation.py",
        "main.py",
        "model.py",
        "sampler.py",
    ],
    deps = [
        "//src/python:deepgnn",
        "//src/python/deepgnn/graph_engine/backends:graph_engine_backends",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/pytorch:deepgnn_pytorch",
        "//src/python/deepgnn/pytorch/common:deepgnn_pytorch_common",
        "//src/python/deepgnn/pytorch/encoding:deepgnn_pytorch_encoding",
        "//src/python/deepgnn/pytorch/modeling:deepgnn_pytorch_modeling",
        "//src/python/deepgnn/pytorch/nn:deepgnn_pytorch_nn",
        "//src/python/deepgnn/pytorch/training:deepgnn_pytorch_training",
        requirement("numpy"),
        requirement("pytest"),
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "example_torch_graphsage",
    srcs = [
        "main.py",
    ],
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/pytorch:deepgnn_pytorch",
        "//src/python/deepgnn/pytorch/common:deepgnn_pytorch_common",
        "//src/python/deepgnn/pytorch/encoding:deepgnn_pytorch_encoding",
        "//src/python/deepgnn/pytorch/modeling:deepgnn_pytorch_modeling",
        "//src/python/deepgnn/pytorch/nn:deepgnn_pytorch_nn",
        "//src/python/deepgnn/pytorch/training:deepgnn_pytorch_training",
        "//examples/pytorch:example_torch_tests",
    ],
)

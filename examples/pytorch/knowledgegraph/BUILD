# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "example_torch_kg",
    srcs = [
        "model.py",
        "main.py",
    ],
    deps = [
        "//src/python/deepgnn/graph_engine/backends:graph_engine_backends",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/pytorch:deepgnn_pytorch",
        "//src/python/deepgnn/pytorch/common:deepgnn_pytorch_common",
        "//src/python/deepgnn/pytorch/encoding:deepgnn_pytorch_encoding",
        "//src/python/deepgnn/pytorch/modeling:deepgnn_pytorch_modeling",
        "//src/python/deepgnn/pytorch/nn:deepgnn_pytorch_nn",
        "//examples/pytorch:example_torch_tests",
    ],
)

py_test(
    name = "test_model",
    srcs = ["test_model.py"],
    imports = ["../../../src/python/"],
    main = "test_model.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":example_torch_kg",
        requirement("numpy"),
        requirement("pytest"),
        requirement("fsspec"),
        requirement("transformers"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "example_torch_lp",
    srcs = [
        "args.py",
        "consts.py",
        "encoder.py",
        "multi_task.py",
        "output_layer.py",
        "model.py",
        "main.py",
        "poolers.py",
    ],
    deps = [
        "//src/python/deepgnn/graph_engine/backends:graph_engine_backends",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine_testlib",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/pytorch:deepgnn_pytorch",
        "//src/python/deepgnn/pytorch/common:deepgnn_pytorch_common",
        "//src/python/deepgnn/pytorch/encoding:deepgnn_pytorch_encoding",
        "//src/python/deepgnn/pytorch/encoding/twinbert:deepgnn_pytorch_encoding_twinbert",
        "//src/python/deepgnn/pytorch/modeling:deepgnn_pytorch_modeling",
        "//src/python/deepgnn/pytorch/encoding/twinbert/deepspeed:deepgnn_pytorch_encoding_twinbert_ds",
        "//src/python/deepgnn/pytorch/nn:deepgnn_pytorch_nn",
        "//examples/pytorch:example_torch_tests",
    ],
)

py_library(
    name = "linkprediction_test_model",
    srcs = [
        "test_model.py",
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
        ":example_torch_lp",
        requirement("numpy"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("boto3"),
        requirement("fsspec"),
        requirement("transformers"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)


py_test(
    name = "test_poolers",
    srcs = ["test_poolers.py"],
    imports = ["../../../src/python/"],
    main = "test_poolers.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":example_torch_lp",
        ":linkprediction_test_model",
        requirement("numpy"),
        requirement("pytest"),
        requirement("fsspec"),
        requirement("scikit-learn"),
        requirement("boto3"),
        requirement("torch"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("transformers"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

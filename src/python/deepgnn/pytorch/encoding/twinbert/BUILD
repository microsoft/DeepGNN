# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "deepgnn_pytorch_encoding_twinbert",
    srcs = [
        "__init__.py",
        "configuration.py",
        "embedding.py",
        "encoder.py",
        "pooler.py",
        "tokenization.py",
    ],
    deps = [
        "//src/python/deepgnn/pytorch:deepgnn_pytorch",
        "//src/python/deepgnn/pytorch/common:deepgnn_pytorch_common",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_encoder",
    srcs = ["test_encoder.py"],
    imports = ["../../../../"],
    main = "test_encoder.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_pytorch_encoding_twinbert",
        requirement("numpy"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("networkx"),
        requirement("transformers"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

py_test(
    name = "test_tokenization",
    srcs = ["test_tokenization.py"],
    imports = ["../../../../"],
    main = "test_tokenization.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_pytorch_encoding_twinbert",
        requirement("numpy"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("transformers"),
        requirement("torch"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

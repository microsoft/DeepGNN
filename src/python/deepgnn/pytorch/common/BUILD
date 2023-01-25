# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "deepgnn_pytorch_common",
    srcs = [
        "__init__.py",
        "aggregators.py",
        "args.py",
        "consts.py",
        "metrics.py",
        "optimization.py",
        "ray_train.py",
        "utils.py",
    ],
    deps = [
        "//src/python/deepgnn/pytorch:deepgnn_pytorch",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_metrics",
    srcs = ["test_metrics.py"],
    imports = ["../../../"],
    main = "test_metrics.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_pytorch_common",
        "//src/python/deepgnn/graph_engine/backends/snark:graph_engine_backends_snark",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        requirement("numpy"),
        requirement("pytest"),
        requirement("torch"),
        requirement("fsspec"),
        requirement("networkx"),
        requirement("scikit-learn"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

py_test(
    name = "test_utils",
    srcs = ["test_utils.py"],
    imports = ["../../../", "./"],
    main = "test_utils.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_pytorch_common",
        "//src/python/deepgnn/graph_engine/backends/snark:graph_engine_backends_snark",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        requirement("numpy"),
        requirement("fsspec"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

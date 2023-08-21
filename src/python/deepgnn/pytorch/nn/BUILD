# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "deepgnn_pytorch_nn",
    srcs = [
        "__init__.py",
        "gat_conv.py",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_conv",
    srcs = ["test_conv.py"],
    imports = ["../../../"],
    main = "test_conv.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_pytorch_nn",
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

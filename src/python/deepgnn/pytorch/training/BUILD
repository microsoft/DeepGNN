# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "deepgnn_pytorch_training",
    srcs = [
        "__init__.py",
        "args.py",
        "utils.py",
    ],
    deps = [
        "//src/python/deepgnn/pytorch/modeling:deepgnn_pytorch_modeling",
        "//src/python/deepgnn/pytorch/encoding:deepgnn_pytorch_encoding",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_trainer",
    srcs = ["test_trainer.py"],
    imports = ["../../../"],
    main = "test_trainer.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:linux": [
            ":deepgnn_pytorch_training",
            requirement("horovod"),
            requirement("fsspec"),
            requirement("numpy"),
            requirement("pytest"),
            requirement("scikit-learn"),
            requirement("torch"),
            requirement("networkx"),
            requirement("tensorboard"),
            requirement("transformers"),
            requirement("opencensus"),
            requirement("opencensus-context"),
            requirement("opencensus-ext-azure"),
            requirement("azure-datalake-store"),
        ],
        "@platforms//os:macos": [
            ":deepgnn_pytorch_training",
            requirement("horovod"),
            requirement("numpy"),
            requirement("fsspec"),
            requirement("pytest"),
            requirement("scikit-learn"),
            requirement("torch"),
            requirement("networkx"),
            requirement("tensorboard"),
            requirement("transformers"),
            requirement("opencensus"),
            requirement("opencensus-context"),
            requirement("opencensus-ext-azure"),
            requirement("azure-datalake-store"),
        ],
        "@platforms//os:windows": [
            ":deepgnn_pytorch_training",
            requirement("numpy"),
            requirement("pytest"),
            requirement("scikit-learn"),
            requirement("fsspec"),
            requirement("torch"),
            requirement("networkx"),
            requirement("tensorboard"),
            requirement("transformers"),
            requirement("opencensus"),
            requirement("opencensus-context"),
            requirement("opencensus-ext-azure"),
            requirement("azure-datalake-store"),
        ],
    }),
)

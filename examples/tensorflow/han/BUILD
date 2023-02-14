# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "example_tf_han",
    srcs = [
        "han.py",
        "main.py",
    ],
    deps = [
        "//src/python/deepgnn/graph_engine/backends:graph_engine_backends",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/tf:deepgnn_tf",
        "//src/python/deepgnn/tf/common:deepgnn_tf_common",
        "//src/python/deepgnn/tf/encoders:deepgnn_tf_encoders",
        "//src/python/deepgnn/tf/nn:deepgnn_tf_nn",
    ],
)

py_test(
    name = "test_han",
    srcs = ["test_han.py"],
    imports = ["../../../src/python/"],
    main = "test_han.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":example_tf_han",
        requirement("numpy"),
        requirement("fsspec"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("tensorflow"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("tenacity"),
    ],
)

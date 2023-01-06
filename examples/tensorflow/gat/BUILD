# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "example_tf_gat",
    srcs = [
        "gat.py",
        "main.py",
    ],
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/tf:deepgnn_tf",
        "//src/python/deepgnn/tf/common:deepgnn_tf_common",
        "//src/python/deepgnn/tf/nn:deepgnn_tf_nn",
    ],
)

py_test(
    name = "test_gat",
    srcs = ["test_gat.py"],
    imports = ["../../../src/python/"],
    main = "test_gat.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":example_tf_gat",
        requirement("numpy"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("tensorflow"),
        requirement("fsspec"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

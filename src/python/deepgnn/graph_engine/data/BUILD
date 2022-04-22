# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test")
load("//config:variables.bzl", "PLATFORM_DEFINES")
load("@rules_python//python:defs.bzl", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "graph_engine_data",
    srcs = [
        "citation.py",
        "citeseer.py",
        "cora.py",
        "data_util.py",
        "ppi.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
    ],
)

py_test(
    name = "test_graph_dataset",
    srcs = ["test_graph_dataset.py"],
    imports = ["../../../"],
    main = "test_graph_dataset.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_engine_data",
        requirement("numpy"),
        requirement("azure-datalake-store"),
        requirement("pytest"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("scipy"),
        requirement("scikit-learn"),
        requirement("networkx"),
        requirement("fsspec"),

    ],
)

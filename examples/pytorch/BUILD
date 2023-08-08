# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")
load("@jvolkman_rules_pycross//pycross:defs.bzl", "pycross_wheel_library")



py_binary(
    name = "gcn",
    srcs = [
        "gcn.py",
    ],
    deps = [
        ":deepgnn_ge_wheel_library",
        requirement("azure-datalake-store"),
        requirement("fsspec"),
        requirement("grpcio"),
        requirement("msgpack"),
        requirement("numpy"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("packaging"),
        requirement("pyyaml"),
        requirement("ray"),
        requirement("referencing"),
        requirement("rpds"),
        requirement("rpds-py"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("torch-cluster"),
        requirement("torch-scatter"),
        requirement("torch-sparse"),
        requirement("torch_geometric"),
        requirement("tenacity"),
    ],
)

pycross_wheel_library(
    name = "deepgnn_ge_wheel_library",
    wheel = "//src/python/deepgnn:deepgnn_ge_wheel",
)

py_binary(
    name = "gat",
    srcs = [
        "gat.py",
    ],
    deps = [
        ":deepgnn_ge_wheel_library",
        requirement("azure-datalake-store"),
        requirement("fsspec"),
        requirement("numpy"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("torch-cluster"),
        requirement("torch-scatter"),
        requirement("torch-sparse"),
        requirement("torch_geometric"),
        requirement("tenacity"),
    ],
)

py_binary(
    name = "tgn",
    srcs = [
        "tgn.py",
    ],
    deps = [
        ":deepgnn_ge_wheel_library",
        requirement("azure-datalake-store"),
        requirement("fsspec"),
        requirement("numpy"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("torch-cluster"),
        requirement("torch-scatter"),
        requirement("torch-sparse"),
        requirement("torch_geometric"),
        requirement("tenacity"),
    ],
)


py_binary(
    name = "sage",
    srcs = [
        "sage.py",
    ],
    deps = [
        ":deepgnn_ge_wheel_library",
        requirement("azure-datalake-store"),
        requirement("fsspec"),
        requirement("numpy"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("torch-cluster"),
        requirement("torch-scatter"),
        requirement("torch-sparse"),
        requirement("torch_geometric"),
        requirement("tenacity"),
    ],
)

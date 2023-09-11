# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

common_deps = [
    "//src/python/deepgnn:deepgnn_ge_wheel_library",
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
    requirement("torch_geometric"),
    requirement("tenacity"),
]

sparse_deps = common_deps + [
    requirement("torch-sparse"),
    requirement("torch-scatter"),
    requirement("torch-cluster"),
]

py_binary(
    name = "gcn",
    srcs = [
        "gcn.py",
    ],
    deps = sparse_deps,
)

py_binary(
    name = "gat",
    srcs = [
        "gat.py",
    ],
    deps = common_deps,
)

py_binary(
    name = "tgn",
    srcs = [
        "tgn.py",
    ],
    deps = sparse_deps,
)

py_binary(
    name = "sage",
    srcs = [
        "sage.py",
    ],
    deps = sparse_deps,
)

py_binary(
    name = "aml",
    srcs = [
        "aml.py",
    ],
    deps = common_deps,
)

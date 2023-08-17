# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "graph_engine_backends_snark",
    srcs = [
        "client.py",
        "synchronized.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/python/deepgnn/graph_engine/backends:graph_engine_backends",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
    ],
)

common_deps = [
    ":graph_engine_backends_snark",
    requirement("numpy"),
    requirement("pytest"),
    requirement("opencensus"),
    requirement("opencensus-context"),
    requirement("opencensus-ext-azure"),
    requirement("azure-datalake-store"),
    requirement("networkx"),
    requirement("fsspec"),
    requirement("tenacity"),
]

py_test(
    name = "test_snark_client",
    srcs = ["test_snark_client.py"],
    imports = ["../../../../"],
    main = "test_snark_client.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_deps,
)

py_test(
    name = "test_synchronized",
    srcs = ["test_synchronized.py"],
    imports = ["../../../../"],
    main = "test_synchronized.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = common_deps,
)

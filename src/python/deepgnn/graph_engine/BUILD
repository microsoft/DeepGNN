# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test")
load("//config:variables.bzl", "PLATFORM_DEFINES")
load("@rules_python//python:defs.bzl", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "graph_engine",
    srcs = [
        "_adl_reader.py",
        "_base.py",
        "adl_uploader.py",
        "graph_dataset.py",
        "graph_fs.py",
        "graph_ops.py",
        "multihop.py",
        "prefetch.py",
        "samplers.py",
        "utils.py",
    ],
    data = ["//src/cc/lib:wrapper"],
    deps = [
        "//src/python/deepgnn:deepgnn",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "graph_engine_exports",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//src/python/deepgnn:deepgnn",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "graph_engine_testlib",
    srcs = [
        "test_adl_reader.py",
    ],
    deps = [
        "//src/python/deepgnn:deepgnn",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_prefetch",
    srcs = ["test_prefetch.py"],
    imports = ["../../"],
    main = "test_prefetch.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_engine",
        requirement("numpy"),
        requirement("fsspec"),
        requirement("pytest"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

py_test(
    name = "test_adl_reader",
    srcs = ["test_adl_reader.py"],
    imports = ["../../"],
    main = "test_adl_reader.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_engine",
        requirement("numpy"),
        requirement("fsspec"),
        requirement("pytest"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

py_test(
    name = "test_multihop",
    srcs = ["test_multihop.py"],
    imports = ["../../"],
    main = "test_multihop.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_engine",
        requirement("numpy"),
        requirement("fsspec"),
        requirement("pytest"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test")
load("//config:variables.bzl", "PLATFORM_DEFINES")
load("@rules_python//python:defs.bzl", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "test_dist_sync",
    srcs = ["test_dist_sync.py"],
    imports = ["../../../../"],
    main = "test_dist_sync.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/tf/common:deepgnn_tf_common",
        requirement("numpy"),
        requirement("pytest"),
        requirement("fsspec"),
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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "deepgnn_tf_common",
    srcs = [
        "__init__.py",
        "args.py",
        "base_trainer.py",
        "dataset.py",
        "dist_sync.py",
        "hooks.py",
        "horovod_trainer.py",
        "ps_trainer.py",
        "tf2_horovod_trainer.py",
        "tf2_trainer.py",
        "test_helper.py",
        "trainer.py",
        "trainer_factory.py",
        "utils.py",
    ],
    deps = [
        "//src/python/deepgnn/tf:deepgnn_tf",
        "//src/python/deepgnn/tf/layers:deepgnn_tf_layers",
        "//src/python/deepgnn/graph_engine/backends/snark:graph_engine_backends_snark",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_helper",
    srcs = ["test_helper.py"],
    imports = ["../../../"],
    main = "test_helper.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_tf_common",
        requirement("numpy"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("tensorflow"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

py_test(
    name = "test_trainer_tf2",
    srcs = ["test_trainer_tf2.py"],
    imports = ["../../../"],
    main = "test_trainer_tf2.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_tf_common",
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
    ],
)

py_test(
    name = "test_trainer",
    srcs = ["test_trainer.py"],
    imports = ["../../../"],
    main = "test_trainer.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_tf_common",
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
    ],
)

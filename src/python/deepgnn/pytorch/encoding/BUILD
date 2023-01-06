# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "deepgnn_pytorch_encoding",
    srcs = [
        "__init__.py",
        "feature_encoder.py",
        "gnn_encoder_gat.py",
        "gnn_encoder_hetgnn.py",
        "gnn_encoder_lgcl.py",
        "gnn_encoder_lightgcn.py",
        "gnn_encoder_sage.py",
    ],
    deps = [
        "//src/python/deepgnn/pytorch/encoding/twinbert:deepgnn_pytorch_encoding_twinbert",
        "//src/python/deepgnn/graph_engine/backends/snark:graph_engine_backends_snark",
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_feature_encoder",
    srcs = ["test_feature_encoder.py"],
    imports = ["../../../"],
    main = "test_feature_encoder.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_pytorch_encoding",
        requirement("numpy"),
        requirement("pytest"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("fsspec"),
        requirement("transformers"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

py_test(
    name = "test_gnn_encoders",
    srcs = ["test_gnn_encoders.py"],
    imports = ["../../../"],
    main = "test_gnn_encoders.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":deepgnn_pytorch_encoding",
        requirement("numpy"),
        requirement("pytest"),
        requirement("fsspec"),
        requirement("scikit-learn"),
        requirement("torch"),
        requirement("transformers"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

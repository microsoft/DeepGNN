# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

load("@rules_python//python:defs.bzl", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "snark_test_utils",
    srcs = [
        "util_test.py",
    ],
    deps = [
        "//src/python/deepgnn:deepgnn",
    ],
    visibility = ["//visibility:public"],
)


py_test(
    name = "sparse_features_tests",
    srcs = ["sparse_features_test.py"],
    imports = ["../../../../"],
    main = "sparse_features_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine/snark/tests:snark_test_utils",
        requirement("numpy"),
        requirement("pytest"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("grpcio"),
        requirement("fsspec"),
        requirement("grpcio-health-checking"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("tenacity"),
    ],
)

py_test(
    name = "e2e_tests",
    srcs = ["e2e_test.py"],
    imports = ["../../../../"],
    main = "e2e_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine/snark/tests:snark_test_utils",
        requirement("numpy"),
        requirement("pytest"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("fsspec"),
        requirement("grpcio"),
        requirement("grpcio-health-checking"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("tenacity"),
    ],
)

py_test(
    name = "convert_tests",
    srcs = ["convert_test.py"],
    imports = ["../../../../"],
    main = "convert_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine/snark/tests:snark_test_utils",
        requirement("importlib_metadata"),
        requirement("numpy"),
        requirement("pytest"),
        requirement("fsspec"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

py_test(
    name = "alias_tests",
    srcs = ["alias_test.py"],
    imports = ["../../../../"],
    main = "alias_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        requirement("numpy"),
        requirement("pytest"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)



py_test(
    name = "neighbor_sampler_tests",
    srcs = ["neighbor_sampler_test.py"],
    imports = ["../../../../"],
    main = "neighbor_sampler_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        requirement("numpy"),
        requirement("pytest"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("fsspec"),
        requirement("azure-datalake-store"),
        requirement("tenacity"),
    ],
)

py_test(
    name = "forest_fire_tests",
    srcs = ["forest_fire_test.py"],
    imports = ["../../../../"],
    main = "forest_fire_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine/snark/preprocess:snark_sampler",
        "//src/python/deepgnn/graph_engine:graph_engine",
        requirement("numpy"),
        requirement("pytest"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("tenacity"),
    ],
)

py_test(
    name = "metric_tests",
    srcs = ["metric_test.py"],
    imports = ["../../../../"],
    main = "metric_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        "//src/python/deepgnn/graph_engine/snark/preprocess:snark_sampler",
        requirement("numpy"),
        requirement("pytest"),
        requirement("networkx"),
        requirement("scipy"),
        requirement("scikit-learn"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
    ],
)

py_test(
    name = "downloader_tests",
    srcs = ["downloader_test.py"],
    imports = ["../../../../"],
    main = "downloader_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        requirement("numpy"),
        requirement("pytest"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("fsspec"),
        requirement("aiohttp"),
        requirement("tenacity"),
    ],
)

py_test(
    name = "random_walk_tests",
    srcs = ["random_walk_test.py"],
    imports = ["../../../../"],
    main = "random_walk_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        requirement("numpy"),
        requirement("pytest"),
        requirement("networkx"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("fsspec"),
        requirement("azure-datalake-store"),
        requirement("tenacity"),
    ],
)

py_test(
    name = "snark_test",
    srcs = ["snark_test.py"],
    imports = ["../../../../"],
    main = "snark_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine:graph_engine",
        requirement("numpy"),
        requirement("pytest"),
        requirement("opencensus"),
        requirement("networkx"),
        requirement("fsspec"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("azure-datalake-store"),
        requirement("tenacity"),
    ],
)

py_test(
    name = "hdfs_test",
    srcs = ["hdfs_test.py"],
    imports = ["../../../../"],
    main = "hdfs_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//src/python/deepgnn/graph_engine/snark:graph_engine_snark",
        "//src/python/deepgnn/graph_engine/data:graph_engine_data",
        "//src/python/deepgnn/graph_engine:graph_engine",
        "@hadoop//:hadoop_py",
        requirement("azure-datalake-store"),
        requirement("fsspec"),
        requirement("networkx"),
        requirement("numpy"),
        requirement("opencensus"),
        requirement("opencensus-context"),
        requirement("opencensus-ext-azure"),
        requirement("pytest"),
        requirement("scipy"),
        requirement("scikit-learn"),
        requirement("tenacity"),
    ],
    target_compatible_with = ["@platforms//os:linux"],
)

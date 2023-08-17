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
        "//src/python/deepgnn",
    ],
)

requirement_deps = [
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
]

# We keep windows and linux/mac dependencies separate because pycross fails to install wheels on windows in
windows_deps = requirement_deps + [
    "//src/python/deepgnn:deepgnn_ge_library",
    "//src/python/deepgnn/graph_engine/snark/tests:snark_test_utils",
]

linux_mac_deps = requirement_deps + [
    "//src/python/deepgnn:deepgnn_ge_wheel_library",
    "//src/python/deepgnn/graph_engine/snark/tests:snark_test_utils",
]

py_test(
    name = "sparse_features_tests",
    srcs = ["sparse_features_test.py"],
    imports = ["../../../../"],
    main = "sparse_features_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "e2e_tests",
    srcs = ["e2e_test.py"],
    imports = ["../../../../"],
    main = "e2e_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "convert_tests",
    srcs = ["convert_test.py"],
    imports = ["../../../../"],
    main = "convert_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "alias_tests",
    srcs = ["alias_test.py"],
    imports = ["../../../../"],
    main = "alias_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "neighbor_sampler_tests",
    srcs = ["neighbor_sampler_test.py"],
    imports = ["../../../../"],
    main = "neighbor_sampler_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "ppr_benchmark_tests",
    srcs = ["ppr_benchmark_test.py"],
    imports = ["../../../../"],
    main = "ppr_benchmark_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps + [
            requirement("py-cpuinfo"),
            requirement("pytest-benchmark"),
        ],
        "//conditions:default": linux_mac_deps + [
            requirement("py-cpuinfo"),
            requirement("pytest-benchmark"),
        ],
    }),
)

py_test(
    name = "temporal_tests",
    srcs = ["temporal_test.py"],
    imports = ["../../../../"],
    main = "temporal_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "forest_fire_tests",
    srcs = ["forest_fire_test.py"],
    imports = ["../../../../"],
    main = "forest_fire_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "metric_tests",
    srcs = ["metric_test.py"],
    imports = ["../../../../"],
    main = "metric_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps + [requirement("scipy")],
        "//conditions:default": linux_mac_deps + [requirement("scipy")],
    }),
)

py_test(
    name = "downloader_tests",
    srcs = ["downloader_test.py"],
    imports = ["../../../../"],
    main = "downloader_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps + [requirement("aiohttp")],
        "//conditions:default": linux_mac_deps + [requirement("aiohttp")],
    }),
)

py_test(
    name = "random_walk_tests",
    srcs = ["random_walk_test.py"],
    imports = ["../../../../"],
    main = "random_walk_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "snark_test",
    srcs = ["snark_test.py"],
    imports = ["../../../../"],
    main = "snark_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = select({
        "@platforms//os:windows": windows_deps,
        "//conditions:default": linux_mac_deps,
    }),
)

py_test(
    name = "hdfs_test",
    srcs = ["hdfs_test.py"],
    imports = ["../../../../"],
    main = "hdfs_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    target_compatible_with = ["@platforms//os:linux"],
    deps = requirement_deps + [
        "//src/python/deepgnn:deepgnn_ge_wheel_library",
        "@hadoop//:hadoop_py",
    ],
)

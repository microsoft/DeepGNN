# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared variables for build configuration based on the host platform."""

CXX_OPTS = select({
    "@bazel_tools//src/conditions:darwin": [
        "-std=c++2a",
        "-Werror",
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
        "-Wno-error=non-pod-varargs",
    ],
    "@bazel_tools//src/conditions:windows": ["/std:c++20", "/W0"],
    "//conditions:default": ["-std=c++20", "-Werror", "-fvisibility=hidden", "-fvisibility-inlines-hidden"],
})

PLATFORM_DEFINES = select({
    "@bazel_tools//src/conditions:linux": ["SNARK_PLATFORM_LINUX"],
    "@bazel_tools//src/conditions:windows": ["SNARK_PLATFORM_WINDOWS"],
    "//conditions:default": [],
})

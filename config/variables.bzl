# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared variables for build configuration based on the host platform."""

CXX_OPTS = select({
    "@platforms//os:macos": [
        "-std=c++20",
        "-Werror",
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
        "-Wno-error=non-pod-varargs",
    ],
    "@platforms//os:windows": ["/std:c++20", "/W3", "/guard:cf", "/Qspectre"],
    "//conditions:default": ["-std=c++20", "-Werror", "-fvisibility=hidden", "-fvisibility-inlines-hidden"],
})

PLATFORM_DEFINES = select({
    "@platforms//os:linux": ["SNARK_PLATFORM_LINUX"],
    "@platforms//os:windows": ["SNARK_PLATFORM_WINDOWS"],
    "//conditions:default": [],
})

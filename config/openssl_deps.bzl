# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A module defining the third party dependency OpenSSL"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def openssl_deps():
    maybe(
        http_archive,
        name = "rules_perl",
        sha256 = "7ad2510e54d530f75058e55f38e3e44acb682d65051514be88636adb1779b383",
        strip_prefix = "rules_perl-0.2.1",
        urls = [
            "https://github.com/bazelbuild/rules_perl/archive/refs/tags/0.2.1.tar.gz",
        ],
    )

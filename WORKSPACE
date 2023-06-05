# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# C++ related modules
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# Python and platform related modules
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "platforms",
    sha256 = "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
    ],
)

http_archive(
    name = "rules_python",
    sha256 = "c03246c11efd49266e8e41e12931090b613e12a59e6f55ba2efd29a7cb8b4258",
    strip_prefix = "rules_python-0.11.0",
    url = "https://github.com/bazelbuild/rules_python/archive/0.11.0.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    python_interpreter = "python",
    requirements_lock = "//:requirements.txt",
)

load("@pip_deps//:requirements.bzl", "install_deps")

install_deps()

pip_parse(
    name = "doc_deps",
    python_interpreter = "python",
    requirements_lock = "//:docs/requirements.txt",
)

load("@doc_deps//:requirements.bzl", install_doc_deps = "install_deps")

install_doc_deps()

git_repository(
    name = "googletest",
    remote = "https://github.com/google/googletest",
    tag = "release-1.11.0",
)

# Needed by glog below.
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "62efeb57ff70db9ea2129a16d0f908941e355d09d6d83c9f7b18557c0a7ab59e",
    strip_prefix = "glog-d516278b1cd33cd148e8989aec488b6049a4ca0b",
    urls = ["https://github.com/google/glog/archive/d516278b1cd33cd148e8989aec488b6049a4ca0b.zip"],
)

git_repository(
    name = "com_google_benchmark",
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.6.1",
)

http_archive(
    name = "rules_proto",
    sha256 = "66bfdf8782796239d3875d37e7de19b1d94301e8972b3cbd2446b332429b4df1",
    strip_prefix = "rules_proto-4.0.0",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

http_archive(
    name = "com_google_protobuf",
    sha256 = "89ac31a93832e204db6d73b1e80f39f142d5747b290f17340adce5be5b122f94",
    strip_prefix = "protobuf-3.19.4",
    urls = ["https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/protobuf-cpp-3.19.4.tar.gz"],
)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "b55696fb249669744de3e71acc54a9382bea0dce7cd5ba379b356b12b82d4229",
    strip_prefix = "grpc-1.51.1",
    urls = ["https://github.com/grpc/grpc/archive/v1.51.1.tar.gz"],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

http_archive(
    name = "com_google_absl",
    sha256 = "5b7640be0e119de1a9d941cb6b2607d76978eba5720196f1d4fc6de0421d2241",
    strip_prefix = "abseil-cpp-20220623.0",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.0.zip"],
)

http_archive(
    name = "rules_foreign_cc",
    sha256 = "bcd0c5f46a49b85b384906daae41d277b3dc0ff27c7c752cc51e43048a58ec83",
    strip_prefix = "rules_foreign_cc-0.7.1",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.7.1.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

http_archive(
    name = "mimalloc",
    build_file = "//config:mimalloc.BUILD",
    sha256 = "2b1bff6f717f9725c70bf8d79e4786da13de8a270059e4ba0bdd262ae7be46eb",
    strip_prefix = "mimalloc-2.1.2",
    urls = ["https://github.com/microsoft/mimalloc/archive/refs/tags/v2.1.2.tar.gz"],
)

http_archive(
    name = "com_github_nelhage_rules_boost",
    sha256 = "1a3316cde21eccc337c067b21d767d252e4ac2e8041d65eb4b7b91da569c5e3f",
    strip_prefix = "rules_boost-6b7c1ce2b8d77cb6b3df6ccca0b6cf7ed13136fc",
    url = "https://github.com/nelhage/rules_boost/archive/6b7c1ce2b8d77cb6b3df6ccca0b6cf7ed13136fc.tar.gz",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

http_archive(
    name = "jvm",
    build_file = "//config:jvm.BUILD",
    sha256 = "fb4bcb0d21ec332b086ce07f62cc7eda7f0099855d7eb53880fc4b275ecc1ceb",
    strip_prefix = "openlogic-openjdk-8u272-b10-linux-x64",
    urls = ["https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u272-b10/openlogic-openjdk-8u272-b10-linux-x64.tar.gz"],
)

http_archive(
    name = "hadoop",
    build_file = "//config:hadoop.BUILD",
    sha256 = "ad770ae3293c8141cc074df4b623e40d79782d952507f511ef0a6b0fa3097bac",
    strip_prefix = "hadoop-3.3.1",
    urls = ["https://dlcdn.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz"],
)

http_archive(
    name = "json",
    build_file = "//config:json.BUILD",
    sha256 = "d69f9deb6a75e2580465c6c4c5111b89c4dc2fa94e3a85fcd2ffcd9a143d9273",
    strip_prefix = "json-3.11.2",
    urls = ["https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz"],
)

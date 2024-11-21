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
    sha256 = "0a8003b044294d7840ac7d9d73eef05d6ceb682d7516781a4ec62eeb34702578",
    strip_prefix = "rules_python-0.24.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.24.0/rules_python-0.24.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    python_interpreter = "python",
    requirements_lock = "//:requirements.txt",
    extra_pip_args = ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
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
    tag = "v1.15.2",
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
    name = "abseil-cpp",
    sha256 = "95e90be7c3643e658670e0dd3c1b27092349c34b632c6e795686355f67eca89f",
    strip_prefix = "abseil-cpp-20240722.0",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.zip"],
)

http_archive(
    name = "rules_foreign_cc",
    sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
    strip_prefix = "rules_foreign_cc-0.9.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.9.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies()

http_archive(
    name = "openssl",
    urls = ["https://github.com/openssl/openssl/releases/download/openssl-3.3.0/openssl-3.3.0.tar.gz"],
    sha256 = "53e66b043322a606abf0087e7699a0e033a37fa13feb9742df35c3a33b18fb02",
    strip_prefix = "openssl-3.3.0",
    build_file = "//config:openssl.BUILD",
)

load("//config:openssl_deps.bzl", "openssl_deps")
openssl_deps()

load("//config:openssl_setup.bzl", "openssl_setup")
openssl_setup()

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
    sha256 = "f5195059c0d4102adaa7fff17f7b2a85df906bcb6e19948716319f9978641a04",
    strip_prefix = "hadoop-3.3.6",
    urls = ["https://dlcdn.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz"],
)

http_archive(
    name = "json",
    build_file = "//config:json.BUILD",
    sha256 = "d69f9deb6a75e2580465c6c4c5111b89c4dc2fa94e3a85fcd2ffcd9a143d9273",
    strip_prefix = "json-3.11.2",
    urls = ["https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz"],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# Use pycross rules to create a python library dependency from local wheels.
http_archive(
    name = "jvolkman_rules_pycross",
    url = "https://github.com/jvolkman/rules_pycross/archive/refs/tags/0.1.tar.gz",
    sha256 = "ad261f9240491732166f1db6c02c2f781a00d5cda50eddc7b02d1d40b3b2d7be",
    strip_prefix = "rules_pycross-0.1",
)

load("@jvolkman_rules_pycross//pycross:repositories.bzl", "rules_pycross_dependencies")
rules_pycross_dependencies()

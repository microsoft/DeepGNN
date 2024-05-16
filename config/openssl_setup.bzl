# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# openssl needs Perl and Text::Template module https://github.com/openssl/openssl/blob/master/INSTALL.md#prerequisites
load("@rules_perl//perl:deps.bzl", "perl_register_toolchains", "perl_rules_dependencies")

def openssl_setup():
    perl_rules_dependencies()
    perl_register_toolchains()

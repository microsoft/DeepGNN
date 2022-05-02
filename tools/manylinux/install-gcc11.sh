#!/bin/bash -eu
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Builds a devtoolset cross-compiler targeting manylinux 2014 (glibc 2.17 /
# libstdc++ 4.4).

TARGET="/dt11"
LIBSTDCXX_VERSION="6.0.29"

apt install unar flex rpm2cpio --yes

mkdir -p "${TARGET}"
chown ${USER} "${TARGET}"
wget "https://old-releases.ubuntu.com/ubuntu/pool/main/e/eglibc/libc6_2.17-0ubuntu5.1_amd64.deb"
unar "libc6_2.17-0ubuntu5.1_amd64.deb"
tar -C "${TARGET}" -xvzf "libc6_2.17-0ubuntu5.1_amd64/data.tar.gz"
rm -rf "libc6_2.17-0ubuntu5.1_amd64.deb" "libc6_2.17-0ubuntu5.1_amd64"
wget "https://old-releases.ubuntu.com/ubuntu/pool/main/e/eglibc/libc6-dev_2.17-0ubuntu5.1_amd64.deb"
unar "libc6-dev_2.17-0ubuntu5.1_amd64.deb"

tar -C "${TARGET}" -xvzf "libc6-dev_2.17-0ubuntu5.1_amd64/data.tar.gz"
rm -rf "libc6-dev_2.17-0ubuntu5.1_amd64.deb" "libc6-dev_2.17-0ubuntu5.1_amd64"

# Put the current kernel headers from ubuntu in place.
ln -s "/usr/include/linux" "${TARGET}/usr/include/linux"
ln -s "/usr/include/asm-generic" "${TARGET}/usr/include/asm-generic"
ln -s "/usr/include/x86_64-linux-gnu/asm" "${TARGET}/usr/include/asm"

# Symlinks in the binary distribution are set up for installation in /usr, we
# need to fix up all the links to stay within /${TARGET}.
./fixlinks.sh "${TARGET}"

# Patch to allow non-glibc 2.12 compatible builds to work.
sed -i '54i#define TCP_USER_TIMEOUT 18' "${TARGET}/usr/include/netinet/tcp.h"

# Download binary libstdc++ 4.4 release we are going to link against.
# We only need the shared library, as we're going to develop against the
# libstdc++ provided by devtoolset.
wget "http://old-releases.ubuntu.com/ubuntu/pool/main/g/gcc-4.4/libstdc++6_4.4.3-4ubuntu5_amd64.deb" && \
    unar "libstdc++6_4.4.3-4ubuntu5_amd64.deb" && \
    tar -C "/${TARGET}" -xvzf "libstdc++6_4.4.3-4ubuntu5_amd64/data.tar.gz" "./usr/lib/libstdc++.so.6.0.13" && \
    rm -rf "libstdc++6_4.4.3-4ubuntu5_amd64.deb" "libstdc++6_4.4.3-4ubuntu5_amd64"

mkdir -p "${TARGET}-src"
cp rpm-patch.sh "${TARGET}-src"
cd "${TARGET}-src"

# Build a devtoolset cross-compiler based on our glibc 2.17 sysroot setup.
wget "https://vault.centos.org/centos/7/sclo/Source/rh/devtoolset-11-gcc-11.2.1-1.2.el7.src.rpm"
rpm2cpio "devtoolset-11-gcc-11.2.1-1.2.el7.src.rpm" |cpio -idmv
tar -xvf "gcc-11.2.1-20210728.tar.xz" --strip 1

# Apply the devtoolset patches to gcc.
./rpm-patch.sh "gcc.spec"
./contrib/download_prerequisites

mkdir -p "${TARGET}-build"
chown ${USER} "${TARGET}-build"
cd "${TARGET}-build"

"${TARGET}-src/configure" \
      --prefix="${TARGET}/usr" \
      --with-sysroot="${TARGET}" \
      --disable-bootstrap \
      --disable-libmpx \
      --disable-libsanitizer \
      --disable-libunwind-exceptions \
      --disable-lto \
      --disable-multilib \
      --enable-__cxa_atexit \
      --enable-gnu-indirect-function \
      --enable-gnu-unique-object \
      --enable-initfini-array \
      --enable-languages="c,c++" \
      --enable-linker-build-id \
      --enable-plugin \
      --enable-shared \
      --enable-threads=posix \
      --with-default-libstdcxx-abi="gcc4-compatible" \
      --with-gcc-major-version-only \
      --with-linker-hash-style="gnu" \
      --with-tune="generic" \
      && \
    make -j 42 && \
    make install

# Create the devtoolset libstdc++ linkerscript that links dynamically against
# the system libstdc++ 4.4 and provides all other symbols statically.
mv "${TARGET}/usr/lib64/libstdc++.so.${LIBSTDCXX_VERSION}" "${TARGET}/usr/lib/libstdc++.so.${LIBSTDCXX_VERSION}.backup"
echo -e "OUTPUT_FORMAT(elf64-x86-64)\nINPUT ( libstdc++.so.6.0.13 -lstdc++_nonshared44 )" > "${TARGET}/usr/lib/x86_64-linux-gnu/libstdc++.so.${LIBSTDCXX_VERSION}"
cp "./x86_64-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++_nonshared44.a" "${TARGET}/usr/lib"

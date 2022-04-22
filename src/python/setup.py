# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import setuptools
import platform
import sys
import os
import shutil
import pathlib
from distutils.command.clean import clean

CODE_URL = "https://github.com/microsoft/DeepGNN"
AUTHOR = "DeepGNN Team"
AUTHOR_EMAIL = "DeepGNN@microsoft.com"

COMMON_PACKAGES = ["requests", "tensorboard"]


def graph_engine(version: str):
    """Setup tool to package graph engine."""

    def _shared_lib():
        if platform.system() == "Windows":
            return "wrapper.dll"
        return "libwrapper.so"

    # Generate manifest to include binary files.
    with open("MANIFEST.in", "w") as fo:
        for path in [
            os.path.join("../../bazel-bin", "src", "cc", "lib", _shared_lib())
        ]:
            shutil.copy(
                path,
                os.path.join(os.path.dirname(__file__), "deepgnn/graph_engine/snark"),
            )
            _, libname = os.path.split(path)
            fo.write("include python/deepgnn/graph_engine/snark/%s\n" % libname)

    here = pathlib.Path(__file__).parent.parent.resolve()
    long_description = (here / "../../README.md").read_text(encoding="utf-8")

    setuptools.setup(
        name="deepgnn-ge",
        version=version,
        description="Graph engine - distributed graph engine to host graphs.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url=CODE_URL,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        packages=setuptools.find_packages(
            include=["deepgnn", "deepgnn.graph_engine", "deepgnn.graph_engine.*"]
        ),
        package_data={"": ["*.so", "*.dll"]},
        install_requires=[
            "numpy>=1.17.0",
            "networkx==2.5.1",
            "azure-datalake-store",
            "opencensus-ext-azure",
            "fsspec>=2021.8.1",
            "scikit-learn",
            "scipy",
        ],
        project_urls={
            "Source": "https://github.com/microsoft/DeepGNN",
        },
        include_package_data=True,
        python_requires=">=3.7, <4",
        cmdclass={"clean": clean},
        license="MIT",
    )

    # Clean up files after build.
    os.remove("MANIFEST.in")
    os.remove(
        os.path.join(
            os.path.dirname(__file__), "deepgnn/graph_engine/snark", _shared_lib()
        )
    )


def deepgnn_tf(version: str):
    """DeepGNN runtime and algorithms for tensorflow."""
    depens = ["tensorflow>=2"]
    if "dev" in version:
        depens.append(f"deepgnn-ge=={version}")
    else:
        depens.append("deepgnn-ge>=0.1")

    depens.extend(COMMON_PACKAGES)

    setuptools.setup(
        name="deepgnn-tf",
        version=version,
        description="DeepGNN algorithms for tensorflow.",
        url=CODE_URL,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        packages=setuptools.find_packages(
            include=["deepgnn", "deepgnn.tf", "deepgnn.tf.*"]
        ),
        install_requires=depens,
        python_requires=">=3.7, <4",
        cmdclass={"clean": clean},
        license="MIT",
    )


def deepgnn_pytorch(version: str):
    """DeepGNN runtime and algorithms for pytorch."""
    if "dev" in version:
        depens = [f"deepgnn-ge=={version}"]
    else:
        depens = ["deepgnn-ge>=0.1"]

    depens.extend(
        [
            "torch>=1.8",
            "boto3>=1.15.16",
            "transformers>=4.3.3",
            "sentencepiece>=0.1.95",
            "tqdm>=4.51.0",
        ]
    )
    depens.extend(COMMON_PACKAGES)

    setuptools.setup(
        name=f"deepgnn-torch",
        version=version,
        description="DeepGNN algorithms for pytorch.",
        url=CODE_URL,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        packages=setuptools.find_packages(
            include=["deepgnn", "deepgnn.pytorch", "deepgnn.pytorch.*"]
        ),
        install_requires=depens,
        python_requires=">=3.7, <4",
        cmdclass={"clean": clean},
        license="MIT",
    )


if __name__ == "__main__":
    assert len(sys.argv) >= 2

    target = sys.argv[1]
    # get the build version from the pipeline.
    build_version = os.getenv("BUILD_VERSION")

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    target = target.lower()
    if target == "deepgnn-ge":
        graph_engine(build_version)
    elif target == "deepgnn-torch":
        deepgnn_pytorch(build_version)
    elif target == "deepgnn-tf":
        deepgnn_tf(build_version)
    else:
        raise ValueError(f"invalid target: {target}")

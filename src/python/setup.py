# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Setuptools to create deepgnn wheels."""

import setuptools
import platform
import sys
import os
import shutil
import stat
import pathlib
from distutils.command.clean import clean

CODE_URL = "https://github.com/microsoft/DeepGNN"
AUTHOR = "DeepGNN Team"
AUTHOR_EMAIL = "DeepGNN@microsoft.com"

COMMON_PACKAGES = ["requests", "tensorboard"]


def graph_engine(version: str):
    """Package graph engine."""

    def _shared_lib():
        if platform.system() == "Windows":
            return "wrapper.dll"
        elif platform.system() == "Darwin":
            return "libwrapper.dylib"
        return "libwrapper.so"

    # Generate manifest to include binary files.
    with open("MANIFEST.in", "w") as fo:
        for path in [
            os.path.join("../../bazel-bin", "src", "cc", "lib", _shared_lib())
        ]:
            if platform.system() == "Windows":
                os.chmod(path, stat.S_IWRITE)
            shutil.copy(
                path,
                os.path.join(
                    os.path.dirname(__file__), "deepgnn", "graph_engine", "snark"
                ),
            )
            _, libname = os.path.split(path)
            fo.write("include python/deepgnn/graph_engine/snark/%s\n" % libname)

    here = pathlib.Path(__file__).parent.parent.resolve()
    if sys.version_info.major > 3 or sys.version_info.minor >= 10:
        readme_path = here / "../README.md"
    else:
        readme_path = here / "../../README.md"
    long_description = readme_path.read_text(encoding="utf-8")

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
            include=[
                "deepgnn",
                "deepgnn.graph_engine",
                "deepgnn.graph_engine.*",
                "deepgnn.migrate",
                "deepgnn.migrate.*",
            ]
        ),
        package_data={"": ["*.so", "*.dll", "*.dylib", "*.zip"]},
        install_requires=[
            "numpy>=1.17.0",
            "networkx==2.5.1",
            "azure-datalake-store",
            "opencensus-ext-azure",
            "fsspec>=2021.8.1",
            "scikit-learn",
            "scipy>=1.10.0",
            "tenacity>=8",
        ],
        project_urls={
            "Source": "https://github.com/microsoft/DeepGNN",
        },
        include_package_data=True,
        python_requires=">=3.7",
        cmdclass={"clean": clean},
        license="MIT",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development :: Libraries",
        ],
    )

    # Clean up files after build.
    os.remove("MANIFEST.in")
    os.remove(
        os.path.join(
            os.path.dirname(__file__), "deepgnn", "graph_engine", "snark", _shared_lib()
        )
    )


if __name__ == "__main__":
    # get the build version from the pipeline.
    build_version = os.getenv("BUILD_VERSION")
    graph_engine(build_version)

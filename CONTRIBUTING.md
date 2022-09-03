# Contributing


This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.


# Prerequisites

## Linux

You'll need [bazel](https://docs.bazel.build/versions/master/install-ubuntu.html) and g++-10 to use project from source:

```bash
sudo apt install curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

sudo apt update && sudo apt install bazel clang-format g++-10
```

## MacOS

You'll need XCode, its command-line utilities and Bazel to build this project from source.
If you're using Homebrew you can install Bazel with:
```bash
brew install bazel
```

## Windows

The easiest way to use bazel on windows is via bazelisk, install [golang](https://golang.org/dl/), [bazelisk](https://github.com/bazelbuild/bazelisk), [compilers and language runtimes](https://docs.bazel.build/versions/4.1.0/install-windows.html#installing-compilers-and-language-runtimes).
Run all the build commands in the sections below in a powershell window, but replace `bazel` with `bazelisk` and use `--config=windows`.

# Build

* Ensure your default python command is for version 3, see "python --version".

Get the repo and build it:

```bash
git clone https://github.com/microsoft/DeepGNN
# for Windows and MacOS builds use `windows` and `darwin` config values.
bazel build -c opt //src/cc/lib:* --config=linux
```

For Debug builds use:

```bash
# for Windows and MacOS builds use `windows` and `darwin` config values.
bazel build -c dbg //src/cc/lib:* --config=linux
```

## Install python package:

```bash
cd src/python
export BUILD_VERSION=0.0.1
python setup.py deepgnn-ge bdist_wheel --plat-name manylinux1_x86_64 clean --all
python setup.py deepgnn-tf bdist_wheel --plat-name manylinux1_x86_64 clean --all
python setup.py deepgnn-torch bdist_wheel --plat-name manylinux1_x86_64 clean --all

python -m pip install --upgrade --force-reinstall dist/deepgnn_ge-$BUILD_VERSION-py3-none-manylinux1_x86_64.whl
python -m pip install --upgrade --force-reinstall dist/deepgnn_tf-$BUILD_VERSION-py3-none-manylinux1_x86_64.whl
python -m pip install --upgrade --force-reinstall dist/deepgnn_torch-$BUILD_VERSION-py3-none-manylinux1_x86_64.whl
```


If you want to submit a pull request to enhance project, please install code formatting tools to keep style uniform before publishing changes.

Linux:
```bash
sudo apt-get install python3-setuptools clang-format
```

Windows(in powershell):
```sh
$env:PATH+=";"+${env:ProgramFiles(x86)}+"\Microsoft Visual Studio\2019\Enterprise\VC\Tools\Llvm\bin"
```

Install and run linters(same on all systems):
```sh
pip install --upgrade pip
pip install -r tests/requirements.txt
pip install wheel pre-commit==2.17.0 mypy==0.971 numpy==1.21.6 torch==1.11.0 tensorflow==2.7.2
pre-commit install
pre-commit run --all-files
```


# Run tests

```bash
# run c++ GE tests:
bazel test //src/cc/tests:* --test_output=all --test_timeout 4 --config=linux

# run deepgnn python tests:
bazel test -c opt //src/python/deepgnn/...:* --test_output=all --test_timeout 6000 --config=linux

# run tests in examples folder:
bazel test -c opt //examples/tensorflow/...:* --test_output=all --test_timeout 6000 --config=linux
bazel test -c opt //examples/pytorch/...:* --test_output=all --test_timeout 6000 --config=linux

```

To run individual python tests you can add arguments to pytests from bazel:

```bash
bazel test //src/python/deepgnn/graph_engine/snark/tests:python_test --test_output=all --test_timeout 4 --config=linux --test_arg=-k --test_arg='test_sanity_neighbors_index'
```

This command will run all tests with prefix `test_sanity_neighbors_index`.

This folder contains scripts to install gcc11 compiler to generate manylinux14 compatible wheels taken from TF repo.
To install it locally run:
```bash
sudo ./install-gcc11.sh
```

Use `manylinux` configuration and `--force_pic` argument to build an almost self contained shared library with bazel:
```bash
bazel build -c opt //src/cc/lib:wrapper --config=manylinux --force_pic
```

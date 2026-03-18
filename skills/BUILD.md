# DeepGNN Build Guide

Complete compilation steps for building DeepGNN from source on all platforms.

---

## Prerequisites

| Requirement | Windows | Linux | macOS |
|---|---|---|---|
| **Bazel 6.4.0** | Install via `npm install -g @bazel/bazelisk` | `sudo apt install bazel=6.4.0` | `brew install bazel` |
| **C++ Compiler** | Visual Studio 2022 (MSVC) | `g++-12` | Xcode + CLI tools |
| **Python 3** | 3.8–3.10 recommended | 3.8–3.10 recommended | 3.8–3.10 recommended |
| **OpenSSL** | `choco install openssl` (installs to `C:\Program Files\OpenSSL-Win64`) | Typically pre-installed | Typically pre-installed |

---

## 1. Clone the Repository

```bash
git clone https://github.com/microsoft/DeepGNN
cd DeepGNN
```

### Windows: Path with Spaces

Bazel **cannot** build from paths containing spaces (e.g. OneDrive folders).
If your clone is under such a path, re-clone to a short path:

```powershell
git clone https://github.com/microsoft/DeepGNN C:\DeepGNN
cd C:\DeepGNN
```

---

## 2. Build the C++ Shared Library

This is the core graph engine (`wrapper.dll` / `libwrapper.so` / `libwrapper.dylib`).

### Linux

```bash
bazel build -c opt //src/cc/lib:wrapper --config=linux
```

### macOS

```bash
bazel build -c opt //src/cc/lib:wrapper --config=macos
```

### Windows (PowerShell)

```powershell
# Point Bazel at your Visual Studio VC directory
$env:BAZEL_VC = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC"

# Build (use bazelisk if bazel is not directly installed)
bazelisk build -c opt //src/cc/lib:wrapper --config=windows
```

**Key notes for Windows:**
- You must set `BAZEL_VC` to a Visual Studio version Bazel recognizes (VS 2022 works; non-standard VS versions may fail to locate C++ standard headers).
- The workspace path must **not** contain spaces.
- Use `bazelisk` (not `bazel`) — it auto-downloads the correct Bazel version.

### Build Output

| Platform | Output Path |
|---|---|
| Linux | `bazel-bin/src/cc/lib/libwrapper.so` |
| macOS | `bazel-bin/src/cc/lib/libwrapper.dylib` |
| Windows | `bazel-bin/src/cc/lib/wrapper.dll` |

### Debug Build

Replace `-c opt` with `-c dbg`:

```bash
bazel build -c dbg //src/cc/lib:wrapper --config=linux
```

---

## 3. Build the Python Wheel

After the C++ library is built:

```bash
cd src/python
export BUILD_VERSION=0.1.60   # or your desired version

# Build the wheel
pip install wheel
python setup.py bdist_wheel --plat-name manylinux1_x86_64 clean --all

# Install it
pip install --upgrade --force-reinstall dist/deepgnn_ge-${BUILD_VERSION}-py3-none-manylinux1_x86_64.whl
```

**Windows equivalent (PowerShell):**

```powershell
cd src\python
$env:BUILD_VERSION = "0.1.60"

pip install wheel
python setup.py bdist_wheel --plat-name win-amd64 clean --all

pip install --upgrade --force-reinstall "dist\deepgnn_ge-$env:BUILD_VERSION-py3-none-win_amd64.whl"
```

**Platform names for `--plat-name`:**

| Platform | Value |
|---|---|
| Linux | `manylinux1_x86_64` |
| Windows | `win-amd64` |
| macOS | `macosx-10.9-x86_64` |

---

## 4. Run Tests

### C++ Tests

```bash
bazel test -c dbg //src/cc/tests:* --test_output=all --test_timeout 30 --config=linux
```

### Python Tests

```bash
bazel test -c dbg //src/python/deepgnn/...:* --jobs 1 --test_output=all --test_timeout 600 --config=linux
```

### Run a Single Python Test

```bash
bazel test //src/python/deepgnn/graph_engine/snark/tests:python_test \
  --test_output=all --test_timeout 4 --config=linux \
  --test_arg=-k --test_arg='test_sanity_neighbors_index'
```

### PyTorch Examples

```bash
bazel run -c dbg //examples/pytorch:sage --config=linux
bazel run -c dbg //examples/pytorch:gcn --config=linux
bazel run -c dbg //examples/pytorch:gat --config=linux
```

> For Windows/macOS, replace `--config=linux` with `--config=windows` or `--config=macos`.

---

## 5. Linting & Formatting

```bash
pip install --upgrade pip
pip install -r tests/requirements.txt
pip install wheel pre-commit==2.17.0 mypy==0.971 numpy==1.22.4 torch==1.13.1 tensorflow==2.13.0
pre-commit install
pre-commit run --all-files
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `bazel does not currently work properly from paths containing spaces` | Clone/build from a path without spaces (e.g. `C:\DeepGNN`) |
| `Cannot open include file: 'string'` or `'limits'` | Set `$env:BAZEL_VC` to a VS version Bazel supports (VS 2022) |
| `cl.exe` not found | Install Visual Studio with "Desktop development with C++" workload |
| OpenSSL headers missing on Windows | `choco install openssl` — installs to `C:\Program Files\OpenSSL-Win64` |
| Bazel version mismatch | Use `bazelisk` which reads `.bazelversion` and auto-downloads the right version |

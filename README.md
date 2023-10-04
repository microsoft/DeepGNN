# DeepGNN Overview

DeepGNN is a framework for training machine learning models on large scale graph data. DeepGNN contains all the necessary features including:

* Distributed GNN training and inferencing on both CPU and GPU.
* Custom graph neural network design.
* Online Sampling: Graph Engine (GE) will load all graph data, each training worker will call GE to get node/edge/neighbor features and labels.
* Automatic graph partitioning.
* Highly performant and scalable.

Project is in alpha version, there might be breaking changes in the future and they will be documented in the changelog.

## Usage

Install pip package:
```bash
python -m pip install deepgnn
```
If you want to build package from source, see instructions in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Train and evaluate a graphsage model with pytorch on cora dataset:
```bash
cd examples/pytorch
python sage.py
```

## Migrating Scripts

We provide a python module to help you upgrade your scripts to new deepgnn versions.

```bash
pip install google-pasta
python -m deepgnn.migrate.0_1_56 --script_dir directory_to_migrate
```

See [`CHANGELOG.md`](CHANGELOG.md) for full change details.

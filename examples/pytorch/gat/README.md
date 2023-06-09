# Introduction

Graph attention networks (GATs) is a novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods’ features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront.

- Reference : [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)

### How to run
 - python main.py


#### Results

| Dataset  | Test Accuracy | Baseline (Paper) |
| -------- | ------------- | ---------------- |
| Cora     | 82.8          | 83.0 (+/-0.5)    |

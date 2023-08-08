# Introduction

This directory contains example models that can be trained with DeepGNN. To train and evaluate any model use `python {model_name}.py`. For example, `python sage.py` with train a sage model with default arguments.

# GAT

Graph attention networks (GATs) leverages masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods’ features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront.

- Reference : [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)
- `gat_pyg.py` contains GAT model implementation based on pytorch-geometric layers.
- `gat_ray.py` contains GAT model implementation based on pytorch-geometric layers and Ray for distributed training.

# SAGE
GraphSAGE is a framework for inductive representation learning on large graphs. GraphSAGE is used to generate low-dimensional vector representations for nodes, and is especially useful for graphs that have rich node attribute information.
Reference: [Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)

- `sage.py` contains a supervised graphsage model with pytorch-geometric layers trained on Cora dataset.

# HetGNN

Heterogeneous graphs contain abundant information with structural relations (edges) among multi-typed nodes as well as unstructured content associated with each node.

HetGNN introduces a random walk with restart strategy to sample a fixed size of strongly correlated heterogeneous neighbors for each node and group them based upon node types. Next, it designs a neural network architecture with two modules to aggregate feature information of those sampled neighboring nodes. The first module encodes “deep” feature interactions of heterogeneous contents and generates content embedding for each node. The second module aggregates content (attribute) embeddings of different neighboring groups (types) and further combines them by considering the impacts of different groups to obtain the ultimate node embedding. Finally, it  leverage a graph context loss and a mini-batch gradient descent procedure to train the model in an end-to-end manner.

- Reference: [Heterogeneous Graph Neural Network](https://www3.nd.edu/~dial/publications/zhang_2019_heterogeneous.pdf)
- `hetgnn/main.py` is the script to train a HetGNN model on academic graph.
- `hetgnn/graph.py` contains a pure python implementation of DeepGNN graph API needed to train the model above.


# TGN

Temporal Graph Networks is a generic framework for deep learning on dynamic graphs represented as sequences of timed events.

- Reference: [https://arxiv.org/abs/2006.10637](https://arxiv.org/abs/2006.10637)
- `tgn.py` contains TGN model implementation with pytorch-geometric modules and temporal graph based on MOOC dataset.

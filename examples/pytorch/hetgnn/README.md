# Introduction
__Heterogeneous graphs__ contain abundant information with structural relations (edges) among multi-typed nodes as
well as unstructured content associated with each node.

HetGNN first introduces a random walk with restart strategy to sample a fixed size of strongly correlated heterogeneous neighbors for each node and group them based upon node types. Next, it designs a neural network architecture with two modules to aggregate feature information of those sampled neighboring nodes. The first module encodes “deep” feature interactions of heterogeneous contents and generates content embedding for each node. The second module aggregates content (attribute) embeddings of different neighboring groups (types) and further combines them by considering the impacts of different groups to obtain the ultimate node embedding. Finally, it  leverage a graph context loss and a mini-batch gradient descent procedure to train the model in an end-to-end manner.

Reference: [Heterogeneous Graph Neural Network](https://www3.nd.edu/~dial/publications/zhang_2019_heterogeneous.pdf)

# Generate Graph Data
* [Prepare Graph Data](../../../docs/graph_engine/data_spec.rst)

# Job Augmentations
## HetGNN
Training
> --mode train --model hetgnn --neighbor_count 10 --model_dir /path/to/save/model --num_epochs 10 --batch_size 1024 --dim 128 --max_id 100000 --node_type_count 3 --feature_idx 0 --feature_dim 128 --learning_rate 0.01

Evaluate
> --mode evaluate --model hetgnn --neighbor_count 10 --model_dir /path/to/save/model --batch_size 1024 --dim 128 --max_id 100000 --node_type_count 3 --feature_idx 0 --feature_dim 128 --learning_rate 0.01 --sample_file=/home/tiantiaw/ppi_data/test_data/node_*

> __Note:__  sample_file is a list of node id files with different node type, for example, there are 3 types of node in a graph, when doing evaluation, we need to prepare 3 files with name "node_0.txt", "node_1.txt", "node_2.txt".


Inference
> --mode inference --model hetgnn --neighbor_count 10 --model_dir /path/to/save/model --sample_file /path/to/node/file --dim 128 --feature_idx 0 --feature_dim 128


# Parameters
Code reference:
- Create: [model.HetGNN (source code)](https://github.com/microsoft/DeepGNN/blob/main/examples/pytorch/hetgnn/model.py)

| Parameters | Default | Description |
| ----- | ----------- | ------- |
| **mode** | train | Run mode. ["train", "evaluate", "save_embedding", "inference"] |
| **model_dir** | ckpt | Model checkpoint. |
| **num_epochs** | 20 | Number of epochs for training. |
| **batch_size** | 512 | Mini-batch size. |
| **learning_rate** | 0.01 | Learning rate. |
| **feature_idx** | -1 | Feature index. |
| **feature_dim** | 0 | Feature dimension. |
| **learning_rate** | 0.01 | Learning rate. |
| **dim** | 0 | embedding dimension currently equal to feature_dim. |
| **max_id** | -1 | Max node id. |
| **neighbor_sampling_strategy** | byweight | samping strategy for node neighbors. ["byweight", "topk", "random", "randomwithoutreplacement"] |
| **neighbor_count** | 10 | Number of neighbors to sample of each node. |
| **node_type_count** | 2 | Number of node type in the graph. |
| **sample_file** | "" | File which contains node id to calculate the embedding. |

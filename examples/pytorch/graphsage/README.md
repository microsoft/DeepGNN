# Introduction
__GraphSAGE__ is a framework for inductive representation learning on large graphs. GraphSAGE is used to generate low-dimensional vector representations for nodes, and is especially useful for graphs that have rich node attribute information.

Reference: [Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)

# Generate Graph Data
* [Prepare Graph Data](../../../docs/graph_engine/data_spec.rst)

# Job Augmentations
## GraphSAGE - Unsupervised
Training
> --mode train --algo unsupervised --num_epochs 3 --batch_size 1024 --node_type 0 --max_id 56944 --feature_idx 1 --feature_dim 50

Evaluate
> --mode evaluate --algo unsupervised --batch_size 1024 --node_type 1 --max_id 56944 --feature_idx 1 --feature_dim 50

Inference
> --mode inference --model unsupervised --batch_size 1024 --node_type 0 --max_id 56944 --feature_idx 1 --feature_dim 50

## GraphSAGE - Supervised
Training
> --mode train --algo supervised --num_epochs 3 --batch_size 1024 --node_type 0 --max_id 56944 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121

Evaluate
> --mode evaluate --algo supervised --batch_size 1024 --node_type 1 --max_id 56944 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121

Inference
> --mode inference --algo supervised --batch_size 1024 --node_type 0 --max_id 56944 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121

# Parameters

## GraghSAGE parameters

| Parameters | Default | Description |
| ----- | ----------- | ------- |
| **mode** | train | task mode.  ["train", "evaluate", "inference"] |
| **model_dir** | "" | Model checkpoint. |
| **metric_dir** | "" | Training metrics save path. |
| **save_path** | "" | Inference result save path. |
| **num_epochs** | 20 | Number of epochs for training. |
| **batch_size** | 512 | Mini-batch size. |
| **learning_rate** | 0.01 | Learning rate. |
| **node_type** | 0 | Node type of training set. |
| **feature_idx** | -1 | Feature index. |
| **feature_dim** | 0 | Feature dimension. |
| **max_id** | -1 | Max node id. |
| **fanouts** | [10, 10] | fanouts settings. |
| **layer_dims** | [] | the dimensions for each layer. |
| **strategy** | byweight | samping strategy for node neighbors. ["byweight, "topk"] |
| **label_idx** | -1 | Label index. |
| **label_dim** | 0 | Label dimension. |
| **num_classes** | None | Number of classes. |
| **num_negs** | 5 | Number of negative samplings. |
| **neighbor_count** | 10 | Number of neighbors to sample of each node. |
| **featenc_config** | "" | Config file name of feature encoder. |
| **backend** | snark | Backend used by GE. |
| **graph_type** | local | Graph load type (local/remote). |

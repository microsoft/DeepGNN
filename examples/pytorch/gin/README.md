# Introduction
__Graph Isomorphic Network (GIN)__ is a framework for inductive representation learning on large graphs which is provably the most expressive amongst GNN variants, including GraphSAGE and GCNs. Further, the GIN model is as powerful as the Weisfeiler-Lehman graph isomorphism test with empirically validated benchmarks, proving its state-of-the-art results on both node and graph classification tasks.

Reference: [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km)

# Job Augmentations
## GIN - Supervised
Training
> --mode train --algo supervised --num_epochs 3 --batch_size 1024 --node_type 0 --max_id 56944 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121

Evaluate
> --mode evaluate --algo supervised --batch_size 1024 --node_type 1 --max_id 56944 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121

Inference
> --mode inference --algo supervised --batch_size 1024 --node_type 0 --max_id 56944 --feature_idx 1 --feature_dim 50 --label_idx 0 --label_dim 121

# Parameters

## GIN parameters

| Parameters | Default | Description |
| ----- | ----------- | ------- |
| **mode** | train | task mode.  ["train", "evaluate", "inference"] |
| **model_dir** | "" | Model checkpoint. |
| **metric_dir** | "" | Training metrics save path. |
| **save_path** | "" | Inference result save path. |
| **num_epochs** | 10 | Number of epochs for training. |
| **batch_size** | 128 | Mini-batch size. |
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

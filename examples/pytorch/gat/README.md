# Introduction

Graph attention networks (GATs) is a novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods’ features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront.

- Reference : [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)

### How to run
 - single worker training [run.sh](./run.sh)


#### Results

| Dataset  | Test Accuracy | Baseline (Paper) |
| -------- | ------------- | ---------------- |
| Cora     | 83.0          | 83.0 (+/-0.5)    |

# Generate Graph Data
* [Prepare Graph Data](../../../docs/advanced/data_spec.md)

# Job Augmentations
## GAT
Training
> --mode train --model gat --num_epochs 10 --batch_size 1024 --max_id 56944 --feature_idx 1 --feature_dim 128 --label_idx 0 --label_dim 121 --num_heads 1 --neighbor_count 10 --learning_rate 0.001 --model_dir /path/to/save/model

Evaluate
> --mode evaluate --model gat --num_epochs 10 --batch_size 1024 --max_id 56944 --feature_idx 1 --feature_dim 128 --label_idx 0 --label_dim 121 --num_heads 1 --neighbor_count 10 --model_dir /path/to/save/model --sample_file=/home/tiantiaw/ppi_data/test_data/node_*

Inference
> --mode inference --model gat --num_epochs 10 --batch_size 1024 --max_id 56944 --feature_idx 1 --feature_dim 128 --label_idx 0 --label_dim 121 --num_heads 1 --neighbor_count 10 --sample_file /path/to/node/file


# Parameters
| Parameters | Default | Description |
| ----- | ----------- | ------- |
| **mode** | train | Run mode. ["train", "evaluate", "save_embedding", "inference"] |
| **model_dir** | ckpt | Model checkpoint. |
| **num_epochs** | 20 | Number of epochs for training. |
| **batch_size** | 512 | Mini-batch size. |
| **learning_rate** | 0.01 | Learning rate. |
| **feature_idx** | -1 | Feature index. |
| **feature_dim** | 0 | Feature dimension. |
| **label_idx** | 0 | Label index. |
| **label_dim** | 0 | Label dimension. |
| **num_heads** | 1 | Number of the heads. |
| **max_id** | -1 | Max node id. |
| **neighbor_sampling_strategy** | byweight | samping strategy for node neighbors. ["byweight", "topk", "random", "randomwithoutreplacement"] |
| **neighbor_count** | 10 | Number of neighbors to sample of each node. |
| **sample_file** | "" | File which contains node id to calculate the embedding. |

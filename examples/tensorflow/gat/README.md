<<<<<<< HEAD
## Graph Attention Networks (GAT)
- Reference : [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)

### How to run
see [run.sh](./run.sh)


#### Results

| Dataset  | Test Accuracy | Baseline (Paper) |
| -------- | ------------- | ---------------- |
| Cora     | 83.3          | 83.0 (+/-0.5)    |
| Citeseer | 71.8          | 72.5 (+/-0.7)    |

Training time: (300 epochs)
| Dataset  | CPU  | GPU |
| -------- | ---- | --- |
| Cora     | 21.9  | 6.1 |
| Citeseer | 30.7  | 6.8 |

Cora test
```shell

# prepare cora dataset
python -m deepgnn.graph_engine.data.citation --dataset cora --data_dir /tmp/citation/cora

# train
python3 /examples/tensorflow/gat/main.py --mode train --seed 123 --model_dir /tmp/tmp70oef8fd --data_dir /tmp/citation/cora --eager --batch_size 140 --learning_rate 0.005 --epochs 300 --neighbor_edge_types 0 --attn_drop 0.6 --ffd_drop 0.6 --head_num 8,1 --l2_coef 0.0005 --hidden_dim 8 --gpu --feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 --num_classes 7 --prefetch_worker_size 1 --log_save_steps 20 --summary_save_steps 1

# evaluate
python3 /examples/tensorflow/gat/main.py --mode evaluate --seed 123 --model_dir /tmp/tmp70oef8fd --data_dir /tmp/citation/cora --eager --batch_size 1000 --evaluate_node_files /tmp/citation/cora/test.nodes --neighbor_edge_types 0 --attn_drop 0.0 --ffd_drop 0.0 --head_num 8,1 --l2_coef 0.0005 --hidden_dim 8 --gpu --feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 --num_classes 7 --prefetch_worker_size 1 --log_save_steps 1 --summary_save_steps 1
```

### Run GAT with your graph
* [Prepare Graph Data](../../../docs/advanced/data_spec.md)
=======
## Graph Attention Networks (GAT)
- Reference : [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)

### How to run
see [run.sh](./run.sh)


#### Results

| Dataset  | Test Accuracy | Baseline (Paper) |
| -------- | ------------- | ---------------- |
| Cora     | 83.3          | 83.0 (+/-0.5)    |
| Citeseer | 71.8          | 72.5 (+/-0.7)    |

Training time: (300 epochs)
| Dataset  | CPU  | GPU |
| -------- | ---- | --- |
| Cora     | 21.9  | 6.1 |
| Citeseer | 30.7  | 6.8 |

Cora test
```shell

# prepare cora dataset
python -m deepgnn.graph_engine.data.citation --dataset cora --data_dir /tmp/citation/cora

# train
python3 /examples/tensorflow/gat/main.py --mode train --seed 123 --model_dir /tmp/tmp70oef8fd --data_dir /tmp/citation/cora --eager --batch_size 140 --learning_rate 0.005 --epochs 300 --neighbor_edge_types 0 --attn_drop 0.6 --ffd_drop 0.6 --head_num 8,1 --l2_coef 0.0005 --hidden_dim 8 --gpu --feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 --num_classes 7 --prefetch_worker_size 1 --log_save_steps 20 --summary_save_steps 1

# evaluate
python3 /examples/tensorflow/gat/main.py --mode evaluate --seed 123 --model_dir /tmp/tmp70oef8fd --data_dir /tmp/citation/cora --eager --batch_size 1000 --evaluate_node_files /tmp/citation/cora/test.nodes --neighbor_edge_types 0 --attn_drop 0.0 --ffd_drop 0.0 --head_num 8,1 --l2_coef 0.0005 --hidden_dim 8 --gpu --feature_idx 0 --feature_dim 1433 --label_idx 1 --label_dim 1 --num_classes 7 --prefetch_worker_size 1 --log_save_steps 1 --summary_save_steps 1
```

### Run GAT with your graph
* [Prepare Graph Data](../../../docs/graph_engine/data_spec.rst)
>>>>>>> b30762e9d10d2ae19e206c7622f6d10554dc84f0

<<<<<<< HEAD
# Introduction
Reference: [Paper](https://arxiv.org/pdf/1903.07293).

# Generate Graph Data
* [Prepare Graph Data](../../../docs/advanced/data_spec.md)

# Job Augmentations
## HAN
Training
> --mode=train --model=han --num_epochs=2 --max_id=56994 --feature_idx=1 --learning_rate=0.001 --feature_dim=50 --label_idx=0 --label_dim=121 --fanouts=10 --all_edge_type=0,1 --batch_size=300 --head_num=8 --layer_dims=8

### Evaluate
> --mode=evaluate --model=han --max_id=56994 --feature_idx=1 --learning_rate=0.001 --feature_dim=50 --label_idx=0 --label_dim=121 --fanouts=10 --all_edge_type=0,1 --batch_size=300 --head_num=8 --layer_dims=8

### Inference
> --mode=inference --model=han --max_id=56994 --feature_idx=1 --learning_rate=0.001 --feature_dim=50 --label_idx=0 --label_dim=121 --fanouts=10 --all_edge_type=0,1 --batch_size=300 --head_num=8 --layer_dims=8

# Parameters
Code reference:
- Create [models.HAN(source code)](https://github.com/microsoft/DeepGNN/blob/main/examples/tensorflow/han/model.py)

| Parameters | Default | Description |
| ----- | ----------- | ------- |
| **mode** | train | Run mode. ["train", "evaluate", "save_embedding", "inference"] |
| **model_dir** | ckpt | Model checkpoint. |
| **num_epochs** | 20 | Number of epochs for training. |
| **batch_size** | 512 | Mini-batch size. |
| **learning_rate** | 0.01 | Learning rate. |
| **optimizer** | adam | TF Optimizer. ["adagrad", "adam", "adadelta", "rmsprop", "ftrl", "sgd", "momentum"] |
| **feature_idx** | -1 | Feature index. |
| **feature_dim** | 0 | Feature dimension. |
| **max_id** | -1 | Max node id. |
| **use_id** | False | Whether to use identity feature. |
| **label_idx** | -1 | Label index. |
| **label_dim** | 0 | Label dimension. |
| **all_edge_type** | [0] | All edge types of training set for HAN metapath. |
| **fanouts** | [10, 10] | neighbor/fanouts parameters for one metapath. HAN support multi-hop neighbor.|
| **head_num** | [1] | head attention num for each layer. HAN can support multipe layers.|
| **layer_dims** | [8] | Hidden dimension for each layer.|
=======
# Introduction
Reference: [Paper](https://arxiv.org/pdf/1903.07293).

# Generate Graph Data
* [Prepare Graph Data](../../../docs/graph_engine/data_spec.rst)

# Job Augmentations
## HAN
Training
> --mode=train --model=han --num_epochs=2 --max_id=56994 --feature_idx=1 --learning_rate=0.001 --feature_dim=50 --label_idx=0 --label_dim=121 --fanouts=10 --all_edge_type=0,1 --batch_size=300 --head_num=8 --layer_dims=8

### Evaluate
> --mode=evaluate --model=han --max_id=56994 --feature_idx=1 --learning_rate=0.001 --feature_dim=50 --label_idx=0 --label_dim=121 --fanouts=10 --all_edge_type=0,1 --batch_size=300 --head_num=8 --layer_dims=8

### Inference
> --mode=inference --model=han --max_id=56994 --feature_idx=1 --learning_rate=0.001 --feature_dim=50 --label_idx=0 --label_dim=121 --fanouts=10 --all_edge_type=0,1 --batch_size=300 --head_num=8 --layer_dims=8

# Parameters
Code reference:
- Create [models.HAN(source code)](https://github.com/microsoft/DeepGNN/blob/main/examples/tensorflow/han/model.py)

| Parameters | Default | Description |
| ----- | ----------- | ------- |
| **mode** | train | Run mode. ["train", "evaluate", "save_embedding", "inference"] |
| **model_dir** | ckpt | Model checkpoint. |
| **num_epochs** | 20 | Number of epochs for training. |
| **batch_size** | 512 | Mini-batch size. |
| **learning_rate** | 0.01 | Learning rate. |
| **optimizer** | adam | TF Optimizer. ["adagrad", "adam", "adadelta", "rmsprop", "ftrl", "sgd", "momentum"] |
| **feature_idx** | -1 | Feature index. |
| **feature_dim** | 0 | Feature dimension. |
| **max_id** | -1 | Max node id. |
| **use_id** | False | Whether to use identity feature. |
| **label_idx** | -1 | Label index. |
| **label_dim** | 0 | Label dimension. |
| **all_edge_type** | [0] | All edge types of training set for HAN metapath. |
| **fanouts** | [10, 10] | neighbor/fanouts parameters for one metapath. HAN support multi-hop neighbor.|
| **head_num** | [1] | head attention num for each layer. HAN can support multipe layers.|
| **layer_dims** | [8] | Hidden dimension for each layer.|
>>>>>>> b30762e9d10d2ae19e206c7622f6d10554dc84f0

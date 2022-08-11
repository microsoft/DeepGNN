# TwinBERT Overview

[TwinBERT](https://arxiv.org/pdf/2002.06275v1.pdf) is a twin-tower transformer based model architecture, which has been widely adopted in Microsoft Ads production stack. DeepGNN extracts the feature encoder from TwinBERT Auther's implementation and use it for generic text feature encoding. TwinBERT encoder does not only support standard `wordpiece` but also support `triletter` for tokenization and embedding generation, the later performs even better for some specific tasks.

DeepGNN also introduced transformer kernel in [DeepSpeed](https://github.com/microsoft/DeepSpeed) to further improve BERT-layer training speed. Several trivial changes were made on top of DeepSpeed original code.
1. Removed the dependency for `deepspeed_config` in `nvidia_modeling.py`.
2. Removed `max_seq_length` from the parameter of `DeepSpeedTransformerConfig` in `nvidia_modeling.py`
3. Replaced `dense_act` with `dense` in `nvidia_modeling.py` to make it compatible to checkpoint published by transformers@huggingface.


# Introduction

Link prediction model is a general DNN+GNN co-training model used to predict potential connection of two nodes in a graph. It contains configurable feature encoder(e.g TwinBERT), GNN encoder(e.g. GAT) and crossing layer. In the model, we use 'src' and 'dst' to represent two nodes associated with the potential connection, for example in query/ad graph 'src' could be used to represent query nodes and 'dst' will be used for ad nodes. 'src' and 'dst' don't have to be different types, different nodes with same type can also be supported.

The data format of the sample file is:

```
row_id, src_id, src_seq_mask, dst_id, dst_seq_mask, label
```

Training steps:

1. The feature encoder(e.g. TwinBERT) in the model will encode the `src (seq + mask)` and `dst (seq + mask)` separately.
2. The GNN encoder will aggregate the embeddings of its neighbors.
3. Both src and dst aggregated embedding will be passed to similarity pool to get the scores.


# Generate Graph Data
* [Prepare Graph Data](../../../docs/advanced/data_spec.md)

* Prepare training data

    e.g.

    |row_id | src_id | src_seq_mask | dst_id | dst_seq_mask | label |
    | ----- | ------ | ------------ | ------ | ------------ | ----- |
    | 0 | 0 | 1266,9292,2298,0,0,0,1,1,1,0,0,0 | 27929 | 11166,2298,1042,0,0,0,1,1,1,0,0,0 | 0 |
    | 1 | 2 | 68,6792,198,0,0,0,1,1,1,0,0,0    | 102   | 123,6387,7135,0,0,0,1,1,1,0,0,0   | 1 |



    * src_id: source node id.
    * src_seq_mask: concatenation of source seq and source mask, separated by ','.
    * dst_id: destination node id.
    * dst_seq_mask: concatenation of destination seq and source mask, separated by ','.
    * label: if or not the source and destination node has link.

    ***The length of sequence and mask is the same which means we can divide it by 2 to get its
    seq: 'data[:feature_dim // 2]' and its mask: 'data[feature_dim // 2:]'***

# Job Augmentations

Training
> --mode train --model link_prediction --num_epochs 10 --batch_size 1024 --meta_dir /path/to/twinbert_json/folder --featenc_config linkprediction.json --num_heads 1  --learning_rate 0.001 --model_dir /path/to/save/model

Evaluate
> --mode evaluate --model link_prediction --num_epochs 10 --batch_size 1024 --meta_dir /path/to/twinbert_json/folder --featenc_config linkprediction.json --num_heads 1  --learning_rate 0.001 --model_dir /path/to/save/model --sample_file=/home/tiantiaw/ppi_data/test_data/node_*

Inference
> --mode inference --model link_prediction --num_epochs 10 --batch_size 1024 --meta_dir /path/to/twinbert_json/folder --featenc_config linkprediction.json --num_heads 1  --learning_rate 0.001 --model_dir /path/to/save/model --sample_file /path/to/node/file


# Parameters
| Parameters | Default | Description |
| ----- | ----------- | ------- |
| **mode** | train | Run mode. ["train", "evaluate", "inference"] |
| **model_dir** | ckpt | Model checkpoint. |
| **meta_dir** | '' | Directory of config files, it will concate with featenc_config to get the full path of the config file. |
| **featenc_config** | '' | linkprediction.json config file path. |
| **num_epochs** | 20 | Number of epochs for training. |
| **batch_size** | 512 | Mini-batch size. |
| **learning_rate** | 0.01 | Learning rate. |
| **num_heads** | 1 | Number of the heads. |
| **neighbor_sampling_strategy** | byweight | samping strategy for node neighbors. ["byweight", "topk", "random", "randomwithoutreplacement"] |
| **sample_file** | "" | File which contains node id to calculate the embedding. |
| **share_encoder** | store_true | Whether or not to share the feature encoder for different node types. |
| **sim_type** | cosine | Pooler type used in the crossing layer. |
| **nsp_gamma** | 1 | Negative sampling factor used in cosine with rng pooler. |
| **weight_decay** | 0.01 | weight_decay. |
| **use_mse** | store_true | If MSELoss will be used. |
| **res_size** | 64 | Residual layer dimension. |
| **res_bn** | store_true | If BatchNorm1d is used in residual layers. |
| **label_map** | None | This is used to normalize the labels in each task. |
| **gnn_encoder** | "gat" | Encoder name of GNN layer. |
| **gnn_acts** | leaky_relu,tanh | Activation functions used in GNN layer. |
| **gnn_head_nums** | 2 | Number of the heads. |
| **gnn_hidden_dims** | 128 | Hidden layer dimensions. |
| **lgcl_largest_k** | 0 | top k neighbors when using LGCL. |
| **gnn_residual** | "add" | residual layer type. |
| **src_encoders** | "" | Types of encoders used to encode feature of src nodes and their 1st/2nd hop neighbors. |
| **dst_encoders** | "" | Types of encoders used to encode feature of dst nodes and their 1st/2nd hop neighbors. |
| **neighbor_mask** | store_true | Nodes which need to be removed if masked. |
| **src_metapath** | "0;2" | neighbor node types of source node which need to be sampled. |
| **src_fanouts** | "3;2" | how many neighbors of source node will be sampled for each hop. |
| **dst_metapath** | "1;4" | neighbor node types of destination node which need to be sampled. |
| **dst_fanouts** | "3;2" | how many neighbors of destination node will be sampled for each hop. |
| **train_file_dir** | "" | Train file directory. It can be local path or adl path. |


# Sample twinbert.json config file

Sample config file is `deepgnn/core/pytorch/testdata/twinbert/linkprediction.json`. When using this config, make sure to specify the `--featenc_config` to 'linkprediction.json' and set `--meta_dir` to its folder.


```
{
    "hidden_size":512,
    "num_hidden_layers":6,
    "num_attention_heads":8,
    "intermediate_size":512,
    "max_position_embeddings":512,
    "type_vocab_size":2,
    "vocab_size":49292,
    "hidden_act":"gelu",

    "pooler_type":"weightpooler",
    "embedding_type":"wordpiece",
    "triletter_max_letters_in_word":20,
    "max_seq_len":12,
    "downscale":128,
    "is_query":true,
    "init_ckpt_file":"",
    "init_ckpt_prefix":"",
    "enable_fp16":false,

    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,

    "deepspeed":
    {
        "transformer_kernel":false,
        "train_batch_size":1024,
        "train_micro_batch_size_per_gpu":1024,
        "stochastic_mode":false,
        "normalize_invertible":false,
        "seed":42
    }
}

```

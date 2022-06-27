# Preparing Graph Data

DeepGNN supports both homogeneous and heterogeneous graphs. Nodes and Edges support the following attributes,
  * Node/Edge Type: `int`.
  * Node/Edge Weight: `float`.
  * Node/Edge Features: `float/uint64/string`.

## Graph Data Format

DeepGNN supports two file formats: JSON and TSV. Users can generate a graph in either format then our pipeline will convert it into binary for training and inference.

1. [Linear](#linear-format): Heterogeneous or homegeneous graph.
2. [JSON](#json-format): Heterogeneous or homegeneous graph.
3. [TSV](#tsv-format): Homogeneous graph only.

## Linear Format

Here is the graph data Linear format. The format requires one file graph.linear and with an optional meta.json.

Linear is the most compact format, fastest to process by far and is most directly what is given to binary writers.

### Graph Data

`graph.linear` layout

```
<node info> <edge_1_info> <edge_2_info> ...
```
  node_info: -1 node_id node_type node_weight <features>
  edge_info: src dst edge_type edge_weight <features>
  features[dense]: dtype_name length v1 v2 ... dtype_name2 length2 v1 v2 ...
  features[sparse]: dtype_name coords.size,values.size c1 c2 ... v1 v2 ...
  features[sparse]: dtype_name coords.shape[0],coords.shape[1],values.size c1 c2 ... v1 v2
  * Nodes must be sorted by node_id, edges sorted by src and then dst.

Here is a concrete example,
A graph with 2 nodes {0, 1} each with type = 1, weight = .5 and
feature vectors [1, 1, 1] dtype=int32 and [1.1, 1.1] dtype=float32.
Edges: {0 -> 1, 1 -> 0} both with type = 0, weight = .5 and a sparse feature
vector (coords=[0, 4], values=[1, 1, 1] dtype=uint8).
```
-1 0 1 .5 int32 3 1 1 1 float32 2 1.1 1.1 0 1 0 .5 uint8 2,3 0 4 1 1 1
-1 1 1 .5 int32 3 1 1 1 float32 2 1.1 1.1 1 0 0 .5 uint8 2,3 0 4 1 1 1
```

### Optional Graph Metadata

Optional headers:
  "node_default_type": int Type of all nodes, if set do not add node type to any nodes.
  "node_default_weight": int Weight of all nodes, if set do not add node weight to any nodes.
  "node_default_features": "dtype_name length ..." Feature vectors dtype and length.
      Any value can be "none" which will require it to be specified for each node.
      There can be more feature vectors than specified.
  "edge_default_type": int Same as node except for all edges.
  "edge_default_weight": int Same as node except for all edges.
  "edge_default_features": "dtype_name length ..." Same as node except for all edges.

e.g. the same graph as above with meta.json fully filled in
```JSON
{
    "node_default_type": 1,
    "node_default_weight": .5,
    "node_default_features": "int32 3 float32 2",
    "edge_default_type": 0,
    "edge_default_weight": .5,
    "edge_default_features": "uint8 2,3",
}
```
graph.linear
```
-1 0 1 1 1 1.1 1.1 0 1 0 4 1 1 1
-1 1 1 1 1 1.1 1.1 1 0 0 4 1 1 1
```

## JSON Format

Here is the graph data JSON format. The format requires one file: graph.json.

### Graph Data

`graph.json` layout

```JSON
{
  "node_id": "int",
  "node_type": "int",
  "node_weight": "float",
  "neighbor": {"edge type": {"neighbor id": "weight(float)", "...": "..."}, "...": "..."},
  "uint64_feature": {"feature id": ["int", "..."], "...": "..."},
  "float_feature": {"feature id": ["float", "..."], "...": "..."},
  "binary_feature": {"feature id": "string", "...": "..."},
  "edge":[{
    "src_id": "int",
    "dst_id": "int",
    "edge_type": "int",
    "weight": "float",
    "uint64_feature": {"feature id": ["int", "..."], "...": ["int", "..."]},
    "float_feature": {"feature id": ["float", "..."], "...": ["float", "..."]},
    "binary_feature": {"feature id": "string", "...": "..."},
    "sparse_int32_feature": {"feature id": {"coordinates": [["non zero coordinates 0"], ["non zero coordinates 1", "..."]], "values": ["value 0", "value 1", "..."]}},
  }, "..."]
}
```

Here is a concrete example,

```JSON
{
  "node_id": 5797133,
  "node_type": 0,
  "node_weight": 1.0,
  "neighbor": {"0": {"6103589": 2.0, "6892569": 1.3}},
  "uint64_feature": {},
  "float_feature": {"0": [-490.0, 797.0, 2069.0], "1": [1967.0, 1280.0]},
  "binary_feature": {"2": "microsoft", "1": "bing"},
  "edge":[
    {
      "src_id": 5797133,
      "dst_id": 6103589,
      "edge_type": 0,
      "weight": 2.0,
      "uint64_feature": {},
      "float_feature": {"0": [-1.531, 1.34, 0.235, 2.3], "1": [-2.1, 0.4, 0.35, 0.3]},
      "binary_feature": {"2": "welcome"},
      "sparse_uint64_feature": {"3": {"coordinates": [[5, 13], [7, 25]], "values": [-1, 1024]}},
    },
    {
      "src_id": 5797133,
      "dst_id": 6892569,
      "edge_type": 0,
      "weight": 1.3,
      "uint64_feature": {},
      "float_feature": {"0": [-0.31, -2.04, 0.53, 0.123], "1": [-3.1, 0.4, 0.35, 0.3]},
      "binary_feature": {"2": "hello DeepGNN."}
    }
  ]
}
```

## TSV Format

Currently TSV format ONLY support homogenous graphs.

The format requires one file: graph.tsv. The graph.tsv format is as follows,

```tsv
| node_id | node_type | node_weight | node_features                | neighbors                                  |
| --------|-----------|-------------|------------------------------|--------------------------------------------|
| 1       | 0         | 0.1         | f:0.1 0.2;b:str_feat;i:1 2 3 | 2, 0, 0.3, 1, f:0.1 0.2;b:str_feat;i:1 2 3 |
...
```

node_id: int, The node's unique identifier.
node_type: int, Node type, typically 0 for training, 1 for testing and 2 for inference.
node_weight: float, Node weight.
node_features: | type1:v1 v2;type2:v1 v2 |, Node feature vectors, type can be one of the following: {f: float, b: binary, i: integer}. There can be any number of values for each feature. There can only be a single vector for each feature type.
neighbors: | int, int, float, int, features |, src_id, dst_id, edge_weight, edge_type and a feature vector in the same form as node_features.

# fsspec support

DeepGNN supports reading graph data from different data sources such as local file system, http, HDFS... This is done by using `fsspec` library. When loading data from a specific storage, user need to specify the full path of the graph data, for example, if the graph is in local file system, the path should be "/path/to/local/graph/data", if graph data is in HDFS, user need to specify `hdfs://domain/path/to/graph/data`. DeepGNN will try to parse the file protocol and download the graph data to local temporary folder, and local them into graph engine servers. More details can be found here: [_downloader.py](https://github.com/microsoft/DeepGNN/blob/main/src/python/deepgnn/graph_engine/snark/_downloader.py)

sample code:

```Python
import deepgnn.graph_engine.snark.server as server

# if the data is in local FS
data_dir = "/home/deepgnn/cora"
# if the data is in HDFS
# data_dir = "hdfs://my.hdfs.domain/data/cora"
port = 12345
s = server.Server(data_dir, [0], f"0.0.0.0:{port}")

```

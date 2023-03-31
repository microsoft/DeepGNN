####################
Preparing Graph Data
####################

DeepGNN supports both homogeneous and heterogeneous graphs. Nodes and Edges support the following attributes,

  * Node/Edge Type: `int, >= 0`.
  * Node/Edge Weight: `float`.
  * Node/Edge Features: `float/uint64/string`.

*****************
Graph Data Format
*****************

DeepGNN supports three file formats: EdgeList, JSON and TSV.
Users can generate a graph in either format then our pipeline will convert it into binary for training and inference.
DeepGNN also supports writing custom decoders, see [the decoders file](https://github.com/microsoft/DeepGNN/blob/main/src/python/deepgnn/graph_engine/snark/decoders.py).
Just inheret the base class Decoder, overwrite the decode function and pass the new decoder as an argument to the converter or dispatcher.

	1. `EdgeList <#EdgeList-format>`_: Heterogeneous or homegeneous graph.

	2. `JSON <#json-format>`_: Heterogeneous or homegeneous graph.

	3. `TSV <#tsv-format>`_: Homogeneous graph only.

EdgeList Format
===============

The EdgeList format,

	* Nodes and edges are on separate lines, so they may be sorted after generating the file.
	* Small files that are fast to create and convert.
	* Supports heterogeneous and homegeneous graphs.

`graph.csv` layout,

.. code-block:: text

	<node_0_info>
	<edge_0,1_info>
	<edge_0,2_info>
	...
	<node_1_info>
	...

.. code-block:: text

	node_info: node_id,-1,node_type,node_weight,<features>
	edge_info: src,edge_type,dst,edge_weight,<features>

Sort the file so the first line has the first node's info, the next few lines have all the first node's
outgoing edges. Then the next line will have the second node's info and so on.

Feature fectors to fill <features> can be dense or sparse. Features will be given
indexes starting at 0 and indexes can be skipped with 0 length vectors. Each node and
edge does not have to have the same number of features or feature types.

.. code-block:: text

	features[dense]: dtype,length,v1,v2,...,dtype2,length2,v1,v2,...
	features[sparse]: dtype,values_length/coords_dim,c1,c2,...,v1,v2,...

This sparse feature representation will generate a values vector with shape (values_length) and coordinates vector
with shape (values_length, coords_dim). If coordinates vector should be flat, use values_length/0 to get the
coordinates vector shape (values_length).

Feature data types supported values

.. code-block:: text

	binary: string feature, requires length=1 but string can be any length, e.g. "binary,1,example".
	bool: array of bools.
	int8/16/32/64: integer array with N bytes per value.
	uint8/16/32/64: unsigned integer array with N bytes per value.
	uint8/16/32/64: float array with N bytes per value.

Here is a concrete example: a graph with 2 nodes {0, 1} each with type = 1, weight = .5 and
feature vectors [1, 1, 1] dtype=int32 and [1.1, 1.1] dtype=float32.
Edges {0 -> 1, 1 -> 0} both with type = 0, weight = .5 and a sparse feature
vector (coords=[0, 4, 10], values=[1, 1, 1] dtype=uint8).

.. code-block:: text

	0,-1,1,.5,int32,3,1,1,1,float32,2,1.1,1.1
	0,0,1,.5,uint8,3/0,0,4,10,1,1,1
	1,-1,1,.5,int32,3,1,1,1,float32,2,1.1,1.1
	1,0,0,.5,uint8,3/0,0,4,10,1,1,1

Delimiters

.. code-block:: text

	"," is the default column delimiter, it can be overriden with the delimiter parameter.
	"/" is the default sparse features length delimiter, it can be overriden with the length_delimiter parameter.
	"\" is the escape for the delimiter in "binary" features, it can be overriden with the binary_escape parameter.

Sorting
-------

For some use cases that use the EdgeList format as an intermediate between their input format and binaries,
it may be more efficient to first convert to EdgeList and sort it afterward.

Here is an example using the bash exteral sort function with N workers.

.. code-block:: text

	sort edgelist.csv -t, -n -k1,1 -k2,2 -k3,3 --parallel=1 -o output.csv

Advanced Usage
--------------

If your graph has the same types, weights, feature types or feature lengths,
you can avoid writing this same info on every line by using the following init args,

.. code-block:: text

	default_node_type: int Type of all nodes, if set do not add node type to any nodes.
	default_node_weight: int Weight of all nodes, if set do not add node weight to any nodes.
	default_node_feature_types: ["dtype" or None, ...] Dtype of each feature vector.
	default_node_feature_lens: [[int, ...] or None, ...] Length value for each feature vector.
	default_edge_type: int Same as node except for all edges.
	default_edge_weight: int Same as node except for all edges.
	default_edge_feature_types: ["dtype" or None, ...] Dtype of each feature vector.
	default_edge_feature_lens: [[int, ...] or None, ...] Length value for each feature vector.

e.g. the same graph as before with init fully filled in,

.. code-block:: python

	EdgeListDecoder(
		default_node_type=1,
		default_node_weight=.5,
		default_node_feature_types=["int32", "float32"],
		default_node_feature_lens=[[3],[2]],
		default_edge_type=0,
		default_edge_weight=.5,
		default_edge_feature_types=["uint8"],
		default_edge_feature_lens=[[3, 0]],
	)

`condensed homogeneous graph.csv`,

.. code-block:: text

	0,-1,1,1,1,1.1,1.1
	0,1,0,4,10,1,1,1
	1,-1,1,1,1,1.1,1.1
	1,0,0,4,10,1,1,1

JSON Format
===========

The JSON format supports heterogeneous and homegeneous graphs.

`graph.json` layout:

.. code-block:: json

	{
	"node_id": "int",
	"node_type": "int, >= 0",
	"node_weight": "float",
	"neighbor": {"edge type": {"neighbor_id": "weight(float)", "...": "..."}, "...": "..."},
	"uint64_feature": {"feature_id": ["int", "..."], "...": "..."},
	"float_feature": {"feature_id": ["float", "..."], "...": "..."},
	"binary_feature": {"feature_id": "string", "...": "..."},
	"edge":[{
		"src_id": "int",
		"dst_id": "int",
		"edge_type": "int, >= 0",
		"weight": "float",
		"uint64_feature": {"feature_id": ["int", "..."], "...": ["int", "..."]},
		"float_feature": {"feature_id": ["float", "..."], "...": ["float", "..."]},
		"binary_feature": {"feature_id": "string", "...": "..."},
		"sparse_int32_feature": {"feature_id": {"coordinates": [["non zero coordinates 0"], ["non zero coordinates 1", "..."]], "values": ["value 0", "value 1", "..."]}},
		}, "..."]
	}

Here is a concrete example:

.. code-block:: json

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
		},
	],
	}

TSV Format
==========

Currently TSV format ONLY support homogenous graphs.

The format requires the file graph.tsv as follows,

.. code-block:: text

	| node_id | node_type | node_weight | node_features                | neighbors                                  |
	| --------|-----------|-------------|------------------------------|--------------------------------------------|
	| 1       | 0         | 0.1         | f:0.1 0.2;b:str*feat;i:1 2 3 | 2, 0, 0.3, 1, f:0.1 0.2;b:str*feat;i:1 2 3 |

	...



node_id: int, The node's unique identifier.

node_type: int, Node type, typically 0 for training, 1 for testing and 2 for inference.

node_weight: float, Node weight.

node_features: *|type1:v1 v2;type2:v1 v2|*, Node feature vectors, type can be one of the following: {f: float, b: binary, i: integer}. There can be any number of values for each feature. There can only be a single vector for each feature type.

neighbors: *| int, int, float, int, features |*, src_id, dst_id, edge_weight, edge_type and a feature vector in the same form as node_features.

Generated meta.json Format
=========================

Graph `meta.json` is as follows with all pieces of text replaced by integers,

.. code-block:: text

	{
	"binary_data_version": binary_data_version,
	"node_count": node_count,
	"edge_count": edge_count,
	"node_type_count": node_type_count,
	"edge_type_count": edge_type_count,
	"node_count_per_type": [node_count_per_type_0, ..., node_count_per_type_n],
	"edge_count_per_type": [edge_count_per_type_0, ..., edge_count_per_type_n],
	"node_feature_count": node_feature_count,
	"edge_feature_count": edge_feature_count,
	"partitions": {
		"0": {
			"node_weight": [p0_node_type_0, ..., p0_node_type_n],
			"edge_weight": [p0_edge_type_0, ..., p0_edge_type_n],
		},
		"N": {
			"node_weight": [pN_node_type_0, ..., pN_node_type_n],
			"edge_weight": [pN_edge_type_0, ..., pN_edge_type_n],
		}
	},
	}

**************
fsspec support
**************

DeepGNN supports reading graph data from different data sources such as local file system, http, HDFS... This is done by using `fsspec` library. When loading data from a specific storage, user need to specify the full path of the graph data, for example, if the graph is in local file system, the path should be "/path/to/local/graph/data", if graph data is in HDFS, user need to specify `hdfs://domain/path/to/graph/data`. DeepGNN will try to parse the file protocol and download the graph data to local temporary folder, and local them into graph engine servers. More details can be found here: `_downloader.py <https://github.com/microsoft/DeepGNN/blob/main/src/python/deepgnn/graph_engine/snark/_downloader.py>`_.
Sample code:

.. code-block:: python

	import deepgnn.graph_engine.snark.server as server

	data_dir = "hdfs://my.hdfs.domain/data/cora"
	port = 12345
	s = server.Server(data_dir, [0], f"0.0.0.0:{port}")

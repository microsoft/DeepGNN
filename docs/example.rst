The ``snark`` module
======================

How to generate ``graph engine`` binary files from a ``networkx`` graph
-------------------

First import ``networkx`` and other relevant modules:

    >>> import random
    >>> import json
    >>> import tempfile
    >>> from pathlib import Path

    >>> import networkx as nx
    >>> import numpy as np
    >>> import deepgnn.graph_engine.snark.convert as convert
    >>> from deepgnn.graph_engine.snark.decoders import DecoderType
    >>> import deepgnn.graph_engine.snark.client as client

We are going to generate a random graph with 30 clusters, each cluster contains exactly 12 nodes.
Nodes are grouped together by id, i.e. first cluster contains nodes [0-11], second has [12-23], etc.:

    >>> random.seed(246)
    >>> g = nx.connected_caveman_graph(30, 12)

We need to assign some features for every node to train the model and to keep things simple we are going
to use random integers and node ids as values for a 2 dimensional feature vector.

    >>> nodes = []
    >>> data = ""


    >>> for node_id in g:
    ...   # Set weights for neighbors
    ...   nbs = {}
    ...   for nb in nx.neighbors(g, node_id):
    ...     nbs[nb] = 1.
    ...
    ...   node = {
    ...     "node_weight": 1,
    ...     "node_id": node_id,
    ...     "node_type": 0,
    ...     "uint64_feature": {},
    ...     "float_feature": {"0": [node_id, random.random()]},
    ...     "binary_feature": {},
    ...     "edge": [{"src_id": node_id, "dst_id": nb, "edge_type": 0, "weight": 1., "uint64_feature": {}, "float_feature": {}, "binary_feature": {}}
    ...       for nb in nx.neighbors(g, node_id)],
    ...   }
    ...   data += json.dumps(node) + "\n"
    ...   nodes.append(node)

We can inspect values of node features:
    >>> nodes[1]["float_feature"]["0"]
    [1, 0.516676816253458]

In order to use this graph we need to create a file with metadata to convert json to the binary data format:

    >>> meta = '{"node_float_feature_num": 1, \
    ...         "edge_binary_feature_num": 0, \
    ...         "edge_type_num": 1, \
    ...         "edge_float_feature_num": 0, \
    ...         "node_type_num": 1, \
    ...         "node_uint64_feature_num": 0, \
    ...         "node_binary_feature_num": 0, \
    ...         "edge_uint64_feature_num": 0}'
    >>> working_dir = tempfile.TemporaryDirectory()

    >>> raw_file = working_dir.name + "/data.json"
    >>> with open(raw_file, "w+") as f:
    ...     f.write(data)
    614386

    >>> meta_file = working_dir.name + "/meta.json"
    >>> with open(meta_file, "w+") as f:
    ...     f.write(meta)
    274

Now we can convert graph to binary data:
    >>> convert.MultiWorkersConverter(
    ...    graph_path=raw_file,
    ...    meta_path=meta_file,
    ...    partition_count=1,
    ...    output_dir=working_dir.name,
    ...    decoder_type=DecoderType.JSON,
    ... ).convert()

Create a client to use from the temp folder:
    >>> cl = client.MemoryGraph(working_dir.name, [0])
    >>> cl.node_features(np.array([1], dtype=np.int64), features=np.array([[0, 2]], dtype=np.int32), dtype=np.float32)
    array([[1.        , 0.51667684]], dtype=float32)

With large graphs we might want to work with samplers to train our models:
    >>> ns = client.NodeSampler(cl, types=[0])
    >>> ns.sample(size=2, seed=1)
    (array(...))

Edge samplers are very similar to the node ones:
    >>> es = client.EdgeSampler(cl, types=[0])
    >>> es.sample(size=2, seed=2)
    (array(...))

.. todo::
    alsamylk: add distributed example once we have more user friendly conversions from networkx graph.

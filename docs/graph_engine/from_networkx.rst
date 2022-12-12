****************************************************
How to create graph engine from a ``networkx`` graph
****************************************************

First import ``networkx`` and other relevant modules:
.. code-block:: python

	>>> import random
	>>> import json
	>>> import tempfile
	>>> from pathlib import Path
	>>> import networkx as nx
	>>> import numpy as np
	>>> import deepgnn.graph_engine.snark.convert as convert
	>>> from deepgnn.graph_engine.snark.decoders import JsonDecoder
	>>> import deepgnn.graph_engine.snark.client as client

We are going to generate a random graph with 30 clusters, each cluster contains exactly 12 nodes.

Nodes are grouped together by id, i.e. first cluster contains nodes [0-11], second has [12-23], etc.:

.. code-block:: python

	>>> random.seed(246)
	>>> g = nx.connected_caveman_graph(30, 12)

We need to assign some features for every node to train the model and to keep things simple we are going
to use random integers and node ids as values for a 2 dimensional feature vector.

.. code-block:: python

	>>> nodes = []
	>>> data = ""
	>>> for node_id in g:
	...   # Set weights for neighbors
	...   nbs = {}
	...   for nb in nx.neighbors(g, node_id):
	...     nbs[nb] = 1.
	...
	...   # Fill detailed data for the node
	...   node = {
	...     "node_weight": 1,
	...     "node_id": node_id,
	...     "node_type": 0,
	...     "float_feature": {"0": [node_id, random.random()]},
	...     "edge": [{
	...         "src_id": node_id,
	...         "dst_id": nb,
	...         "edge_type": 0,
	...         "weight": 1.
	...       }
	...       for nb in nx.neighbors(g, node_id)],
	...   }
	...   data += json.dumps(node) + "\n"
	...   nodes.append(node)

We can inspect values of node features:

.. code-block:: python

	>>> nodes[1]["float_feature"]["0"]
	[1, 0.516676816253458]

	>>> working_dir = tempfile.TemporaryDirectory()
	>>> raw_file = working_dir.name + "/data.json"
	>>> with open(raw_file, "w+") as f:
	...     f.write(data)
	287274

Now we can convert graph to binary data:

.. code-block:: python

	>>> convert.MultiWorkersConverter(
	...    graph_path=raw_file,
	...    partition_count=1,
	...    output_dir=working_dir.name,
	...    decoder=JsonDecoder,
	... ).convert()

Create a client to use from the temp folder:

.. code-block:: python

	>>> cl = client.MemoryGraph(working_dir.name)
	>>> cl.node_features(nodes=[1], features=[[0, 2]], dtype=np.float32)
	array([[1.        , 0.51667684]], dtype=float32)

With large graphs we might want to work with samplers to train our models:

.. code-block:: python

	>>> ns = client.NodeSampler(cl, types=[0])
	>>> ns.sample(size=2, seed=1)
	(array([ 68, 242]), array([0, 0], dtype=int32))

The first item in a tuple, `[68, 242]` is a list of sampled nodes and the second item is their corresponding types(all zeros).
Edge samplers are very similar to the node ones:

.. code-block:: python

	>>> es = client.EdgeSampler(cl, types=[0])
	>>> es.sample(size=2, seed=2)
	(array([292,  53]), array([298,  54]), array([0, 0], dtype=int32))

The returned result is a triple of lists with source nodes, destination nodes and edge types.

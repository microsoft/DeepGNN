Distributed conversion
======================

We are going to use pyspark to split graph into multiple partitions, convert each partition into a binary file and then load graph in memory to extract some features.

Data preparation
----------------

Graph will be stored in a json format, each line in the file represents a node, it's features and all edges outgoing from this node.
To keep things simple, we'll create a random graph with 100 nodes and ~3K edges. Each node and edge will have 3 different types to mimic
train/test/validation data split typically used for training models.

.. code-block:: python

    >>> import json
    >>> import random
    >>> import os
    >>> import numpy as np
    >>> NODE_COUNT = 100
    >>> MAX_NODE_ID = 10000
    >>> graph_file = "graph.json"
    >>> binary_dir = "binary_data"
    >>> os.makedirs(binary_dir)
    >>> random.seed(42)
    >>> np.random.seed(42)
    >>> with open(graph_file, "w+") as f:
    ...    edge_types = [0, 1, 2]
    ...    lines = []
    ...    for i in range(NODE_COUNT):
    ...         src = random.randint(0, MAX_NODE_ID)
    ...         node = {
    ...             "node_id": src,
    ...             "node_type": 0,
    ...             "node_weight": 1,
    ...             "float_feature": {"0": [src, i], "1": np.random.random(size=64).tolist()},
    ...             "edge": [],
    ...         }
    ...         for x in edge_types:
    ...             edge_count = random.randint(0, 20)
    ...             for edge in range(edge_count):
    ...                 dst = random.randint(0, MAX_NODE_ID)
    ...                 node["edge"].append(
    ...                     {
    ...                         "src_id": src,
    ...                         "dst_id": dst,
    ...                         "edge_type": x,
    ...                         "weight": 1,
    ...                     }
    ...                 )
    ...         lines.append(json.dumps(node))
    ...    f.write("\n".join(lines))
    333093

Each node was assigned a random id from interval `[0, MAX_NODE_ID)` and we put original processing order in the feature with `id=0`.
For example the last processed node has id `2693`.

.. code-block:: python

    >>> node["float_feature"]["0"]
    [2693, 99]

Each partition requires a separate metadata file with global information about the graph subset it represents. `PartitionMeta` class will store this data and
increment relevant counters for each node and edge for each processed node. When conversion is over, a spark task will call `close` method to dump this data to a file.

.. code-block:: python

    >>> import itertools
    >>> from deepgnn.graph_engine.snark.meta import BINARY_DATA_VERSION
    >>> class PartitionMeta:
    ...     def __init__(self, partition_id:int):
    ...         self.node_count = 0
    ...         self.edge_count = 0
    ...         self.node_type_count = 3
    ...         self.edge_type_count = 3
    ...         self.node_feature_count = 2
    ...         self.edge_feature_count = 0
    ...         self.partition_count = 1
    ...         self.partition_ids = [partition_id]
    ...         self.node_weights = [0, 0, 0]
    ...         self.edge_weights = [0, 0, 0]
    ...         self.node_count_per_type = [0, 0, 0]
    ...         self.edge_count_per_type = [0, 0, 0]
    ...
    ...     def add(self, node):
    ...         self.node_count += 1
    ...         self.node_weights[node["node_type"]] += node["node_weight"]
    ...         self.node_count_per_type[node["node_type"]] += 1
    ...         for edge in node["edge"]:
    ...             self.edge_count += 1
    ...             self.edge_weights[edge["edge_type"]] += edge["weight"]
    ...             self.edge_count_per_type[edge["edge_type"]] += 1
    ...
    ...     def close(self, binary_dir: str):
    ...         with open(os.path.join(binary_dir, "meta_%d.txt" % self.partition_ids[0]), "w+") as f:
    ...             contents = "\n".join(list(map(str, itertools.chain(
    ...                 [BINARY_DATA_VERSION,
    ...                 self.node_count,
    ...                 self.edge_count,
    ...                 self.node_type_count,
    ...                 self.edge_type_count,
    ...                 self.node_feature_count,
    ...                 self.edge_feature_count,
    ...                 self.partition_count,
    ...                 self.partition_ids[0]],
    ...                 self.node_weights,
    ...                 self.edge_weights,
    ...                 self.node_count_per_type,
    ...                 self.edge_count_per_type))))
    ...             f.write(contents)

Spark task is very straitforward: deserialize node from json and pass it to both `BinaryWriter` to generate binary data and `PartitionMeta` to update metadata.

.. code-block:: python

    >>> from pyspark import TaskContext
    >>> from deepgnn.graph_engine.snark.converter.writers import BinaryWriter
    >>> from deepgnn.graph_engine.snark.decoders import JsonDecoder
    >>> class SparkTask:
    ...     def __init__(self, binary_dir: str):
    ...         self.binary_dir = binary_dir
    ...
    ...     def __call__(self, iterator):
    ...         tc = TaskContext()
    ...         id = tc.partitionId()
    ...         decoder = JsonDecoder()
    ...         writer = BinaryWriter(self.binary_dir, id)
    ...         pm = PartitionMeta(id)
    ...         for n in iterator:
    ...             writer.add(decoder.decode(n))
    ...             pm.add(json.loads(n))
    ...         writer.close()
    ...         pm.close(self.binary_dir)

We can now run the job and split it across `NUM_PARTITIONS`:

.. code-block:: python

    >>> from pyspark.sql import SparkSession
    >>> import deepgnn.graph_engine.snark.meta_merger as meta_merger
    >>> from deepgnn.graph_engine.snark.client import MemoryGraph
    >>> NUM_PARTITIONS = 4
    >>> spark = SparkSession.builder.appName("deepgnn.distributed.convert").getOrCreate()
    >>> rdd = spark.sparkContext.textFile(graph_file)
    >>> rdd.repartition(NUM_PARTITIONS).foreachPartition(SparkTask(binary_dir))
    >>> meta_merger.merge_metadata_files(binary_dir)

Lets validate the graph loaded correctly by extracting node features from the node we used in the very beginning:

.. code-block:: python

    >>> graph = MemoryGraph(binary_dir, list(range(NUM_PARTITIONS)))
    >>> graph.node_features([2693], [[0, 2]], dtype=np.float32)
    array([[2693.,   99.]], dtype=float32)

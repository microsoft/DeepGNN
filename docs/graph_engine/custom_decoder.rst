*****************************************************
How to Create a Custom Decoder from Parquet to Binary
*****************************************************

In this example we take a Parquet file as input, write a Parquet to binary
decoder and then use this decoder in the MultiWorkersConverter to generate binaries
that can be loaded by our graph engine.

First we generate an example Parquet file to use. This is a table with the columns "src", "dst",
"type", "weight", "f0_0", "f0_1" and "i1_0". Each row represents one record, either a node or an
edge. It is important we never have a record split accross multiple rows, there can be multiple
records in a row but they all must be fully contained in the row.

In this file we have two nodes, 0 and 1, on lines 0 and 2 in which the src column gives their
node ID and the dst is -1. We also have an edge in each direction between the two on lines 1 and 3.
It is important that we order these rows specifically, first with a node then all of its outgoing
edges in order of type then dst, then the next node and so on.
Each of these items have four fields: a type, weight, float feature vector and integer feature vector.
For any graph you use, all items are required to have a type and weight, then they can have a variable
number of indexable feature vectors.

.. code-block:: python

    >>> import tempfile
    >>> data = """src dst type weight f0_0 f0_1 i1_0
    ... 0 -1 0 0.5 3.5 6.0 0
    ...	0 1 0 1.0 5.1 4.4 0
    ... 1 -1 0 1.0 2.0 3.0 1
    ...	1 0 0 1.0 5.1 4.4 1"""
    >>> working_dir = tempfile.TemporaryDirectory()
    >>> raw_file = working_dir.name + "/data.parquet"
    >>> with open(raw_file, "w") as f:
    ...		f.write(data)
    1...

Next we write the ParquetDecoder that will be used to decode our input file
line by line to yield node and edge information sequentially. We keep it simple
and just use string manipulations to handle this.
The decoder uses yields to return information, this means you can
yield the information about more than one node or edge per a single input line.
If you wish to ignore a line and return nothing, just use an empty return.

The yield format is just like the columns in the Parquet file:
src_id: int, dst_id: int, type: int, weight: float, and features: list[ndarray, string or None].
In this example each item has two feature vectors, at index 0 is a 2 dim float vector and at
index 1 is a 1 dim int64 feature which are both encoded as ndarrays.
Each node and edge can have different types of vectors at each index.
None can be used to skip defining a feature at a specific index, so you can define a feature at
index 0 and 2 but skip index 1.

.. code-block:: python

    >>> from deepgnn.graph_engine.snark.decoders import Decoder
    >>> from typing import Iterator, Tuple
    >>> class ParquetDecoder(Decoder):
    ...     def __init__(self):
    ...         # Args can be added to init and added before passing to MultiWorkersConverter
    ...         pass
    ...
    ...     def decode(self, line: str) -> Iterator[Tuple[int, int, int, float, list]]:
    ...         # For a node we will yield: (node_id, -1, node_type, node_weight, [feature_0, ...])
    ...         # For an edge: (src_id, dst_id, edge_type, edge_weight, [feature_0, ...])
    ...         # We can return multiple items per line, but the order of the file, described above, must be maintained.
    ...         src, dst, typ, weight, *features = line.split(" ")
    ...         try:
    ...             float_feature = np.array(features[:2], dtype=np.float32)
    ...             int_feature = np.array(features[2], dtype=np.int64)
    ...             yield int(src), int(dst), int(typ), float(weight), [float_feature, int_feature]
    ...         except ValueError:  # if is header line
    ...             return

Finally, we execute the MultiWorkersConverter on our input file using our custom decoder to generate binaries.
Notably if you set the partition_count greater than 1, each partition will create a separate decoder and writer
in which the file will be split line by line between the decoders.

.. code-block:: python

    >>> import deepgnn.graph_engine.snark.convert as convert
    >>> import deepgnn.graph_engine.snark.client as client
    >>> import numpy as np
    >>> convert.MultiWorkersConverter(
    ...    graph_path=raw_file,
    ...    partition_count=1,
    ...    output_dir=working_dir.name,
    ...    decoder=ParquetDecoder(),
    ... ).convert()

We load the generated binaries into a graph engine and demonstrate it working.

.. code-block:: python

    >>> cl = client.MemoryGraph(working_dir.name, [0])
    >>> cl.node_features(nodes=[0, 1], features=[[0, 2]], dtype=np.float32)
    array([[3.5, 6. ],
           [2. , 3. ]], dtype=float32)
    >>> cl.node_features(nodes=[0, 1], features=[[1, 1]], dtype=np.int64)
    array([[0],
           [1]])

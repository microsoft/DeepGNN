**************************************************
How to Create a Custom Decoder from Avro to Binary
**************************************************

In this example we take an Avro file as input, write a Avro to binary
decoder and then use this decoder in to generate binaries
that can be loaded by our graph engine.

First we generate an example Avro file to use. This is a binary format with a schema to
define the columns and their types and a binary file where each row is a record.
This table has the columns: "src", "dst", "type", "weight", "float_feature" and "int_feature".
Each row represents one record, either a node or an edge.

In this file we have two nodes, 0 and 1, on lines 0 and 2 in which the src column gives their
node ID and the dst is -1. We also have an edge in each direction between the two on lines 1 and 3.
It is important that we order these rows specifically, first with a node then all of its outgoing
edges in order of type then dst, then the next node and so on.
Each of these items have four fields: a type, weight, float feature vector and integer feature vector.
For any graph you use, all items are required to have a type and weight, then they can have a variable
number of indexable feature vectors.

.. code-block:: python

    >>> import tempfile
    >>> import avro.schema
    >>> from avro.datafile import DataFileReader, DataFileWriter
    >>> from avro.io import DatumReader, DatumWriter
    >>> working_dir = tempfile.TemporaryDirectory()
    >>> raw_file = working_dir.name + "/data.avro"

    >>> schema = avro.schema.parse("""{"namespace": "example.avro", "type": "record", "name": "Graph", "fields": [
    ... {"name": "src", "type": "int"},
    ... {"name": "dst",  "type": "int"},
    ... {"name": "type", "type": "int"},
    ... {"name": "weight", "type": "float"},
    ... {"name": "float_feature", "type": {"type": "array", "items": "float", "default": []}},
    ... {"name": "sparse_int16_coords", "type": {"type": "array", "items": "int", "default": []}},
    ... {"name": "sparse_int16_feature", "type": {"type": "array", "items": "int", "default": []}}
    ... ]}""")

    >>> data = [  # src, node_type/dst, -1/edge_type, weight, feature_float, sparse_feature_int16
    ...     (0, 0, -1, 1.0, [3.5, 6.], ([7, 3], [255, 16])),  # Node 0
    ...     (1, 0, -1, 1.0, [2., 3.], ([6, 3], [100, 101])),  # Node 1
    ...     (0, 1, 0, 1.0, [4., 7.], ([5, 1], [16, 244])),  # Edge from 0 to 1
    ...     (1, 0, 0, 1.0, [5., 1.], ([2, 3], [51, 255])),  # Edge from 1 to 0
    ... ]

    >>> data = sorted(data)
    >>> data
    [(0, 0, -1, 1.0, [3.5, 6.0], ([7, 3], [255, 16])), (0, 1, 0, 1.0, [4.0, 7.0], ([5, 1], [16, 244])), (1, 0, -1, 1.0, [2.0, 3.0], ([6, 3], [100, 101])), (1, 0, 0, 1.0, [5.0, 1.0], ([2, 3], [51, 255]))]

    >>> writer = DataFileWriter(open(raw_file, "wb"), DatumWriter(), schema)
    >>> for src, dst, typ, weight, feature_float, sparse_feature_int16 in data:
    ...     if typ == -1:  # if is node, flip type and dst
    ...         typ = dst
    ...         dst = -1
    ...     writer.append({"src": src, "dst": dst, "type": typ, "weight": weight,
    ...         "float_feature": feature_float,
    ...         "sparse_int16_coords": sparse_feature_int16[0],
    ...         "sparse_int16_feature": sparse_feature_int16[1]})
    >>> writer.close()

Next we write the AvroDecoder that will be used to decode our input file
record by record to yield node and edge information sequentially. We keep it simple
and just use dictionary lookups and create np arrays from the bytes.
The decoder uses yields to return information, this means you can
yield the information about more than one node or edge per a single input line.
If you wish to ignore a line and return nothing, just use an empty return.

The yield format is just like the columns in the Avro file:
src_id: int, dst_id: int, type: int, weight: float, and features: list[ndarray, string or None].
In this example each item has two feature vectors, at index 0 is a 2 dim float vector and at
index 1 is a 1 dim int64 feature which are both encoded as ndarrays.
Each node and edge can have different types of vectors at each index.
None can be used to skip defining a feature at a specific index, so you can define a feature at
index 0 and 2 but skip index 1.

.. code-block:: python

    >>> from deepgnn.graph_engine.snark.decoders import Decoder
    >>> from typing import Iterator, Tuple
    >>> import numpy as np
    >>> class AvroDecoder(Decoder):
    ...     def __init__(self):
    ...         # Args can be added to init and added before passing to MultiWorkersConverter
    ...         pass
    ...
    ...     def decode(self, line: dict) -> Iterator[Tuple[int, int, int, float, list]]:
    ...         # For a node we will yield: (node_id, -1, node_type, node_weight, [feature_0, ...])
    ...         # For an edge: (src_id, dst_id, edge_type, edge_weight, [feature_0, ...])
    ...         # We can return multiple items per line, but the order of the file, described above, must be maintained.
    ...         yield int(line["src"]), int(line["dst"]), int(line["type"]), float(line["weight"]), [np.array(line["float_feature"], dtype=np.float32), (np.array(line["sparse_int16_coords"], dtype=np.int64), np.array(line["sparse_int16_feature"], dtype=np.int16))]

Finally, we convert our input file to binaries using an Avro reader, a BinaryWriter and the AvroDecoder.
Here we only have one partition and therefore one binary writer, it is okay to have multiple binary writers
as long as the suffix value is incremented.

.. code-block:: python

    >>> from deepgnn.graph_engine.snark.converter.writers import BinaryWriter
    >>> reader = DataFileReader(open(raw_file, "rb"), DatumReader())
    >>> decoder = AvroDecoder()
    >>> writer = BinaryWriter(working_dir.name, "0")
    >>> for record in reader:
    ...     writer.add(decoder.decode(record))
    >>> reader.close()
    >>> writer.close()

Here we manually write a meta.json file for our graph engine to load.

.. code-block:: python

    >>> import json
    >>> with open(working_dir.name + "/meta.json", "w") as f:
    ...     content = {
    ...         "binary_data_version": "v2",  # converter version
    ...         "node_count": writer.node_count,
    ...         "edge_count": writer.edge_count,
    ...         "node_type_count": writer.node_type_num,
    ...         "edge_type_count": writer.edge_type_num,
    ...         "node_feature_count": writer.node_feature_count,
    ...         "edge_feature_count": writer.edge_feature_count,
    ...         "partitions": {"0": {"node_weight": writer.node_weight, "edge_weight": writer.edge_weight}},
    ...         "node_count_per_type": writer.node_type_count,
    ...         "edge_count_per_type": writer.edge_type_count,
    ...     }
    ...     f.write(json.dumps(content))
    291

We load the generated binaries into a graph engine and demonstrate it working.

.. code-block:: python

    >>> import deepgnn.graph_engine.snark.client as client
    >>> import numpy as np
    >>> cl = client.MemoryGraph(working_dir.name, [(working_dir.name, 0)])
    >>> cl.node_features(nodes=[0, 1], features=[[0, 2]], dtype=np.float32)
    array([[3.5, 6. ],
           [2. , 3. ]], dtype=float32)
    >>> indices, values, dimensions = cl.node_sparse_features(nodes=[0, 1], features=np.array([1], dtype=np.int32), dtype=np.int16)
    >>> indices
    [array([[0, 7],
           [0, 3],
           [1, 6],
           [1, 3]])]
    >>> values
    [array([255,  16, 100, 101], dtype=int16)]
    >>> dimensions
    array([1])

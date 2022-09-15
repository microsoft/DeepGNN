Example of using SQL database to work with graphs.
==================================================

For inference scenarios with data updated in real time, best practice is to combine original graph with an SQL database.
Python graph client interacts with servers through gRPC and we can implement graph engine service functionality with python.

First import ``grpc_tools`` and other relevant modules:

    >>> import grpc
    >>> import grpc_tools.command as protoc

Now we can generate python modules to work with protocol buffers and import them:

    >>> protoc.protoc.main(
    ...       [
    ...           "gen_proto_files",
    ...           "--grpc_python_out=.",
    ...           "--python_out=.",
    ...           "-Isrc/cc/lib/distributed/",
    ...           "src/cc/lib/distributed/service.proto",
    ...       ]
    ...   )
    0

    >>> import service_pb2_grpc
    >>> import service_pb2
    >>> import sqlite3
    >>> from concurrent import futures

Data will be stored in a table with 2 columns: node_id and feature. First column represents a node and second is a blob to store features.

    >>> DB_NAME = "example.db"
    >>> con = sqlite3.connect(DB_NAME)
    >>> with con:
    ...    con.execute("CREATE TABLE node_features (node_id INTEGER, feature BLOB)")
    <...>

Next task is to implement a python service to fetch node features from a SQL database.
In this example we'll use sqlite3 module to connect to a database and fetch features with a straightforward `SELECT` query.

    >>> class GraphEngineServicer(service_pb2_grpc.GraphEngineServicer):
    ...    def __init__(self, DB_NAME):
    ...        self.DB_NAME = DB_NAME
    ...
    ...    def GetNodeFeatures(self, request, context):
    ...        """Extract node features"""
    ...        cur = sqlite3.connect(self.DB_NAME).cursor()
    ...        feature_values = b""
    ...        offsets = []
    ...        for node in request.node_ids:
    ...            cur.execute(
    ...                "SELECT feature FROM node_features WHERE node_id=:id", {"id": node}
    ...            )
    ...            values = cur.fetchall()
    ...            if len(values) > 0:
    ...                offsets.append(len(feature_values))
    ...                feature_values += values[0][0]
    ...
    ...        return service_pb2.NodeFeaturesReply(
    ...            feature_values=feature_values, offsets=offsets
    ...        )
    ...
    ...    def GetMetadata(self, request, context):
    ...        """Global information about graph. Needed to initialize client."""
    ...        return service_pb2.MetadataReply(
    ...            version=1,
    ...            nodes=1,
    ...            edges=0,
    ...            node_types=1,
    ...            edge_types=1,
    ...            node_features=1,
    ...            partitions=1,
    ...            node_partition_weights=[1.0],
    ...            edge_partition_weights=[1.0],
    ...            node_count_per_type=[1],
    ...            edge_count_per_type=[1],
    ...        )


Start grpc service on port 50051 and use `example.db` for data.

    >>> server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    >>> service_pb2_grpc.add_GraphEngineServicer_to_server(GraphEngineServicer(DB_NAME), server)
    >>> server.add_insecure_port("[::]:50051")
    50051

    >>> server.start()

Create a graph engine client and connect it to the service above.

    >>> import struct
    >>> import numpy as np
    >>> import deepgnn.graph_engine.snark.client as client

    >>> c = client.DistributedGraph(["localhost:50051"])

Initially the `node_features` table is empty, so when we'll try to fetch features we expect to recive an empty feature vector.

    >>> c.node_features(np.array([1], dtype=np.int64), [[1, 2]], dtype=np.float32)
    array([[0., 0.]], dtype=float32)

Lets put some features in our database, add node with `id=1` and a feature vector with two elements: `13, 42`.

    >>> with con:
    ...    con.execute("insert into node_features values (?, ?)", (1, struct.pack("ff", 13, 42)))
    <...>

Now we expect to receive non-zero response for the request.

    >>> c.node_features([1], [[1, 2]], dtype=np.float32)
    array([[13., 42.]], dtype=float32)
    >>> server.stop(1)
    <...>

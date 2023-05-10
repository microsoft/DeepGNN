Temporal Graph Support in DeepGNN
=================================

DeepGNN supports temporal graphs, which are graphs with timestamps assigned to each element, such as nodes, edges, or features. This document explains how to use them in detail.

Temporal Nodes
--------------

Temporal nodes are nodes that can be added or removed from a graph at different times. The JSON schema for temporal nodes extends [the schema for regular nodes]((../../../docs/graph_engine/data_spec.rst) with two new fields: created_at and removed_at, with values recorded in `Unix time <https://en.wikipedia.org/wiki/Unix_time>`_ format. The following example demonstrates a temporal node in JSON format:

.. code-block:: json

	{
		"node_id": 42,
		"node_type": 0,
		"node_weight": 1.0,
		"binary_feature": {"0": "microsoft", "1": "xbox"},
		"created_at": 13,
		"removed_at": 23,
	}

This example describes a node with an ID of 42, two binary features, and a single outgoing edge. The node exists from time 13 to time 23. If the created_at field is not specified, it defaults to 0. If the removed_at field is not specified, the node is assumed to be always present after it was created. DeepGNN does not yet support changing node types over time. A workaround is to delete the node and create a new node with a new type or store the node type as a feature.

Temporal Edges
--------------

Temporal edges are edges that can be added or removed from the graph at specific times. Edges have two fields: created_at and removed_at, with values recorded in Unix time format. The following example extends the previous example to include temporal edges:

.. code-block:: json

	{
		"node_id": 42,
		"node_type": 0,
		"node_weight": 1.0,
		"created_at": 13,
		"removed_at": 23,
		"edge": [
			{
				"src_id": 42,
				"dst_id": 99,
				"edge_type": 0,
				"weight": 2.0,
				"created_at": 13,
				"removed_at": 15
			},
			{
				"src_id": 42,
				"dst_id": 99,
				"edge_type": 0,
				"weight": 2.0,
				"created_at": 17
			},
			{
				"src_id": 42,
				"dst_id": 101,
				"edge_type": 1,
				"weight": 1.0,
				"removed_at": 19
			}
		]
	}

The node with ID 42 has two outgoing edges with types 0 and 1. The first edge exists during intervals [13, 15) and [17, 23), while the second edge is created with the node at time 13 and removed at 19. If the created_at field is not specified, it defaults to the created_at of the node. Similarly, the default value for removed_at is the same as the removed_at of the node.

Temporal Features
-----------------

Temporal features are node or edge features that can change values over time. To support this behavior, the schema for existing features is expanded. For binary and dense features, the format includes a union of two types:

* A list of values for features that are present during the node/edge lifetime.
* A list of objects with three fields: values, created_at, and removed_at.

The default values for created_at and removed_at are based on the creation/removal of corresponding node or edge this feature belongs to. Sparse features are extended in a similar way, with the accepted types being a union of a sparse feature object and a list of such objects. The following example demonstrates the use of temporal features:

.. code-block:: json

	{
		"node_id": 42,
		"node_type": 0,
		"node_weight": 1.0,
		"binary_feature": {
			"0": [
				{"values": ["microsoft"], "created_at": 13, "removed_at": 20},
				{"values": ["xbox"], "created_at": 20}
			]
		},
		"created_at": 13,
		"removed_at": 23,
		"edge": [
			{
				"src_id": 42,
				"dst_id": 99,
				"edge_type": 0,
				"weight": 2.0,
				"created_at": 13,
				"removed_at": 23,
				"sparse_int32_feature": {
					"0": [
						{"coordinates": [[5, 13], [7, 25]], "values": [-1, 1024], "removed_at": 21},
						{"coordinates": [[4, 2], [1, 3]], "values": [2, 4], "created_at": 21}
					],
					"1": {"coordinates": [13, 42], "values": [1, 1]}
				}
			}
		]
	}

In the example above, there are three temporal features: a binary feature belonging to the node with ID 42 and two sparse integer features attached to an edge with source node 42 and destination node 99. Feature with ID 0 has two temporal values assigned to it, split around time 21. Feature with ID 1 is present during the entire life of the edge, from 13 to 23.

Edge List Format
================

The Edge List format supports temporal edges and nodes, but not features. The original order of the node first and then all outgoing edges remains the same. EdgeInfo and NodeInfo are extended with new optional items to include creation and removal times:

.. code-block:: text

	node_info: node_id,-1,node_type,node_weight,created_at,removed_at,<features>
	edge_info: src,edge_type,dst,edge_weight,created_at,removed_at,<features>

Binary conversion
=================

Decoders supporting temporal graphs have a new argument, `watermark`, which must be set for time information about the graph to be added to the binary format. The `watermark`` represents the latest timestamp recorded for the graph. In most cases, it can be set to any positive value. If you are using the `MultiWorkersConverter` class, the `watermark`` argument must also be set.

Python API
==========

Every graph engine method (except for fetching node types and node/edge counts) is amended with an extra argument, timestamps, to fetch information about a graph snapshot at a specific time.

For example, to fetch node neighbors from the sample above, you can use the following code:

.. code-block:: python

	graph.neighbors(nodes=np.array([42], dtype=np.int64), edge_types=np.array([0, 1], dtype=np.int32), timestamps=np.array([16], dtype=np.int64))

The code above will return just one node, 101, because node 99 was deleted at time 15 and added back only at time 17.

Similarly, features can be fetched at different timestamps as well:

.. code-block:: python

	graph.node_string_features(
		nodes=np.array([42, 42], dtype=np.int64),
		features=np.array([[0, 1]], dtype=np.int32),
		timestamps=np.array([13, 21], dtype=np.int64))

This call will return two strings: "microsoft" and "xbox".

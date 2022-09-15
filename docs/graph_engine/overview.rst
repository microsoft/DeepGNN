Graph engine overview
=====================

Graphs for GNN can be stored in different formats and in order to have ability to plug them in the existing
algorithms we define an API for basic operations in
`_base.py <https://github.com/microsoft/DeepGNN/blob/main/src/python/deepgnn/graph_engine/_base.py#L30>`_.
Typically nodes are represented as integers while edges are triple of integers:
(source node, destination node, edge type). Each node and edge can store multiple features that can be addressed
by an integer id. We support float/integer/binary feature types.


Graph as dataset
----------------

Typically we want to traverse all nodes or edges in a graph to train some model. A straightforward way to do it
in Pytorch is to subclass `IterableDataset <https://pytorch.org/docs/stable/data.html#iterable-style-datasets>`_
and provide a node/edge iterator.

Graph operations
----------------

Most of algorithms need information not only about a current node/edge, but also their neighborood.
Here are the descriptions of the functions that are part of the API:

- :code:`sample_nodes(size: int, node_types: Union[int, np.array], strategy: SamplingStrategy) -> np.array`

  Return an array of nodes with a specified type. This operation is helpful to provide negative samples
  for unsupervised training.

- :code:`sample_edges(size: int, edge_types: Union[int, np.array], strategy: SamplingStrategy) -> np.array`

  Return an array of edges with a specified type.

- :code:`sample_neighbors(self, nodes: np.array, edge_types: np.array, count: int = 10, strategy: string = byweight, default_node: int = -1, default_weight: float = 0.0, default_node_type: int = -1) -> Tuple[np.array, np.array, np.array, np.array]`

  Sample direct neighbors connected with one of the edge_types to the seed nodes. Count parameter specifies how many neighbors to sample and it is up to implementation how to fill that array.
  Online engine for example might use repeatition in case of a node with fewer neighbors than `count` parameter.
  Function returns `count` neighbors per `nodes` element that are connected to the node with edge `type`. This function
  doesn't specify whether sampling will be with or without repetition. Online graph samples with repetitions
  and in case of nodes without any neighbors node ids will be `-1`. The existing algorithms don't have explcit
  requirement to use unique nodes.

- :code:`node_features(nodes: np.array, features: List[(id, dim)], feature_type: dtype) -> np.array`

  Fetch node features specified by ids. All the features should share the same type. The return type is an array
  of shape `(len(nodes), sum(map(lambda f: f[1], features)))` where features values are placed in the same order
  as in the features list. It is also used to fetch labels for supervised training.

  Example: how to extract multiple features with different dimensions in a single graph call. Lets extract
  feature with *id 0* and dimension **64**, multiply by some matrix with dim **64x32** (so the result is
  **1x32** vector) and then do some aggregation via a dot product on a feature vector with *id 1* and dimension
  **32**, so the result is a single number per node.

  .. code-block:: python

      >>> nodes = graph.sample_nodes(10, 0)
      >>> weights = np.random.rand(64,32)
      >>> features = graph.node_features(nodes, [(0, 64), (1, 32)], np.float32)
      >>> embed = torch.mm(features[:,0:64], weights)
      >>> result = torch.mm(embed, features[:, 64:])


- :code:`edge_features(edges: np.array, features: List[(id, dim)], feature_type: dtype) -> np.array`

  Similar function to node_features, but for edges.

- :code:`node_count(types: Union[int, np.array]) -> int`

  Return the number of nodes for the specific node types.

- :code:`edge_count(types: Union[int, np.array]) -> int`

  Return the number of edge for the specific edge types.

- :code:`random_walk(self, nodes: np.array, edge_types: np.array, walk_len: int, p: float, q: float, default_node: int = -1, seed: Optional[int] = None) -> np.array`

  Execute a random walk starting with nodes given. A random walk is equivalent to repeatedly calling sample_neighbors walk_len times, each time setting nodes to be the newly sampled neighbors.

  For the p return parameter, 1/p is the probability to return a parent node.
  For the q in-out parameter, 1/q is the probability to select a neighbor not connected to a parent.

  Return is ndarray with shape [nodes.size, walk_len+1] as the original nodes start the sampled walk.

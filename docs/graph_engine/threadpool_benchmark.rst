Graph Engine Thread Pool Benchmark
==================================

Graph engine enable thread pool automatically when necessary, here are some bechmark data about the thread pool:

GRPC GetNodeFeatures
--------------------

.. csv-table:: GRPC GetNodeFeatures
   :file: benchmarks/grpc_get_node_feature_threadpool.csv
   :header-rows: 1


GRPC SampleNeighbors
--------------------

.. csv-table:: GRPC SampleNeighbors
   :file: benchmarks/grpc_neighbor_threadpool_benchmark.csv
   :header-rows: 1


GRPC Multiple Servers GetNodeFeatures
-------------------------------------

.. csv-table:: GRPC Multiple Servers GetNodeFeatures
   :file: benchmarks/grpc_multiple_node_features_threadpool.csv
   :header-rows: 1


Local GetNodeFeatures
---------------------

.. csv-table:: Local GetNodeFeatures
   :file: benchmarks/local_get_node_feature_threadpool.csv
   :header-rows: 1


Local SampleNeighbors
---------------------

.. csv-table:: Local SampleNeighbors
   :file: benchmarks/local_neighbor_threadpool_benchmark.csv
   :header-rows: 1


Load Graph
----------

.. csv-table:: Load graph
   :file: benchmarks/load_graph_threadpool_benchmark.csv
   :header-rows: 1

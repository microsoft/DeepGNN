Welcome to DeepGNN's documentation!
===================================
DeepGNN is a package for training/evaluating ML models on graph data. It is a Python library that provides:

* A graph engine object designed for ML tasks with an assortment of routines for sampling nodes, edges and neighbors as well as feature fetching.
* Various aggregators, encoders and decoders to pass graph data to neural nets.
* Basic NN layers for training: convolution, attention and bindings to pytorch-geometric library.
* A collection of trainers to work with models in local and distributed environments.

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :titlesonly:

   graph_engine/index

   torch/index
   tf/index

   advanced/index

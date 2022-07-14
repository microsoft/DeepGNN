Welcome to DeepGNN's documentation!
===================================
DeepGNN is a package for training/evaluating ML models on graph data. It is a Python library that provides:

* a graph engine object to fetch data about nodes/edges and an assortment of routines for sampling neighbors/nodes/edges.
* various aggregators, encoders and decoders to pass graph data to neural nets.
* basic NN layers for training: convolution, attention and bindings to pytorch-geometric library.

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :titlesonly:

   Quick start <torch/quickstart>

   graph_engine/index

   torch/index
   tf/index

   advanced/index

***********************************
Ray Dataset and Data Pipeline Usage
***********************************

In this guide we show how to create and use Ray Datasets and Data Pipelines with the DeepGNN graph engine.
We show several different sampling strategies and advanced usage.

Generate Dataset
================

First we generate a Cora dataset to use in our examples.

.. code-block:: python

    >>> import numpy as np
    >>> import ray
    >>> from deepgnn.graph_engine.snark.local import Client

    >>> import tempfile
    >>> from deepgnn.graph_engine.data.citation import Cora
	>>> data_dir = tempfile.TemporaryDirectory()
    >>> Cora(data_dir.name)  # (Train: 140, Valid: 500, Test: 1000)
    <deepgnn.graph_engine.data.citation.Cora object at 0x...>

Simple Cora Dataset
================


In this example we have a dataset to generate initial samples of node ids.
These samples are pipelined into a mapping function to gather a dictionary of
node features and labels which are given to the model.

Ray pipelines data in terms of windows, windows set discrete portions of the dataset that will be loaded at a time // pre-processed in the background. Read more https://docs.ray.io/en/latest/data/pipelining-compute.html#pipelining-datasets
* As a rule of thumb, higher parallelism settings perform better, however blocks_per_window == num_blocks effectively disables pipelining, since the DatasetPipeline will only contain a single Dataset.
The other extreme is setting blocks_per_window=1, which minimizes the latency to initial output but only allows one concurrent transformation task per stage:
* As a rule of thumb, the cluster memory should be at least 2-5x the window size to avoid spilling.

Parrelelism for all datasets muust be manually set =1 for graph engine usage. May run into rare conflicts
with multiple GE calls happening at once if not specified.

Train test splis
https://docs.ray.io/en/latest/data/api/dataset_pipeline.html#splitting-datasetpipelines

.. code-block:: python

    # Setup initial node index samples, just integers of node ids
    >>> dataset = ray.data.range(2708, parallelism=1)
    >>> dataset
    Dataset(num_blocks=1, num_rows=2708, schema=<class 'int'>)

    # Convert dataset to a pipeline to pipeline stages
    >>> pipe = dataset.window(blocks_per_window=2)
    >>> pipe
    DatasetPipeline(num_windows=1, num_stages=2)

    # Map input indicies to a dict of node features and labels. This will run during iteration, not all at once.
    >>> def transform_batch(idx: list) -> dict:
    ...     g = Client(data_dir.name, [0])
    ...     return {"features": g.node_features(idx, np.array([[1, 50]]), feature_type=np.float32), "labels": np.ones((len(idx)))}
    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=1, num_stages=3)

    # Fetch the size of the dataset
    >>> size = dataset.count()

    #>>> train_dataloader, test_dataloader = pipe.split_at_indices([int(size * .5)])

    # Iterate over dataset n_epochs times
    >>> n_epochs = 1
    >>> epoch_pipe = next(pipe.repeat(n_epochs).iter_epochs())

    # Iterate over epoch and shuffle windows each time, use windowed shuffling to maintain pipeline windows
    >>> batch_size = 2
    >>> batch = next(epoch_pipe.random_shuffle_each_window(seed=100).iter_torch_batches(batch_size=batch_size))
    >>> batch
    {'features': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 'labels': tensor([1., 1.], dtype=torch.float64)}

File Node Sampler Dataset
================

File node sampler, memory efficient.

.. code-block:: python

    >>> dataset = ray.data.read_text("/tmp/cora/train.nodes", parallelism=1)
    >>> dataset
    Dataset(num_blocks=1, num_rows=140, schema=<class 'str'>)

    >>> pipe = dataset.window(blocks_per_window=2)   # This turns it into a pipeline thtat pipelines data functions instead of all at once, window is piopeline unit. block is parralelism unit.
    >>> pipe
    DatasetPipeline(num_windows=1, num_stages=1)

    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=1, num_stages=2)

# TODO add output check here

Weighted Sampler with Split on Train / Test nodes
================

For using diff types as diff modes

    # This pipeline has num_windows=None because it is streaming
iterator uses () so it is a gneerator
10 batches per ecpoh is run

- ALSO TODO need to handle 

.. code-block:: python

    >>> from ray.data import DatasetPipeline
    >>> from deepgnn.graph_engine import SamplingStrategy

    >>> g = Client(data_dir.name, [0])
    >>> node_batch_iterator = (lambda: ray.data.from_numpy(g.sample_nodes(140, np.array([0], dtype=np.int32), SamplingStrategy.Weighted)[0]) for _ in range(10))
    >>> pipe = DatasetPipeline.from_iterable(node_batch_iterator)
    >>> pipe
    DatasetPipeline(num_windows=None, num_stages=1)

    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=None, num_stages=2)

# TODO add output check here

Edge Sampling Dataset
=====================

In this example we have a dataset to generate initial samples of edge ids.
These samples are pipelined into a mapping function to gather a dictionary of
edge features and labels which are given to the model.

For more details on iteratoe see above example.

.. code-block:: python

    >>> from ray.data import DatasetPipeline
    >>> from deepgnn.graph_engine import SamplingStrategy

    >>> g = Client(data_dir.name, [0])
    >>> edge_batch_iterator = (lambda: ray.data.from_numpy(g.sample_edges(140, np.array([0], dtype=np.int32), SamplingStrategy.Weighted)) for _ in range(10))
    >>> pipe = DatasetPipeline.from_iterable(edge_batch_iterator)
    >>> pipe
    DatasetPipeline(num_windows=None, num_stages=1)

    # Map input indicies to a dict of node features and labels. This will run during iteration, not all at once.
    >>> def transform_batch(idx: list) -> dict:
    ...     g = Client(data_dir.name, [0])
    ...     return {"features": g.edge_features(idx, np.array([[0, 2]]), feature_type=np.float32), "labels": np.ones((len(idx)))}
    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=None, num_stages=2)

    #>>> train_dataloader, test_dataloader = pipe.split_at_indices([int(size * .5)])

    # Iterate over dataset n_epochs times
    >>> n_epochs = 1
    >>> #epoch_pipe = next(pipe.repeat(n_epochs).iter_epochs())

    # Iterate over epoch and shuffle windows each time, use windowed shuffling to maintain pipeline windows
    >>> batch_size = 2
    >>> batch = next(pipe.random_shuffle_each_window(seed=100).iter_torch_batches(batch_size=batch_size))
    >>> batch
    {'features': tensor([[0., 0.],
            [0., 0.]]), 'labels': tensor([1., 1.], dtype=torch.float64)}

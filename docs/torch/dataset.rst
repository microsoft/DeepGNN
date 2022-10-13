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

https://docs.ray.io/en/latest/data/api/dataset_pipeline.html#splitting-datasetpipelines

    # pipeline incrementally on windows of the base data. This can be used for streaming data loading into ML training,
    # or to execute batch transformations on large datasets without needing to load the entire dataset into cluster memory.
    # Create a dataset and then create a pipeline from it.

    # https://docs.ray.io/en/latest/data/pipelining-compute.html#pipelining-datasets
    # As a rule of thumb, higher parallelism settings perform better, however blocks_per_window == num_blocks effectively disables pipelining, since the DatasetPipeline will only contain a single Dataset.
    # The other extreme is setting blocks_per_window=1, which minimizes the latency to initial output but only allows one concurrent transformation task per stage:
    # As a rule of thumb, the cluster memory should be at least 2-5x the window size to avoid spilling.
    # TODO Check out the reported statistics for window size and blocks per window to ensure efficient pipeline execution.
    pipe = dataset.window(blocks_per_window=2)  # can be 10 or something

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
    >>> train_epoch_dataloader = pipe.repeat(n_epochs).iter_epochs():

    # Iterate over epoch and shuffle windows each time, use windowed shuffling to maintain pipeline windows
    >>> epoch_iterator = train_epoch_dataloader.random_shuffle_each_window().iter_torch_batches():
    >>> next(epoch_iterator)

File Node Sampler Dataset
================

File node sampler, memory efficient.


.. code-block:: python

    >>> train_dataset = ray.data.read_text("/tmp/cora/train.nodes", parallelism=1)
    >>> pipe = dataset.window(blocks_per_window=2)   # This turns it into a pipeline thtat pipelines data functions instead of all at once, window is piopeline unit. block is parralelism unit.
    >>> pipe
    DatasetPipeline(num_windows=1, num_stages=2)

    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=1, num_stages=3)

# TODO add output check here

Weighted Sampler with Split on Train / Test nodes
================

For using diff types as diff modes

.. code-block:: python

    >>> from ray.data.datasource import SimpleTorchDatasource
    >>> from deepgnn.graph_engine import SamplingStrategy
    >>> def generate_dataset():
    ...     g = Client(data_dir.name, [0])
    ...     return g.sample_nodes(2708, 0, SamplingStrategy.Weighted)[0]

    >>> ds = ray.data.read_datasource(
    ...     SimpleTorchDatasource(), parallelism=1, dataset_factory=generate_dataset
    ... )

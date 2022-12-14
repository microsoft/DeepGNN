***********************************
Ray Dataset and Data Pipeline Usage
***********************************

In this guide we show how to create and use `Ray Datasets <https://docs.ray.io/en/latest/data/dataset.html>`
and `Data Pipelines <https://docs.ray.io/en/latest/data/pipelining-compute.html#pipelining-datasets>`
with the DeepGNN graph engine.
We show several different sampling strategies and advanced usage.

Generate Dataset
================

First we generate a Cora dataset and load it into a server to use in our examples.
We load up a server, when ray tries to pull it into a function, it will pull a
client instead of the whole server.

.. code-block:: python

    >>> import numpy as np
    >>> import ray
    >>> from deepgnn.graph_engine.snark.distributed import Server, Client as DistributedClient

    >>> import tempfile
    >>> from deepgnn.graph_engine.data.citation import Cora
    >>> data_dir = tempfile.TemporaryDirectory()
    >>> Cora(data_dir.name)  # (Train: 140, Valid: 500, Test: 1000)
    <deepgnn.graph_engine.data.citation.Cora object at 0x...>

    >>> address = "localhost:9999"
    >>> g = Server(address, data_dir.name, 0, 1)

Simple Cora Dataset
===================

In this example we create a simple dataset using Ray Data.

First we initialize a Ray Dataset of node ids ranging from 0 to 2708.
`ray.data.range <https://docs.ray.io/en/latest/data/api/input_output.html#synthetic-data>`
Then we repartition it to be one block per batch.

.. code-block:: python

    >>> dataset = ray.data.range(2708).repartition(2708 // 512)
    >>> dataset
    Dataset(num_blocks=5, num_rows=2708, schema=<class 'int'>)

We convert this dataset to a data pipeline by splitting it into windows.

"Dataset pipelines allow Dataset transformations to be executed incrementally
on windows of the base data, instead of on all of the data at once.
This can be used for streaming data loading into ML training, or to execute batch
transformations on large datasets without needing to load the entire dataset into cluster memory."
More about dataset pipelines, `here <https://docs.ray.io/en/latest/data/pipelining-compute.html#pipelining-datasets>`.

* "As a rule of thumb, higher parallelism settings perform better, however blocks_per_window == num_blocks effectively disables pipelining, since the DatasetPipeline will only contain a single Dataset.
The other extreme is setting blocks_per_window=1, which minimizes the latency to initial output but only allows one concurrent transformation task per stage."
* "As a rule of thumb, the cluster memory should be at least 2-5x the window size to avoid spilling."

.. code-block:: python

    >>> pipe = dataset.window(blocks_per_window=2)
    >>> pipe
    DatasetPipeline(num_windows=3, num_stages=1)

In order to rerun this dataset multiple times, one per epoch, we use the repeat command.
In this example we call repeat before running any transforms on the dataset, therefore the transform outputs will not be cached between epochs.
If repeat is run after a transform, the result of the transform will be cached, `more here <https://docs.ray.io/en/latest/data/advanced-pipelines.html#dataset-pipeline-per-epoch-shuffle>`.

.. code-block:: python

    >>> n_epochs = 2
    >>> pipe = pipe.repeat(n_epochs)
    >>> pipe
    DatasetPipeline(num_windows=6, num_stages=1)

We add shuffling at this part of the dataset so that we only shuffle node ids, not the whole query return. It is important to add this after repeat so it does not get cached.

    >>> pipe = pipe.random_shuffle_each_window(seed=0)
    >>> pipe
    DatasetPipeline(num_windows=6, num_stages=2)

Use `map_batches <https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset.map_batches>`
to map node ids from the sampler to a dictionary of node features and labels for the model forward function.
Since this is run on the dataset pipeline, the node ids will not be mapped all at once, only when needed during iteration.

For each query output vector, each first dimension needs to be equal to the batch size == len(idx).

.. code-block:: python

    >>> def transform_batch(idx: list) -> dict:
    ...     return {"features": g.node_features(idx, np.array([[1, 50]]), feature_type=np.float32), "labels": np.ones((len(idx)))}
    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=6, num_stages=3)

Finally we iterate over the dataset n_epochs times.

.. code-block:: python

    >>> epoch_iter = pipe.iter_epochs()
    >>> epoch_pipe = next(epoch_iter)
    >>> batch = next(epoch_pipe.iter_torch_batches(batch_size=2))
    >>> batch
    {'features': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 'labels': tensor([1., 1.], dtype=torch.float64)}

    >>> epoch_pipe = next(epoch_iter)
    >>> batch = next(epoch_pipe.iter_torch_batches(batch_size=2))
    >>> batch
    {'features': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 'labels': tensor([1., 1.], dtype=torch.float64)}

File Node Sampler
=================

Here we replace the node id sampler with a file line sampler, `ray.data.read_text() <https://docs.ray.io/en/latest/data/api/input_output.html#ray.data.read_text>`.

.. code-block:: python

    >>> batch_size = 2
    >>> dataset = ray.data.read_text("/tmp/cora/train.nodes")
    >>> dataset = dataset.repartition(dataset.count() // batch_size)
    >>> dataset
    Dataset(num_blocks=70, num_rows=140, schema=<class 'str'>)

    >>> pipe = dataset.window(blocks_per_window=2)
    >>> pipe
    DatasetPipeline(num_windows=35, num_stages=1)

    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=35, num_stages=2)

    >>> batch = next(pipe.iter_torch_batches(batch_size=batch_size))
    >>> batch
    {'features': tensor([[3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 'labels': tensor([1., 1.], dtype=torch.float64)}

Graph Engine Node Sampler
=========================

In this example we use the graph engine `sample_nodes` function to generate inputs to the query function.
Since this method uses `DatasetPipeline.from_iterable <https://docs.ray.io/en/latest/data/api/dataset_pipeline.html#creating-datasetpipelines>`
with a generator as input, it streams the windows instead of loading them.

.. code-block:: python

    >>> from ray.data import DatasetPipeline
    >>> from deepgnn.graph_engine import SamplingStrategy

    >>> cl = DistributedClient([address])
    >>> node_batch_generator = (lambda: ray.data.from_numpy(cl.sample_nodes(140, np.array([0], dtype=np.int32), SamplingStrategy.Weighted)[0]) for _ in range(10))
    >>> pipe = DatasetPipeline.from_iterable(node_batch_generator)
    >>> pipe
    DatasetPipeline(num_windows=None, num_stages=1)

    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=None, num_stages=2)

    >>> batch = next(pipe.iter_torch_batches(batch_size=2))
    >>> batch
    {'features': tensor([[...]]), 'labels': tensor([1., 1.], dtype=torch.float64)}

Graph Engine Edge Sampler
=========================

In this example we use the graph engine `sample_edge` function to generate edge ids as inputs to the query function.
Since this method uses `DatasetPipeline.from_iterable <https://docs.ray.io/en/latest/data/api/dataset_pipeline.html#creating-datasetpipelines>`
with a generator as input, it streams the windows instead of loading them.

.. code-block:: python

    >>> from ray.data import DatasetPipeline
    >>> from deepgnn.graph_engine import SamplingStrategy

    >>> cl = DistributedClient([address])
    >>> edge_batch_generator = (lambda: ray.data.from_numpy(cl.sample_edges(140, np.array([0], dtype=np.int32), SamplingStrategy.Weighted)) for _ in range(10))
    >>> pipe = DatasetPipeline.from_iterable(edge_batch_generator)
    >>> pipe
    DatasetPipeline(num_windows=None, num_stages=1)

    >>> def transform_batch(idx: list) -> dict:
    ...     return {"features": g.edge_features(idx, np.array([[0, 2]]), feature_type=np.float32), "labels": np.ones((len(idx)))}
    >>> pipe = pipe.map_batches(transform_batch)
    >>> pipe
    DatasetPipeline(num_windows=None, num_stages=2)

    >>> batch = next(pipe.iter_torch_batches(batch_size=2))
    >>> batch
    {'features': tensor([[0., 0.],
            [0., 0.]]), 'labels': tensor([1., 1.], dtype=torch.float64)}

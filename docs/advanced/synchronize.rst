********************************
Synchronized Server-Client Setup
********************************

In this guide we show how to setup a N server to M client pair using our synchronization tools.
This will ensure that clients do not start until all servers are started and servers do not close
until all clients are done.

Cora Dataset
============

.. code-block:: python

    >>> import tempfile
	>>> from deepgnn.graph_engine.data.citation import Cora
    >>> data_dir = tempfile.TemporaryDirectory()
	>>> Cora(data_dir.name)
	<deepgnn.graph_engine.data.citation.Cora object at 0x...>

Synchronized
============

Import

.. code-block:: python

    >>> from deepgnn.graph_engine.snark.synchronized import start_servers, start_clients, train
    >>> from ray import workflow
    >>> import numpy as np

Define train function to use

.. code-block:: python

    >>> def train_fn(clients):
    ...     cl = clients[0]
    ...     result = cl.node_features(np.array([0, 1]), np.array([[1, 1]]), np.float32)
    ...     return result

Start up 2 servers

.. code-block:: python

    >>> servers = start_servers.bind("localhost:9999", data_dir.name, 2, 2)

Start 3 clients

.. code-block:: python

    >>> clients = start_clients.bind(servers, 3)

And train

.. code-block:: python

    >>> output = train.bind(servers, clients, train_fn)
    >>> result = workflow.run(output)

****************************
Node Classification with GAT
****************************

.. code-block:: python

    >>> from deepgnn.graph_engine.data.cora import CoraFull
    >>> CoraFull("/tmp/cora/")
    <deepgnn.graph_engine.data.cora.CoraFull object at ...>

.. code-block:: python

    >>> import argparse
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from dataclasses import dataclass
    >>> from typing import Dict, List, Union, Callable, Any, Tuple
    >>> from contextlib import closing
    >>> from deepgnn import str2list_int, setup_default_logging_config
    >>> from deepgnn.graph_engine import Graph, graph_ops
    >>> from deepgnn.graph_engine import (
    ...    SamplingStrategy,
    ...    GENodeSampler,
    ...    RangeNodeSampler,
    ...    FileNodeSampler,
    ...    BackendOptions,
    ...    create_backend,
    ... )
    >>> from deepgnn.tf import common
    >>> from deepgnn.tf.nn.gat_conv import GATConv
    >>> from deepgnn.tf.nn.metrics import masked_accuracy, masked_softmax_cross_entropy
    >>> from deepgnn.tf.common.dataset import create_tf_dataset, get_distributed_dataset
    >>> from deepgnn.tf.common.trainer_factory import get_trainer

.. code-block:: python

    >>> @dataclass
    ... class GATQueryParameter:
    ...    neighbor_edge_types: np.array
    ...    feature_idx: int
    ...    feature_dim: int
    ...    label_idx: int
    ...    label_dim: int
    ...    feature_type: np.dtype = np.float32
    ...    label_type: np.dtype = np.float32
    ...    num_hops: int = 2


.. code-block:: python

    >>> class GATQuery:
    ...    """Graph Query: get sub graph for GAT training"""
    ...
    ...    def __init__(self, param: GATQueryParameter):
    ...        self.param = param
    ...        self.label_meta = np.array([[param.label_idx, param.label_dim]], np.int32)
    ...        self.feat_meta = np.array([[param.feature_idx, param.feature_dim]], np.int32)
    ...
    ...    def query_training(
    ...        self, graph: Graph, inputs: np.array, return_shape: bool = False
    ...    ):
    ...        nodes, edges, src_idx = graph_ops.sub_graph(
    ...            graph=graph,
    ...            src_nodes=inputs,
    ...            edge_types=self.param.neighbor_edge_types,
    ...            num_hops=self.param.num_hops,
    ...            self_loop=True,
    ...            undirected=True,
    ...            return_edges=True,
    ...        )
    ...        input_mask = np.zeros(nodes.size, np.bool_)
    ...        input_mask[src_idx] = True
    ...
    ...        feat = graph.node_features(nodes, self.feat_meta, self.param.feature_type)
    ...        label = graph.node_features(nodes, self.label_meta, self.param.label_type)
    ...        label = label.astype(np.int32)
    ...
    ...        edges_value = np.ones(edges.shape[0], np.float32)
    ...        adj_shape = np.array([nodes.size, nodes.size], np.int64)
    ...        graph_tensor = (nodes, feat, input_mask, label, edges, edges_value, adj_shape)
    ...        if return_shape:
    ...            # fmt: off
    ...            # N is the number of `nodes`, which is variable because `inputs` nodes are different.
    ...            N = None
    ...            shapes = (
    ...                [N],                            # Nodes
    ...                [N, self.param.feature_dim],    # feat
    ...                [N],                            # input_mask
    ...                [N, self.param.label_dim],      # label
    ...                [None, 2],                      # edges
    ...                [None],                         # edges_value
    ...                [2]                             # adj_shape
    ...            )
    ...            # fmt: on
    ...            return graph_tensor, shapes
    ...
    ...        return graph_tensor


.. code-block:: python

    >>> class GAT(tf.keras.Model):
    ...    """ GAT Model (supervised)"""
    ...
    ...    def __init__(
    ...        self,
    ...        head_num: List[int] = [8, 1],
    ...        hidden_dim: int = 8,
    ...        num_classes: int = -1,
    ...        ffd_drop: float = 0.0,
    ...        attn_drop: float = 0.0,
    ...        l2_coef: float = 0.0005,
    ...    ):
    ...        super().__init__()
    ...        self.num_classes = num_classes
    ...        self.l2_coef = l2_coef
    ...
    ...        self.out_dim = num_classes
    ...
    ...        self.input_layer = GATConv(
    ...            attn_heads=head_num[0],
    ...            out_dim=hidden_dim,
    ...            act=tf.nn.elu,
    ...            in_drop=ffd_drop,
    ...            coef_drop=attn_drop,
    ...            attn_aggregate="concat",
    ...        )
    ...        ## TODO: support hidden layer
    ...        assert len(head_num) == 2
    ...        self.out_layer = GATConv(
    ...            attn_heads=head_num[1],
    ...            out_dim=self.out_dim,
    ...            act=None,
    ...            in_drop=ffd_drop,
    ...            coef_drop=attn_drop,
    ...            attn_aggregate="average",
    ...        )
    ...
    ...    def forward(self, feat, bias_mat, training):
    ...        h_1 = self.input_layer([feat, bias_mat], training=training)
    ...        out = self.out_layer([h_1, bias_mat], training=training)
    ...        #tf.compat.v1.logging.info("h_1 {}, out shape {}".format(h_1.shape, out.shape))
    ...        return out
    ...
    ...    def call(self, inputs, training=True):
    ...        # inputs: nodes    feat      mask    labels   edges       edges_value  adj_shape
    ...        # shape:  [N]      [N, F]    [N]     [N]      [num_e, 2]  [num_e]      [2]
    ...        nodes, feat, mask, labels, edges, edges_value, adj_shape = inputs
    ...
    ...        # bias_mat = -1e9 * (1.0 - adj)
    ...        sp_adj = tf.SparseTensor(edges, edges_value, adj_shape)
    ...        logits = self.forward(feat, sp_adj, training)
    ...
    ...        ## embedding results
    ...        self.src_emb = tf.boolean_mask(logits, mask)
    ...        self.src_nodes = tf.boolean_mask(nodes, mask)
    ...
    ...        labels = tf.one_hot(labels, self.num_classes)
    ...        logits = tf.reshape(logits, [-1, self.num_classes])
    ...        labels = tf.reshape(labels, [-1, self.num_classes])
    ...        mask = tf.reshape(mask, [-1])
    ...
    ...        ## loss
    ...        xent_loss = masked_softmax_cross_entropy(logits, labels, mask)
    ...        loss = xent_loss + self.l2_loss()
    ...
    ...        ## metric
    ...        acc = masked_accuracy(logits, labels, mask)
    ...        return logits, loss, {"accuracy": acc}
    ...
    ...    def l2_loss(self):
    ...        vs = []
    ...        for v in self.trainable_variables:
    ...            vs.append(tf.nn.l2_loss(v))
    ...        lossL2 = tf.add_n(vs) * self.l2_coef
    ...        return lossL2
    ...
    ...    def train_step(self, data: dict):
    ...        """override base train_step."""
    ...        with tf.GradientTape() as tape:
    ...            _, loss, metrics = self(data, training=True)
    ...
    ...        grads = tape.gradient(loss, self.trainable_variables)
    ...        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    ...        result = {"loss": loss}
    ...        result.update(metrics)
    ...        return result
    ...
    ...    def test_step(self, data: dict):
    ...        """override base test_step."""
    ...        _, loss, metrics = self(data, training=False)
    ...        result = {"loss": loss}
    ...        result.update(metrics)
    ...        return result
    ...
    ...    def predict_step(self, data: dict):
    ...        """override base predict_step."""
    ...        self(data, training=False)
    ...        return [self.src_nodes, self.src_emb]


.. code-block:: python

    >>> def build_model(param):
    ...    p = GATQueryParameter(
    ...        neighbor_edge_types=np.array(param.neighbor_edge_types, np.int32),
    ...        feature_idx=param.feature_idx,
    ...        feature_dim=param.feature_dim,
    ...        label_idx=param.label_idx,
    ...        label_dim=param.label_dim,
    ...        num_hops=len(param.head_num),
    ...    )
    ...    query_obj = GATQuery(p)
    ...
    ...    model = GAT(
    ...        head_num=param.head_num,
    ...        hidden_dim=param.hidden_dim,
    ...        num_classes=param.num_classes,
    ...        ffd_drop=param.ffd_drop,
    ...        attn_drop=param.attn_drop,
    ...        l2_coef=param.l2_coef,
    ...    )
    ...
    ...    return model, query_obj

.. code-block:: python

    >>> def define_param_gat(parser):
    ...    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    ...    parser.add_argument("--epochs", type=int, default=200, help="num of epochs for training")
    ...    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    ...
    ...    # GAT Model Parameters.
    ...    parser.add_argument("--head_num", type=str2list_int, default="8,1", help="the number of attention headers.")
    ...    parser.add_argument("--hidden_dim", type=int, default=8, help="hidden layer dimension.")
    ...    parser.add_argument("--num_classes", type=int, default=-1, help="number of classes for category")
    ...    parser.add_argument("--ffd_drop", type=float, default=0.0, help="feature dropout rate.")
    ...    parser.add_argument("--attn_drop", type=float, default=0.0, help="attention layer dropout rate.")
    ...    parser.add_argument("--l2_coef", type=float, default=0.0005, help="l2 loss")
    ...
    ...    ## training node types.
    ...    parser.add_argument("--node_types", type=str2list_int, default="0", help="Graph Node for training.")
    ...    ## evaluate node files.
    ...    parser.add_argument("--evaluate_node_files", type=str, help="evaluate node file list.")
    ...    ## inference node id
    ...    parser.add_argument("--inf_min_id", type=int, default=0, help="inferece min node id.")
    ...    parser.add_argument("--inf_max_id", type=int, default=-1, help="inference max node id.")
    ...
    ...    parser.add_argument(
    ...        "--distributed_strategy",
    ...        type=str,
    ...        default=None,
    ...        choices=[None, "Mirrored", "MultiWorkerMirrored"],
    ...        help="Distributed strategies to use.",
    ...    )
    ...    def register_gat_query_param(parser):
    ...            group = parser.add_argument_group("GAT Query Parameters")
    ...            group.add_argument("--neighbor_edge_types", type=str2list_int, default="0", help="Graph Edge for attention encoder.",)
    ...            group.add_argument("--feature_idx", type=int, default=0, help="feature index.")
    ...            group.add_argument("--feature_dim", type=int, default=16, help="feature dim.")
    ...            group.add_argument("--label_idx", type=int, default=1, help="label index.")
    ...            group.add_argument("--label_dim", type=int, default=1, help="label dim.")
    ...    register_gat_query_param(parser)

.. code-block:: python

    >>> def run_train(param, trainer, query, model, tf1_mode, backend):
    ...    tf_dataset, steps_per_epoch = create_tf_dataset(
    ...        sampler_class=GENodeSampler,
    ...        query_fn=query.query_training,
    ...        backend=backend,
    ...        node_types=np.array(param.node_types, dtype=np.int32),
    ...        batch_size=param.batch_size,
    ...        num_workers=trainer.worker_size,
    ...        worker_index=trainer.task_index,
    ...        strategy=SamplingStrategy.RandomWithoutReplacement,
    ...    )
    ...
    ...    distributed_dataset = get_distributed_dataset(
    ...        # NOTE: here we flatten all the epochs into 1 to increase performance.
    ...        lambda ctx: tf_dataset.repeat(param.epochs)
    ...    )
    ...
    ...    # we need to make sure the steps_per_epoch are provided in distributed dataset.
    ...    assert steps_per_epoch is not None or param.steps_per_epoch is not None
    ...    # Since we flatten the dataset to len(dataset) * param.epochs,
    ...    # we alos need to update steps_per_epoch.
    ...    steps_per_epoch = param.epochs * (steps_per_epoch or param.steps_per_epoch)
    ...
    ...    if tf1_mode:
    ...        opt = tf.compat.v1.train.AdamOptimizer(param.learning_rate * trainer.lr_scaler)
    ...    else:
    ...        opt = tf.keras.optimizers.Adam(
    ...            learning_rate=param.learning_rate * trainer.lr_scaler
    ...        )
    ...
    ...    trainer.train(
    ...        dataset=distributed_dataset,
    ...        model=model,
    ...        optimizer=opt,
    ...        epochs=1,
    ...        steps_per_epoch=steps_per_epoch,
    ...    )


.. code-block:: python

    >>> try:
    ...    define_param_base
    ... except NameError:
    ...    define_param_base = define_param_gat

.. code-block:: python

    >>> MODEL_DIR = f"tmp/gat_{np.random.randint(9999999)}"
    >>> arg_list = [
    ...    "--data_dir", "/tmp/cora",
    ...    "--mode", "train",
    ...    # "--trainer", "hvd",
    ...    "--seed", "123",
    ...    "--eager",
    ...    "--log_save_steps", "1",
    ...    "--backend", "snark",
    ...    "--graph_type", "local",
    ...    "--converter", "skip",
    ... #   "--sample_file", "/tmp/cora/train.nodes",
    ... #   "--node_type", "0",
    ...    "--neighbor_edge_types", "0",
    ...    "--feature_idx", "0",
    ...    "--feature_dim", "1433",
    ...    "--label_idx", "1",
    ...    "--label_dim", "1",
    ...    "--num_classes", "7",
    ...    "--batch_size", "140",
    ...    "--epochs", "20",
    ...    "--learning_rate", "0.005",
    ...    "--l2_coef", "0.0005",
    ...    "--attn_drop", "0.6",
    ...    "--ffd_drop", "0.6",
    ...    "--head_num", "8,1",
    ...    "--hidden_dim", "8",
    ...    "--model_dir", MODEL_DIR,
    ... #  "--metric_dir", MODEL_DIR,
    ... #  "--save_path", MODEL_DIR,
    ... ]

    >>> def define_param_wrap(define_param):
    ...    def define_param_new(parser):
    ...        define_param(parser)
    ...        parse_args = parser.parse_args
    ...        parser.parse_args = lambda: parse_args(arg_list)
    ...    return define_param_new
    >>> define_param_gat = define_param_wrap(define_param_base)

.. code-block:: python

    >>> def _main():
    ...    # setup default logging component.
    ...    setup_default_logging_config(enable_telemetry=True)
    ...
    ...    parser = argparse.ArgumentParser(
    ...        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    ...    )
    ...    common.args.import_default_parameters(parser)
    ...    define_param_gat(parser)
    ...
    ...    param = parser.parse_args()
    ...    common.args.log_all_parameters(param)
    ...
    ...    trainer = get_trainer(param)
    ...
    ...    backend = create_backend(BackendOptions(param), is_leader=(trainer.task_index == 0))
    ...
    ...    def run(tf1_mode=False):
    ...        model, query = build_model(param)
    ...        if param.mode == common.args.TrainMode.TRAIN:
    ...            run_train(param, trainer, query, model, tf1_mode, backend)
    ...        elif param.mode == common.args.TrainMode.EVALUATE:
    ...            run_eval(param, trainer, query, model, backend)
    ...        elif param.mode == common.args.TrainMode.INFERENCE:
    ...            run_inference(param, trainer, query, model, backend)
    ...
    ...    with closing(backend):
    ...        if param.eager:
    ...            strategy = None
    ...            if param.distributed_strategy == "Default":
    ...                strategy = tf.distribute.get_strategy()
    ...            elif param.distributed_strategy == "Mirrored":
    ...                strategy = tf.distribute.MirroredStrategy()
    ...            elif param.distributed_strategy == "MultiWorkerMirrored":
    ...                strategy = tf.distribute.MultiWorkerMirroredStrategy()
    ...
    ...            if strategy:
    ...                with strategy.scope():
    ...                    run()
    ...            else:
    ...                run()
    ...        else:
    ...            with tf.Graph().as_default():
    ...                trainer.set_random_seed(param.seed)
    ...                with trainer.tf_device():
    ...                    run(tf1_mode=True)


.. code-block:: python

    >>> _main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""AML."""
from typing import List, Any, Dict
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig, RunConfig

from deepgnn.pytorch.nn.gat_conv import GATConv
from deepgnn.graph_engine import graph_ops
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.graph_engine.snark.local import Client

from azureml.core import Workspace
from ray_on_aml.core import Ray_On_AML

from deepgnn.graph_engine import Graph
from deepgnn.pytorch.common import MeanAggregator, BaseMetric, MRR
from deepgnn.pytorch.modeling import BaseSupervisedModel, BaseUnsupervisedModel
from deepgnn.pytorch.encoding import FeatureEncoder, SageEncoder

from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import FileNodeSampler
from deepgnn.graph_engine import (
    CSVNodeSampler,
    GENodeSampler,
    SamplingStrategy,
    GraphEngineBackend,
)


@dataclass
class GATQuery:
    """GAT Query."""

    feature_meta: list = field(default_factory=lambda: np.array([[0, 1433]]))
    label_meta: list = field(default_factory=lambda: np.array([[1, 1]]))
    feature_type: np.dtype = np.float32
    label_type: np.dtype = np.float32

    def query(self, graph, inputs: np.ndarray) -> dict:
        """Fetch training data from graph."""
        if isinstance(inputs, (int, float)):
            inputs = [inputs]

        context = {"inputs": inputs}
        context["label"] = graph.node_features(
            context["inputs"],
            self.label_meta,
            np.int64,
        )
        context["encoder"] = self.enc.query(
            context["inputs"],
            graph,
            self.feature_type,
            self.feature_meta[0],
            self.feature_meta[1],
        )
        # self.transform(context)
        return {k: np.expand_dims(v, 0) for k, v in context.items()}


class SupervisedGraphSage(BaseSupervisedModel):
    """Simple supervised GraphSAGE model."""

    def __init__(
        self,
        num_classes: int,
        label_idx: int,
        label_dim: int,
        feature_type: np.dtype,
        feature_idx: int,
        feature_dim: int,
        edge_type: int,
        fanouts: list,
        embed_dim: int = 128,
        feature_enc=None,
    ):
        """Initialize a graphsage model for node classification."""
        super(SupervisedGraphSage, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

        # only 1 or 2 hops are allowed.
        assert len(fanouts) in [1, 2]
        self.fanouts = fanouts
        self.edge_type = edge_type

        def feature_func(features):
            return features.squeeze(0)

        first_layer_enc = SageEncoder(
            features=feature_func,
            query_func=None,
            feature_dim=self.feature_dim,
            intermediate_dim=self.feature_enc.embed_dim
            if self.feature_enc
            else self.feature_dim,
            aggregator=MeanAggregator(feature_func),
            embed_dim=embed_dim,
            edge_type=self.edge_type,
            num_sample=self.fanouts[0],
        )

        self.enc = (
            SageEncoder(
                features=lambda context: first_layer_enc(context),
                query_func=first_layer_enc.query,
                feature_dim=self.feature_dim,
                intermediate_dim=embed_dim,
                aggregator=MeanAggregator(lambda context: first_layer_enc(context)),
                embed_dim=embed_dim,
                edge_type=self.edge_type,
                num_sample=self.fanouts[1],
                base_model=first_layer_enc,
            )
            if len(self.fanouts) == 2
            else first_layer_enc
        )

        self.label_idx = label_idx
        self.label_dim = label_dim
        self.weight = nn.Parameter(
            torch.empty(embed_dim, num_classes, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.weight)

    def get_score(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate scores for a list of nodes."""
        self.encode_feature(context)
        embeds = self.enc(context["encoder"])
        scores = torch.matmul(embeds, self.weight)

        return scores

    def metric_name(self):
        """Metric used for model evaluation."""
        return self.metric.name()

    def get_embedding(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Generate embedding."""
        return self.enc(context["encoder"])

    def query(self, graph, inputs: np.ndarray) -> dict:
        """Fetch training data from graph."""
        # if isinstance(inputs, (int, float)):
        #    inputs = [inputs]

        context = {"inputs": inputs}
        context["label"] = graph.node_features(
            context["inputs"],
            np.array([[self.label_idx, self.label_dim]]),
            np.int64,
        ).reshape((-1, self.label_dim))
        context["encoder"] = self.enc.query(
            context["inputs"],
            graph,
            self.feature_type,
            self.feature_idx,
            self.feature_dim,
        )
        # self.transform(context)
        # assert False, context
        return context  # {k: np.expand_dims(v, 0) for k, v in context.items()}

    def _loss_inner(self, context):
        """Cross entropy loss for a list of nodes."""
        if isinstance(context, dict):
            labels = context["label"].squeeze()  # type: ignore
        elif isinstance(context, torch.Tensor):
            labels = context.squeeze()  # type: ignore
        else:
            raise TypeError("Invalid input type.")
        device = labels.device

        # TODO(chaoyl): Due to the bug of pytorch argmax, we have to copy labels to numpy for argmax
        # then copy back to Tensor. The fix has been merged to pytorch master branch but not included
        # in latest stable version. Revisit this part after updating pytorch with the fix included.
        # issue: https://github.com/pytorch/pytorch/issues/32343
        # fix: https://github.com/pytorch/pytorch/pull/37864
        labels = labels.cpu().numpy().argmax(1)
        scores = self.get_score(context)
        return (
            # self.xent(
            #    scores,
            #    Variable(torch.tensor(labels.squeeze(), dtype=torch.int64).to(device)),
            # ),
            scores,
            scores.argmax(dim=1),
            torch.tensor(labels.squeeze(), dtype=torch.int64),
        )

    def forward(self, context):
        """Return cross entropy loss."""
        return self._loss_inner(context)


def train_func(config: Dict):
    """Train function main."""
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    model = SupervisedGraphSage(
        num_classes=50,
        label_idx=0,
        label_dim=50,
        feature_type=np.float32,
        feature_idx=1,
        feature_dim=300,
        edge_type=0,
        fanouts=[25, 25],
    )

    model_original = model
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    optimizer = train.torch.prepare_optimizer(optimizer)

    loss_fn = nn.CrossEntropyLoss()

    """
    dataset = ray.data.range(2708, parallelism=1)
    pipe = dataset.window(blocks_per_window=10).repeat(10)
    g = Client("/tmp/cora", [0], delayed_start=True)
    q = GATQuery()
    def transform_batch(batch: list) -> dict:
        return model.query(g, batch)
    pipe = pipe.map_batches(transform_batch)
    """

    SAMPLE_NUM = 152410
    BATCH_SIZE = 512

    g = Client("/tmp/reddit", [0])
    dataset = TorchDeepGNNDataset(
        sampler_class=GENodeSampler,
        backend=type("Backend", (object,), {"graph": g})(),  # type: ignore
        sample_num=SAMPLE_NUM,
        num_workers=2,
        worker_index=0,
        node_types=np.array([0], dtype=np.int32),
        batch_size=BATCH_SIZE,
        query_fn=model_original.query,
        strategy=SamplingStrategy.RandomWithoutReplacement,
        prefetch_queue_size=10,
        prefetch_worker_size=2,

    )
    dataset = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=2,  # data_parralel_num = 2
    )

    model.train()
    for epoch in range(10):  # , epoch_pipe in enumerate(pipe.iter_epochs()):
        metrics = []
        losses = []
        for i, batch in enumerate(dataset):  # enumerate(
            # epoch_pipe.random_shuffle_each_window().iter_torch_batches(batch_size=2708)
            # ):
            scores = model(batch)[0]
            labels = batch["label"]

            loss = loss_fn(scores.type(torch.float32), labels.squeeze().float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics.append((scores.squeeze().argmax(1) == labels.squeeze().argmax(1)).float().mean())
            losses.append(loss.item())

            if i >= SAMPLE_NUM / BATCH_SIZE / session.get_world_size():
                break

        print("RESULTS:!", np.mean(metrics), np.mean(losses))

        session.report(
            {
                "metric": np.mean(metrics),
                "loss": np.mean(losses),
            }
        )

           
#ws = Workspace.from_config("config.json")


#ray_on_aml = Ray_On_AML(ws=ws, compute_cluster="multi-node", maxnode=2)
#ray = ray_on_aml.getRay()

import ray
ray.init()

trainer = TorchTrainer(
    train_func,
    train_loop_config={},
    run_config=RunConfig(),
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False, trainer_resources={"CPU": 2}, resources_per_worker={"CPU": 2}),
)
result = trainer.fit()

# ray_on_aml.shutdown()

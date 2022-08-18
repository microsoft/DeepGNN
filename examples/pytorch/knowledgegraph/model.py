# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Collection of graphsage based models."""
import configparser
import enum
import numpy as np
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn

from deepgnn.graph_engine import Graph, FeatureType
from deepgnn.pytorch.modeling.base_model import BaseModel, BaseMetric
from deepgnn.pytorch.common.metrics import MRR

EMB_INIT_EPS = 2.0


class Mode(enum.Enum):
    """Mode to control training accross batches."""

    Head = 0
    Tail = 1
    Single = 2


class KGEModel(BaseModel):
    """Knowledge graph embedding model."""

    def __init__(
        self,
        model_args: dict,
        num_negs: int = 20,
        embed_dim: int = 1000,
        metric: BaseMetric = MRR(),
        gpu: bool = False,
    ):
        """Initialize model."""
        super(KGEModel, self).__init__(
            feature_dim=0,
            feature_idx=0,
            feature_type=FeatureType.INT64,
            feature_enc=None,
        )
        self.model_args = model_args
        self.gpu = gpu
        self.num_negs = num_negs
        self.metric = metric
        self.embed_dim = embed_dim
        self.epsilon = EMB_INIT_EPS
        self.model_name = self.model_args["score_func"]

        config = configparser.ConfigParser()
        config.read(self.model_args["metadata_path"])
        self.nentities = int(config["DEFAULT"]["num_entities"])
        self.nrelations = int(config["DEFAULT"]["num_relations"])

        self.adversarial_temperature = (
            0.0
            if "adversarial_temperature" not in self.model_args
            else float(self.model_args["adversarial_temperature"])
        )
        self.negative_adversarial_sampling = (
            True if self.adversarial_temperature > 0.0 else False
        )
        self.regularization = (
            0.0
            if "regularization" not in self.model_args
            else float(self.model_args["regularization"])
        )
        self.uni_weight = (
            0
            if "uni_weight" not in self.model_args
            else int(self.model_args["uni_weight"])
        )
        self.gamma = nn.Parameter(
            torch.tensor([self.model_args["gamma"]]), requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.tensor([(self.gamma.item() + self.epsilon) / self.embed_dim]),
            requires_grad=False,
        )

        double_entity_embedding = (
            False
            if "double_entity_embedding" not in self.model_args
            else bool(self.model_args["double_entity_embedding"])
        )
        double_relation_embedding = (
            False
            if "double_relation_embedding" not in self.model_args
            else bool(self.model_args["double_relation_embedding"])
        )
        self.entity_dim = (
            self.embed_dim * 2 if double_entity_embedding else self.embed_dim
        )
        self.relation_dim = (
            self.embed_dim * 2 if double_relation_embedding else self.embed_dim
        )

        self.entity_embedding = nn.Parameter(
            torch.zeros(self.nentities, self.entity_dim)
        )
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(self.nrelations, self.relation_dim)
        )
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )
        self.step = 0

    def query(self, graph: Graph, inputs: np.ndarray) -> dict:
        """Fetch data from graph for training."""
        context = {"inputs": inputs}

        features = graph.edge_features(
            inputs, np.array([[0, 2]], dtype=np.int32), FeatureType.INT64
        )
        # features[:,0] is relation type for embedding index.
        inputs[:, 2] = features[:, 0].astype("int64")
        # features[:,1] is subsampling weight(number of tail node + number of head node.).
        context["weights"] = features[:, 1].astype("float")

        self.step += 1
        if self.step % 2 == 0:
            context["mode"] = Mode.Head.value
        else:
            context["mode"] = Mode.Tail.value

        if context["mode"] == 0:
            node_ids = inputs[:, 1]
            edge_type = 1
        else:
            node_ids = inputs[:, 0]
            edge_type = 0

        neighbor_nodes, _, _, neighbor_counts = graph.neighbors(
            node_ids, np.array([edge_type], dtype=np.int32)
        )

        cur_pos = 0
        all_negative_list = []
        for count in neighbor_counts:
            negative_sample_size = 0
            negative_sample_list = []
            while negative_sample_size < self.num_negs:
                negative_sample = np.random.randint(
                    self.nentities, size=self.num_negs * 2
                )
                mask = np.in1d(
                    negative_sample,
                    neighbor_nodes[cur_pos : int(cur_pos + count)],
                    assume_unique=True,
                    invert=True,
                )
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[: self.num_negs]
            all_negative_list.append(negative_sample)
            cur_pos += int(count)

        context["negs"] = np.array(all_negative_list)
        return context

    def query_eval(self, graph: Graph, inputs: np.ndarray) -> dict:
        """Fetch data from graph for evaluation."""
        context = {}
        inputs[0][:, 2] = inputs[1][:, 0].astype("int64")
        context["inputs"] = inputs[0]
        context["mode"] = inputs[1][:, 1]

        inputs = context["inputs"]
        mode = context["mode"]

        all_negative_list = []
        all_filter_bias = []
        for triple, m in zip(inputs, mode):
            if m == 0:
                pos_id = triple[1]
                edge_type = 1
            else:
                pos_id = triple[0]
                edge_type = 0

            neighbor_nodes, _, _, _ = graph.neighbors(
                np.array([pos_id], dtype=np.int64),
                np.array([edge_type], dtype=np.int32),
            )

            if m == 0:
                tmp = [
                    (0, rand_head)
                    if rand_head not in neighbor_nodes
                    else (-1, triple[0])
                    for rand_head in range(self.nentities)
                ]
                tmp[triple[0]] = (0, triple[0])
            else:
                print(m)
                tmp = [
                    (0, rand_tail)
                    if rand_tail not in neighbor_nodes
                    else (-1, triple[1])
                    for rand_tail in range(self.nentities)
                ]
                tmp[triple[1]] = (0, triple[1])
            tmp_arr = np.array(tmp)
            filter_bias = tmp_arr[:, 0].astype("float")
            all_filter_bias.append(filter_bias)
            negative_sample = tmp_arr[:, 1]
            all_negative_list.append(negative_sample)

        context["bias"] = np.array(all_filter_bias)
        context["negs"] = np.array(all_negative_list)
        return context

    def get_score(  # type: ignore
        self, sample: Union[torch.Tensor, Tuple[torch.Tensor]], mode=Mode.Single
    ) -> torch.Tensor:
        """Calculate score according to model_name."""
        # 0: head-batch; 1: tail-batch; 2:single
        if mode == Mode.Single:
            batch_size, negative_sample_size = sample.size(0), 1  # type: ignore
            sample = sample.squeeze(0)  # type: ignore
            head = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 0]
            ).unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding, dim=0, index=sample[:, 2]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 1]
            ).unsqueeze(1)
        elif mode == Mode.Head:
            tail_part, head_part = sample  # type: ignore
            head_part = head_part.squeeze(0)
            tail_part = tail_part.squeeze(0)
            relation = torch.index_select(
                self.relation_embedding, dim=0, index=tail_part[:, 2]
            )

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part[:, 1]
            )

            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            tail = tail.unsqueeze(1)
            relation = relation.unsqueeze(1)
            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == Mode.Tail:
            head_part, tail_part = sample  # type: ignore
            head_part = head_part.squeeze(0)
            tail_part = tail_part.squeeze(0)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part[:, 0]
            )

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=head_part[:, 2]
            )

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = head.unsqueeze(1)
            relation = relation.unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError("mode %s not supported" % mode)

        model_func = {
            "TransE": self.transE,
            "DistMult": self.distMult,
            "ComplEx": self.complEx,
            "RotatE": self.rotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError("model %s not supported" % self.model_name)

        return score

    def transE(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, mode: Mode
    ):
        """Calculate score for TransE model."""
        if mode == Mode.Head:
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def distMult(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, mode: Mode
    ):
        """Calculate score for DistMult model."""
        if mode == Mode.Head:
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def complEx(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, mode: Mode
    ):
        """Calculate score for ComplEx model."""
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == Mode.Head:
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def rotatE(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, mode: Mode
    ):
        """Calculate score for RotatE model."""
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / np.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == Mode.Head:
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def forward(self, context: dict):  # type: ignore[override]
        """Calculate loss."""
        if "bias" in context.keys():
            return self.loss_eval(context)

        positive_sample = context["inputs"]
        negative_sample = context["negs"]
        subsampling_weight = torch.sqrt(1 / context["weights"]).squeeze(0)

        if self.gpu:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = self.get_score(
            (positive_sample, negative_sample),  # type: ignore
            mode=Mode(int(context["mode"])),
        )

        negative_score = negative_score.reshape(-1, self.num_negs)

        if self.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (
                fn.softmax(
                    negative_score * self.adversarial_temperature, dim=1
                ).detach()
                * fn.logsigmoid(-negative_score)
            ).sum(dim=1)
        else:
            negative_score = fn.logsigmoid(-negative_score).mean(dim=1)

        positive_score = self.get_score(positive_sample)

        positive_score = fn.logsigmoid(positive_score).squeeze(dim=1)

        if self.uni_weight:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = (
                -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            )
            negative_sample_loss = (
                -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
            )

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if self.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = self.regularization * (
                self.entity_embedding.norm(p=3) ** 3
                + self.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization

        return loss, 0.0, 0.0

    def loss_eval(self, context: dict):
        """Calculate loss with bias."""
        positive_sample = context["inputs"]
        negative_sample = context["negs"]
        filter_bias = context["bias"]

        if self.gpu:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            filter_bias = filter_bias.cuda()
        mode = Mode(int(context["mode"].squeeze(0)[0]))
        score = self.get_score(
            (positive_sample, negative_sample), mode=mode  # type: ignore
        )
        score += filter_bias.squeeze(0)

        argsort = torch.argsort(score, dim=1, descending=True)

        positive_sample = positive_sample.squeeze(0)
        if mode == Mode.Head:
            positive_arg = positive_sample[:, 0]
        else:
            positive_arg = positive_sample[:, 1]

        batch_size = positive_sample.size(0)

        mmr_score = 0
        for i in range(batch_size):
            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
            ranking = 1 + ranking.item()
            mmr_score += 1.0 / ranking

        mmr_score = mmr_score / batch_size
        return torch.zeros(1), mmr_score

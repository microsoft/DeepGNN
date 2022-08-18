# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Link prediction model implementation."""
import numpy as np
import os
import torch
import torch.nn as nn

from typing import Optional, Tuple
from deepgnn import TrainMode, vec2str
from deepgnn.graph_engine import (
    Graph,
    FeatureType,
    INVALID_NODE_ID,
    multihop,
)

from deepgnn.pytorch.common.consts import (
    NODE_SRC,
    NODE_DST,
    NODE_FEATURES,
    ENCODER_SEQ,
    ENCODER_MASK,
    ENCODER_TYPES,
    FANOUTS,
    INPUTS,
)
from deepgnn.pytorch.common.metrics import BaseMetric, ROC
from deepgnn.pytorch.modeling.base_model import BaseSupervisedModel
from deepgnn.pytorch.encoding import FeatureEncoder, MultiTypeFeatureEncoder

from encoder import GnnEncoder  # type: ignore
from output_layer import OutputLayer  # type: ignore
from multi_task import MultiTaskAggregator  # type: ignore
from consts import SIM_TYPE_COSINE_WITH_RNS, ENCODER_LABEL, FANOUTS_NAME  # type: ignore


class LinkPredictionModel(BaseSupervisedModel):
    """Supervised link prediction model."""

    def __init__(
        self,
        args,
        feature_dim=0,
        feature_idx=0,
        feature_type=None,
        feature_enc: Optional[FeatureEncoder] = None,
        metric: BaseMetric = ROC(),
        vocab_index: int = 0,
    ):
        """Initialize model."""
        assert feature_enc is not None and isinstance(
            feature_enc, MultiTypeFeatureEncoder
        )

        super(LinkPredictionModel, self).__init__(
            feature_type=feature_type,
            feature_idx=feature_idx,
            feature_dim=feature_dim,
            feature_enc=feature_enc,
        )

        self.args = args
        self.src_metapath = self.args.src_metapath
        self.src_fanouts = self.args.src_fanouts
        self.dst_metapath = self.args.dst_metapath
        self.dst_fanouts = self.args.dst_fanouts
        self.default_node = INVALID_NODE_ID
        # only mask neighbors in train mode.
        self.neighbor_mask = (
            self.args.neighbor_mask and self.args.mode == TrainMode.TRAIN
        )
        self.metric = metric
        self.vocab_index = vocab_index

        self.gnn_encoder = GnnEncoder(
            input_dim=feature_enc.embed_dim,
            fanouts_dict={
                NODE_SRC: self.args.src_fanouts,
                NODE_DST: self.args.dst_fanouts,
            },
            act_functions=args.gnn_acts,
            encoder_name=args.gnn_encoder,
            head_nums=args.gnn_head_nums,
            hidden_dims=args.gnn_hidden_dims,
            residual=args.gnn_residual,
            lgcl_largest_k=args.lgcl_largest_k,
        )
        self.output_layer = OutputLayer(
            input_dim=args.gnn_hidden_dims[-1],
            sim_type=self.args.sim_type,
            res_size=self.args.res_size,
            res_bn=self.args.res_bn,
            featenc_config=os.path.join(self.args.meta_dir, self.args.featenc_config),
            random_negative=self.args.num_negs,
            nsp_gamma=self.args.nsp_gamma,
        )
        if self.args.multi_task_config != "":
            config = MultiTaskAggregator.init_config_from_file(
                os.path.join(self.args.meta_dir, self.args.multi_task_config)
            )
            self.multi_task_aggregator = MultiTaskAggregator(config)

    def get_score(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Calculate score."""
        self.encode_feature(context)
        return self.gnn_encoder(context)

    def forward(self, context: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Calculate scores and return them with loss and labels."""
        batch = context[NODE_FEATURES]
        # source nodes
        src_info = self.get_score(
            {
                ENCODER_SEQ: batch[1][0],
                ENCODER_MASK: batch[1][1],
                ENCODER_LABEL: batch[3],
                ENCODER_TYPES: self.args.src_encoders,
                FANOUTS: self.args.src_fanouts,
                FANOUTS_NAME: NODE_SRC,
            }
        )
        # destination nodes
        dst_info = self.get_score(
            {
                ENCODER_SEQ: batch[2][0],
                ENCODER_MASK: batch[2][1],
                ENCODER_LABEL: None,
                ENCODER_TYPES: self.args.dst_encoders,
                FANOUTS: self.args.dst_fanouts,
                FANOUTS_NAME: NODE_DST,
            }
        )

        loss, scores = self._calculate_loss(
            src_info[0].float(),
            src_info[1],
            src_info[2],
            dst_info[0].float(),
            dst_info[1],
            dst_info[2],
            batch[3],
            sample_weight=None,
        )
        loss = loss.mean()

        if self.args.multi_task_config != "":
            multi_task_loss = self.multi_task_aggregator.calculate_loss(
                src_info, dst_info, batch[4]
            )
            loss += torch.sum(multi_task_loss)

        return loss, scores, batch[3].squeeze(0)

    def _calculate_loss(
        self,
        src_vec,
        src_terms,
        src_mask,
        dst_vec,
        dst_terms,
        dst_mask,
        inf_label,
        sample_weight=None,
        prefix="",
    ):
        with_rng = self.args.num_negs > 0
        simscore = self.output_layer.simpooler(
            src_vec,
            src_terms,
            src_mask,
            dst_vec,
            dst_terms,
            dst_mask,
            with_rng=with_rng,
            prefix=prefix,
        )

        if self.args.label_map is not None:
            ori_label, target_label = self.args.label_map.split(":")
            ori_label = [float(i) for i in ori_label.split(",")]
            target_label = [float(i) for i in target_label.split(",")]
            inf_label = (inf_label - ori_label[0]) / (
                (ori_label[1] - ori_label[0]) / (target_label[1] - target_label[0])
            ) + target_label[0]
            inf_label = torch.max(inf_label, torch.zeros_like(inf_label))
            inf_label = torch.min(inf_label, torch.ones_like(inf_label))

        if with_rng:
            inf_label = torch.cat(
                [inf_label] + [torch.zeros_like(inf_label)] * self.args.num_negs, dim=0
            )
            if sample_weight is not None:
                sample_weight = torch.cat(
                    [sample_weight]
                    + [torch.ones_like(sample_weight)] * self.args.num_negs,
                    dim=0,
                )

        if self.args.use_mse:
            inf_label = self._inverse_sigmoid(inf_label.view(-1, 1))
            loss = nn.MSELoss(reduce=False)(simscore.view(-1, 1), inf_label)
        elif self.args.sim_type == SIM_TYPE_COSINE_WITH_RNS:
            loss = -(simscore.view(-1, 1))
        else:
            loss = nn.BCEWithLogitsLoss(reduce=False)(
                simscore.view(-1, 1), inf_label.view(-1, 1)
            )
        if sample_weight is not None:
            loss = loss * sample_weight

        loss = torch.mean(loss)
        return loss, simscore

    def _inverse_sigmoid(self, x):
        return torch.log(x / (-x + 1))

    def metric_name(self):
        """Metric used for evaluation."""
        return self.metric.name()

    def query(self, graph: Graph, inputs: np.ndarray) -> dict:
        """
        Query graph for training data.

        Current format: row_id, source, seqid + seq_mask, destination, seqid + seq_mask, label.
        Input data format:
            |row_id | src_id | src_seq_mask | dst_id | dst_seq_mask | label |
            | ----- | ------ | ------------ | ------ | ------------ | ----- |
            | 0 | 0 | 1266,9292,2298,0,0,0,1,1,1,0,0,0 | 27929 | 11166,2298,1042,0,0,0,1,1,1,0,0,0 | 0 |
            | 1 | 2 | 68,6792,198,0,0,0,1,1,1,0,0,0    | 102   | 123,6387,7135,0,0,0,1,1,1,0,0,0   | 1 |

            src_id:       source node id.
            src_seq_mask: concatenation of source seq and source mask, separated by ','.
            dst_id:       destination node id.
            dst_seq_mask: concatenation of destination seq and source mask, separated by ','.
            label:        if or not the source and destination node has link.

            The length of sequence and mask is the same which means we can divide it by 2 to get its
            seq: 'data[:feature_dim // 2]' and its mask: 'data[feature_dim // 2:]'
        """
        context = {INPUTS: inputs}
        batch = context[INPUTS]
        batch_size = 0

        src_ids = []
        dst_ids = []
        labels = []
        multi_task_labels = []
        row_ids = []
        src_feats = []
        dst_feats = []

        default_feature = np.array(
            [self.vocab_index]
            + [0] * (self.feature_dim // 2 - 1)
            + [1]
            + [0] * (self.feature_dim // 2 - 1)
        ).astype(np.int64)

        for line in batch:
            tokens = line.split("\t")
            if len(tokens) > 5:
                row_ids.append(tokens[0])
                src_ids.append(int(tokens[1]))
                dst_ids.append(int(tokens[3]))
                src_feats += [int(x) for x in tokens[2].split(",")][: self.feature_dim]
                dst_feats += [int(x) for x in tokens[4].split(",")][: self.feature_dim]
                labels.append(float(tokens[5]))
                multi_task_labels.append(tokens[6:])
                batch_size += 1

        src_ids = np.array(src_ids)
        dst_ids = np.array(dst_ids)

        def _get_multihop_features(node_ids, features, metapaths, fanouts, forbid=None):
            features = np.reshape(np.array(features), [batch_size, self.feature_dim])
            multi_type_seqs = [[features[:, : self.feature_dim // 2]]]
            multi_type_mask = [[features[:, self.feature_dim // 2 :]]]

            for type, fanout in enumerate(fanouts):
                metapath = [[t] for t in metapaths[type]]
                multihop_ids, _, _ = multihop.sample_fanout(
                    graph, node_ids, metapath, fanout, self.default_node
                )
                type_seqs = []
                type_mask = []
                for layer, ids in enumerate(multihop_ids[1:]):
                    one_hop_feats = graph.node_features(
                        ids, np.array([[0, self.feature_dim]]), FeatureType.INT64
                    ).astype(np.int64)
                    for idx, node_id in enumerate(ids):
                        if (
                            len(
                                torch.nonzero(
                                    torch.as_tensor(one_hop_feats[idx]), as_tuple=True
                                )[0]
                            )
                            == 0
                            or node_id == self.default_node
                            or forbid is not None
                            and layer == 0
                            and node_id == forbid[idx // fanout[0]]
                        ):
                            one_hop_feats[idx] = default_feature
                    one_hop_feats = np.reshape(one_hop_feats, [-1, self.feature_dim])
                    type_seqs.append(one_hop_feats[:, : self.feature_dim // 2])
                    type_mask.append(one_hop_feats[:, self.feature_dim // 2 :])
                multi_type_seqs.append(type_seqs)
                multi_type_mask.append(type_mask)

            return multi_type_seqs, multi_type_mask

        src_multihop_feats = _get_multihop_features(
            src_ids,
            src_feats,
            self.src_metapath,
            self.src_fanouts,
            dst_ids if self.neighbor_mask else None,
        )
        dst_multihop_feats = _get_multihop_features(
            dst_ids,
            dst_feats,
            self.dst_metapath,
            self.dst_fanouts,
            src_ids if self.neighbor_mask else None,
        )

        labels = np.array(labels)
        multi_task_labels = np.array(multi_task_labels)

        context[NODE_FEATURES] = [
            row_ids,
            src_multihop_feats,
            dst_multihop_feats,
            labels,
            multi_task_labels,
        ]
        del context[INPUTS]
        self.transform(context)
        return context

    def get_embedding(self, context: dict) -> torch.Tensor:  # type: ignore[override]
        """Compute embeddings, scores for src and destination nodes."""
        batch = context[NODE_FEATURES]
        # source nodes
        src_info = self.get_score(
            {
                ENCODER_SEQ: batch[1][0],
                ENCODER_MASK: batch[1][1],
                ENCODER_LABEL: batch[3],
                ENCODER_TYPES: self.args.src_encoders,
                FANOUTS: self.args.src_fanouts,
                FANOUTS_NAME: NODE_SRC,
            }
        )
        # destination nodes
        dst_info = self.get_score(
            {
                ENCODER_SEQ: batch[2][0],
                ENCODER_MASK: batch[2][1],
                ENCODER_LABEL: None,
                ENCODER_TYPES: self.args.dst_encoders,
                FANOUTS: self.args.dst_fanouts,
                FANOUTS_NAME: NODE_DST,
            }
        )

        scores = self.output_layer.simpooler(
            src_info[0].float(),
            src_info[1],
            src_info[2],
            dst_info[0].float(),
            dst_info[1],
            dst_info[2],
        )

        return scores, src_info[0], dst_info[0], batch[0]

    def output_embedding(self, output, context: dict, embeddings):  # type: ignore[override]
        """Print embeddings."""
        scores, src_batch, dst_batch, row_id = embeddings
        scores = torch.sigmoid(scores).cpu().detach().numpy()
        src = src_batch.cpu().detach().numpy()
        dst = dst_batch.cpu().detach().numpy()
        for i in range(src.shape[0]):
            record = [
                str(row_id[i][0]),
                vec2str(src[i]),
                vec2str(dst[i]),
                str(scores[i]),
            ]
            output.write("\t".join(record) + "\n")

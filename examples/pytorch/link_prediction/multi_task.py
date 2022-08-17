# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Mutli-task layer implememntation."""
import json
import torch
import torch.nn as nn
from output_layer import OutputLayer  # type: ignore
from consts import SIM_TYPE_COSINE  # type: ignore


class MultiTaskAggregator(nn.Module):
    """Multi-task Layer, aggregate loss from separate tasks."""

    def __init__(self, config):
        """
        Initialize layer.

        Args:
            config: config of tasks.
        """
        super(MultiTaskAggregator, self).__init__()
        self.config = config
        for task_name in self.config.keys():
            getattr(self, f"_init_{task_name}_task")(self.config[task_name])

    def calculate_loss(self, src_info, dst_info, multi_task_labels):
        """Calculate multi-task loss."""
        src_vec = src_info[0]
        src_terms = src_info[1]
        src_mask = src_info[2]
        dst_vec = dst_info[0]
        dst_terms = dst_info[1]
        dst_mask = dst_info[2]

        loss_list = []
        multi_task_labels.squeeze(0).transpose(1, 0)
        for task_name, label in zip(self.config.keys(), multi_task_labels):
            task_loss = getattr(self, f"_calculate_{task_name}_loss")(
                src_vec,
                src_terms,
                src_mask,
                dst_vec,
                dst_terms,
                dst_mask,
                label,
                self.config[task_name],
            )
            loss_list.append(task_loss.unsqueeze(0) * self.config[task_name]["weight"])

        loss_list = torch.cat(loss_list)
        return loss_list

    def _init_relevance_task(self, task_config):
        self.rel_output_layer = OutputLayer(
            input_dim=task_config["input_dim"],
            sim_type=SIM_TYPE_COSINE,
            random_negative=task_config["rel_num_negs"],
        )
        self.rel_output_layer.apply(self._constant_initialization)

    def _calculate_relevance_loss(
        self,
        src_vec,
        src_terms,
        src_mask,
        dst_vec,
        dst_terms,
        dst_mask,
        rel_label,
        task_config,
        prefix="",
    ):
        with_rng = task_config["rel_num_negs"] > 0
        simscore = self.rel_output_layer.simpooler(
            src_vec,
            src_terms,
            src_mask,
            dst_vec,
            dst_terms,
            dst_mask,
            with_rng=with_rng,
            prefix=prefix,
        )

        if with_rng:
            rel_label = torch.cat(
                [rel_label]
                + [torch.zeros_like(rel_label)] * task_config["rel_num_negs"],
                dim=0,
            )

            loss = nn.BCEWithLogitsLoss(reduce=False)(
                simscore.view(-1, 1), rel_label.view(-1, 1)
            )
        else:
            rel_label = torch.cat([rel_label], dim=0)

            pos_weight = torch.ones([1]).to(simscore.device)
            loss = nn.BCEWithLogitsLoss(reduce=False, pos_weight=pos_weight)(
                simscore.view(-1, 1), rel_label.view(-1, 1)
            )

        loss = torch.mean(loss)
        return loss

    def _constant_initialization(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 1.0)

    @classmethod
    def init_config_from_file(cls, filename):
        """
        Initialize aggregator from a JSON file.

        Multi-task config file format:
        {
            "relevance": { # task name
                "weight": 1.0, # task parameters
                "rel_num_negs": 32,
                "input_dim": 64
            }
        }
        """
        config = {}
        with open(filename, "r", encoding="utf-8") as fin:
            config = json.load(fin)
        return config

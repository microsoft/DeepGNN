# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import torch
import torch.distributed as dist

from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from typing import Any, Optional
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from deepgnn.pytorch.training.trainer_fp16 import FP16Trainer, BaseModel
from deepgnn.pytorch.training.utils import disable_infini_band


class DDPTrainer(FP16Trainer):
    """Distributed Data Parallel(DDP) based trainer"""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._init_process_group()

    def __del__(self):
        dist.destroy_process_group()

    def _evaluate(self, model: Module):
        metric, loss = super()._evaluate(model)
        metric = self._allreduce(metric)
        loss = self._allreduce(loss)
        self.logger.info(
            self._wrap_log(
                f"AllReduced {self.model.metric_name()}: {metric:.4f}; loss: {loss:.4f};"
            )
        )
        return metric, loss

    def _init_process_group(self):
        if self.args.disable_ib:
            disable_infini_band()
        # torch.distributed.launch will set below env variables.
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        self.logger.info(f"Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl" if self.args.gpu else "gloo")
        self.rank = dist.get_rank()
        self.local_rank = self.args.local_rank
        self.world_size = dist.get_world_size()

        self.logger.info(
            f"Initialized ddp trainer. rank:{self.rank}, local_rank:{self.local_rank},"
            f" world_size:{self.world_size}"
        )

    def _init_model(self, model: BaseModel):
        model = super()._init_model(model)
        self._broadcast_model_state(model)
        return model

    def _initialize(
        self,
        model: BaseModel,
        dataset: Any,
        optimizer: Optional[Optimizer] = None,
        eval_dataset_for_training: Any = None,
    ):
        model = super()._initialize(
            model, dataset, optimizer, eval_dataset_for_training
        )
        return self._wrap_ddp(model)

    def _broadcast_model_state(self, model: Module):
        vector = parameters_to_vector(model.parameters())
        dist.broadcast(vector, 0)
        if self.rank != 0:
            vector_to_parameters(vector, model.parameters())
        del vector

    def _wrap_ddp(self, model: BaseModel) -> DistributedDataParallel:
        return DistributedDataParallel(  # type: ignore
            model,
            device_ids=[self.local_rank] if self.args.gpu else None,
            output_device=self.local_rank if self.args.gpu else None,
            find_unused_parameters=True,
        )

    def _allreduce(self, metric: torch.Tensor):
        if self.args.gpu:
            metric = metric.cuda()
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        return metric / self.world_size

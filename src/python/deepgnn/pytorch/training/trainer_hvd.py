# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch
from torch.nn import Module
from torch.optim import Optimizer
from deepgnn.pytorch.training.trainer_fp16 import FP16Trainer, BaseModel
from deepgnn.pytorch.training.utils import disable_infini_band
import horovod.torch as hvd  # type: ignore


class HVDTrainer(FP16Trainer):
    """Horovod based distributed trainer."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._init_hvd()

    def _evaluate(self, model: Module):
        metric, loss = super()._evaluate(model)
        metric = hvd.allreduce(metric)
        loss = hvd.allreduce(loss)
        self.logger.info(
            self._wrap_log(
                f"AllReduced {self.model.metric_name()}: {metric:.4f}; loss: {loss:.4f}"
            )
        )
        return metric, loss

    def _init_hvd(self):
        """Initialize Horovod."""
        if self.args.disable_ib:
            disable_infini_band()
        hvd.init()
        self.rank = hvd.rank()
        self.local_rank = hvd.local_rank()
        self.world_size = hvd.size()
        self.logger.info(
            f"Initialized horovod trainer. rank:{self.rank}, local_rank:{self.local_rank},"
            f" world_size:{self.world_size}"
        )

    def _init_model(self, model: BaseModel):
        model = super()._init_model(model)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        return model

    def _init_optimizer(self, optimizer: Optimizer):
        optimizer = super()._init_optimizer(optimizer)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = (
            hvd.Compression.fp16 if self.fp16_enabled() else hvd.Compression.none
        )
        return hvd.DistributedOptimizer(
            optimizer=optimizer,
            named_parameters=self.model.named_parameters(),
            compression=compression,
            op=hvd.Average,
        )

    def _train_one_epoch(self, model: Module, epoch: int):
        super()._train_one_epoch(model, epoch)
        hvd.join()

    def _inference(self, model: Module):
        super()._inference(model)
        hvd.join()

    def _apex_backward(self, scaled_loss: torch.Tensor):
        scaled_loss.backward()
        self.optimizer.synchronize()

    def _apex_step(self):
        with self.optimizer.skip_synchronize():
            self.optimizer.step()

    def _amp_backward(self, loss):
        self.grad_scaler.scale(loss).backward()
        self.optimizer.synchronize()

    def _amp_step(self):
        with self.optimizer.skip_synchronize():
            self.grad_scaler.step(self.optimizer)

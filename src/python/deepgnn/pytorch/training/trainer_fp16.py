# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch

from torch.nn import Module
from torch.optim import Optimizer
from typing import Any, Optional, Dict
from deepgnn.pytorch.common.consts import FP16_APEX, FP16_AMP, FP16_NO
from deepgnn.pytorch.training.trainer import Trainer, BaseModel

try:
    import apex  # type: ignore

    is_apex_available = True
except ImportError:
    is_apex_available = False


class FP16Trainer(Trainer):
    """ "FP16Trainer supports FP16 mixed precision training with torch.amp or apex."""

    def __init__(self, args: argparse.Namespace):
        assert args.fp16 != FP16_APEX or is_apex_available
        super().__init__(args)

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

        if not self.fp16_enabled():
            return model

        if self.optimizer:
            if self.args.fp16 == FP16_AMP:
                self.grad_scaler = torch.cuda.amp.GradScaler()

            # For training, wrap apex for both model and optimizer.
            if self.args.fp16 == FP16_APEX:
                model, self.optimizer = apex.amp.initialize(
                    model, self.optimizer, opt_level=self.args.apex_opt_level
                )
        else:
            # For evaluation or inference, just wrap apex for model.
            if self.args.fp16 == FP16_APEX:
                model = apex.amp.initialize(model, opt_level=self.args.apex_opt_level)

        return model

    def _apex_backward(self, scaled_loss: torch.Tensor):
        scaled_loss.backward()

    def _apex_step(self):
        self.optimizer.step()

    def _amp_backward(self, loss):
        self.grad_scaler.scale(loss).backward()

    def _amp_step(self):
        self.grad_scaler.step(self.optimizer)

    def _train_one_step_internal(self, model: Module, data: Dict):
        if not self.fp16_enabled():
            return super()._train_one_step_internal(model, data)

        if self.args.fp16 == FP16_APEX:
            loss, score, label = model(data)

            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                self._apex_backward(scaled_loss)

            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    apex.amp.master_params(self.optimizer), self.args.grad_max_norm
                )

            self._apex_step()

        elif self.args.fp16 == FP16_AMP:
            with torch.cuda.amp.autocast():
                loss, score, label = model(data)

            self._amp_backward(loss)

            if self.args.clip_grad:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.grad_max_norm
                )

            self._amp_step()
            self.grad_scaler.update()
        else:
            raise RuntimeError("Unknown FP16 type.")

        return loss, score, label

    def _evaluate_one_step_internal(self, model: Module, data: Dict):
        if self.args.gpu and self.args.fp16 == FP16_AMP:
            with torch.cuda.amp.autocast():
                return super()._evaluate_one_step_internal(model, data)
        return super()._evaluate_one_step_internal(model, data)

    def _inference_one_step_internal(self, model: Module, data: Dict):
        if self.args.gpu and self.args.fp16 == FP16_AMP:
            with torch.cuda.amp.autocast():
                return super()._inference_one_step_internal(model, data)
        return super()._inference_one_step_internal(model, data)

    def fp16_enabled(self):
        return self.args.gpu and self.args.fp16 != FP16_NO

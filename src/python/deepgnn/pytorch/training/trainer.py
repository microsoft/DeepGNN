# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import time
import torch
import os
import numpy as np

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Optional, Dict, List

from deepgnn import (
    get_logger,
    log_telemetry,
    TrainMode,
    LOG_PROPS_EVENT_START_WORKER,
    LOG_PROPS_PLATFORM_PYTORCH,
    LOG_PROPS_EVENT_END_WORKER,
)
from deepgnn.pytorch.modeling.base_model import BaseModel
from deepgnn.pytorch.common.consts import PREFIX_CHECKPOINT, PREFIX_EMBEDDING
from deepgnn.pytorch.common.optimization import get_linear_schedule_with_warmup
from deepgnn.pytorch.common.utils import (
    dump_gpu_memory,
    print_model,
    tally_parameters,
    rotate_checkpoints,
    get_sorted_checkpoints,
    to_cuda,
)
from deepgnn.graph_engine.adl_uploader import AdlDataWriter


class Trainer:
    """
    Pytorch trainer, which controls the workflow of training/evaluation/inference.
    - Implementation in this class only works for sinle worker FP32 training requirement.
    - For FP16 mixed precision training, please use FP16Trainer.
    - For distributed training, please use DDPTrainer or HVDTrainer.
    """

    def __init__(self, args: argparse.Namespace):
        self.logger = get_logger()
        self.args = args
        # Initialize rank, local_rank, world_size.
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1

        # Initialize trainer state.
        self.step = 0
        self.global_step = 0
        self.epochs_trained = 0
        self.steps_in_epoch_trained = 0
        self.start_time = time.time()
        self.max_steps = 0

    def run(
        self,
        model: BaseModel,
        dataset: Any,
        optimizer: Optional[Optimizer] = None,
        eval_dataset_for_training: Any = None,
    ):
        """
        The main entry of trainer, which takes iterable dataset as input and
        perform training/evaluation/inference according to training mode.

        Args:
            model: target model.
            dataset: dataset for training, evaluation or inference.
            optimizer: optimizer for training.
            eval_during_train_dataset: optional dataset for evaluation
                during training.
        """

        self._init_max_steps()
        model = self._initialize(model, dataset, optimizer, eval_dataset_for_training)

        # On-demand enable telemetry.
        log_telemetry(
            self.logger,
            f"Training worker started. Model: {self.model_name}.",
            LOG_PROPS_EVENT_START_WORKER,
            f"{self.args.mode}",
            self.model_name,
            self.args.user_name,
            self.args.job_id,
            self.rank,
            self.world_size,
            LOG_PROPS_PLATFORM_PYTORCH,
        )

        result = None
        if self.args.mode == TrainMode.TRAIN:
            assert self.optimizer is not None
            self._train(model)
        elif self.args.mode == TrainMode.EVALUATE:
            result, loss = self._evaluate(model)
        elif self.args.mode == TrainMode.INFERENCE:
            self._inference(model)
        else:
            raise RuntimeError(f"Unsupported TrainMode:{self.args.mode}")

        log_telemetry(
            self.logger,
            f"Training worker finished. Model: {self.model_name}.",
            LOG_PROPS_EVENT_END_WORKER,
            f"{self.args.mode}",
            self.model_name,
            self.args.user_name,
            self.args.job_id,
            self.rank,
            self.world_size,
            LOG_PROPS_PLATFORM_PYTORCH,
        )

        if self.args.gpu:
            self.logger.info(self._wrap_log(dump_gpu_memory()))

        return result

    def _init_model(self, model: BaseModel):
        """Initialize model related setting."""
        self.model = model
        self.model_name = type(self.model).__name__

        if self.args.gpu:
            torch.cuda.set_device(self.local_rank)
            self.model.cuda()

        if self.rank == 0:
            if (
                not self.args.enable_adl_uploader
                or self.args.mode != TrainMode.INFERENCE
            ):
                os.makedirs(self.args.save_path, exist_ok=True)
            print_model(self.model)
            tally_parameters(self.model)

        self._load_checkpoint()
        return model

    def _init_optimizer(self, optimizer: Optimizer):
        """Initialize optimizer related setting."""
        self.lr_scheduler = self._create_lr_scheduler(optimizer)
        return optimizer

    def _initialize(
        self,
        model: BaseModel,
        dataset: Any,
        optimizer: Optional[Optimizer] = None,
        eval_dataset_for_training: Any = None,
    ):
        """Initialize model, datasets and optimizer and other settings."""
        model = self._init_model(model)
        self.dataset = dataset
        self.eval_dataset_for_training = eval_dataset_for_training
        self.optimizer = self._init_optimizer(optimizer) if optimizer else optimizer

        return model

    def _train(self, model: Module):
        """Perform training process on specified dataset."""
        self._init_summary_writer(prefix="train/worker")
        model.train()

        # Continous training from epoch and step saved in checkpoint.
        self.step = self.steps_in_epoch_trained
        for epoch in range(self.epochs_trained, self.args.num_epochs):
            self._train_one_epoch(model, epoch)

    def _train_one_epoch(self, model: Module, epoch: int):
        for i, data in enumerate(self.dataset):
            # Skip trained steps.
            if i < self.step:
                continue

            self.train_losses = []  # type: List[float]
            self.train_metrics = []  # type: List[float]
            self._train_one_step(model, data, epoch)
            if self._should_stop():
                break

        # Epoch finishes.
        self.step = 0
        if self.rank == 0 and epoch % self.args.save_ckpt_by_epochs == 0:
            self._save_checkpoint(epoch + 1)

    def _train_one_step(self, model: Module, data: Dict, epoch: int):
        """Update model state with one step data."""
        self._increment_step()
        self._prepare_data(data)

        self.optimizer.zero_grad()
        loss, pred, label = self._train_one_step_internal(model, data)
        self.train_losses.append(loss.data.item())

        if self.lr_scheduler:
            self.lr_scheduler.step()

        if (
            self.rank == 0
            and self.args.save_ckpt_by_steps > 0
            and self.step % self.args.save_ckpt_by_steps == 0
        ):
            self._save_checkpoint(epoch)

        # Calculate training metrics for one batch data.
        metric = (
            self.model.compute_metric([pred], [label]).data.item()
            if self.args.use_per_step_metrics
            else torch.tensor(0.0)
        )
        self.train_metrics.append(metric)

        if self.args.log_by_steps > 0 and self.step % self.args.log_by_steps == 0:
            train_loss = np.mean(self.train_losses)
            train_metric = np.mean(self.train_metrics)
            self.train_losses = []
            self.train_metrics = []
            duration = self._check_duration()
            self.summary_writer.add_scalar(
                "Training/Loss", train_loss, self.global_step
            )
            self.summary_writer.add_scalar("Training/Time", duration, self.global_step)
            if self.args.use_per_step_metrics:
                self.summary_writer.add_scalar(
                    f"Training/{self.model.metric_name()}",
                    train_metric,
                    self.global_step,
                )
            if self.lr_scheduler:
                self.summary_writer.add_scalar(
                    "Training/LR", self.lr_scheduler.get_last_lr()[0], self.global_step
                )

            self.logger.info(
                self._wrap_log(
                    f"epoch: {epoch}; step: {self.step:05d}; loss: {train_loss:.4f};"
                    + (
                        f" {self.model.metric_name()}: {train_metric:.4f}; time: {duration:.4f}s"
                        if self.args.use_per_step_metrics
                        else f" time: {duration:.4f}s"
                    )
                )
            )

        if (
            self.eval_dataset_for_training is not None
            and self.args.eval_during_train_by_steps > 0
            and self.step % self.args.eval_during_train_by_steps == 0
        ):
            model.eval()
            eval_metric, eval_loss = self._evaluate(model)
            if self.args.use_per_step_metrics:
                self.summary_writer.add_scalar(
                    f"Validation/{self.model.metric_name()}",
                    eval_metric,
                    self.global_step,
                )
                self.summary_writer.add_scalar(
                    f"Validation/Loss", eval_loss, self.global_step
                )
            self.logger.info(
                self._wrap_log(
                    f"epoch: {epoch}; step: {self.step:05d};"
                    + (
                        f" Validation/{self.model.metric_name()}: {eval_metric:.4f}, Validation/Loss: {eval_loss:.4f}"
                        if self.args.use_per_step_metrics
                        else ""
                    )
                )
            )
            model.train()

    def _train_one_step_internal(self, model: Module, data: Dict):
        """ "Perform one step forward-backward update."""
        loss, pred, label = model(data)
        loss.backward()
        if self.args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_max_norm)
        self.optimizer.step()
        return loss, pred, label

    def _evaluate(self, model: Module):
        """Perform evaluation process on specified dataset."""
        if self.args.mode != TrainMode.TRAIN:
            self._init_summary_writer(prefix="evaluate/worker")
        model.eval()

        preds = []
        labels = []
        eval_metrics = []
        eval_losses = []
        data_size = 0
        dataset = (
            self.eval_dataset_for_training
            if self.args.mode == TrainMode.TRAIN
            else self.dataset
        )
        assert dataset is not None
        with torch.no_grad():
            for data in dataset:
                pred, label, loss = self._evaluate_one_step(model, data)
                data_size += pred.size(0)
                eval_losses.append(loss.data.item())

                if self.args.use_per_step_metrics:
                    metric = self.model.compute_metric([pred], [label])
                    eval_metrics.append(metric.data.item())
                else:
                    preds.append(pred.detach().cpu())
                    labels.append(label.detach().cpu())

                if self._should_stop():
                    break

        if self.args.use_per_step_metrics:
            eval_metric = np.mean(eval_metrics) if len(eval_metrics) > 0 else 0
            eval_metric = torch.tensor(eval_metric)
        else:
            eval_metric = self.model.compute_metric(preds, labels)
        self.logger.info(
            self._wrap_log(
                f"Evaluation {self.model.metric_name()}: {eval_metric:.4f}; data size: {data_size};"
            )
        )
        eval_loss = np.mean(eval_losses) if len(eval_losses) > 0 else 0
        eval_loss = torch.tensor(eval_loss)
        return eval_metric, eval_loss

    def _evaluate_one_step(self, model: Module, data: Dict):
        """Evaluate model with one step data."""
        is_eval_during_training = self.args.mode == TrainMode.TRAIN
        if not is_eval_during_training:
            self._increment_step()
        self._prepare_data(data)

        loss, pred, label = self._evaluate_one_step_internal(model, data)

        if (
            not is_eval_during_training
            and self.args.log_by_steps > 0
            and self.step % self.args.log_by_steps == 0
        ):
            duration = self._check_duration()
            loss_val = loss.data.item()
            self.summary_writer.add_scalar(
                "Evaluation/Loss", loss_val, self.global_step
            )
            self.summary_writer.add_scalar(
                "Evaluation/Time", duration, self.global_step
            )
            self.logger.info(
                self._wrap_log(
                    f"step: {self.step:05d}; loss: {loss_val:.4f}; time: {duration:.4f}s"
                )
            )

        return pred, label, loss

    def _evaluate_one_step_internal(self, model: Module, data: Dict):
        return model(data)

    def _inference(self, model: Module):
        """Perform inference process on specified dataset."""
        self._init_summary_writer(prefix="inference/worker")
        model.eval()

        with self._get_embedding_writer() as fp:
            with torch.no_grad():
                for data in self.dataset:
                    self._inference_one_step(model, data, fp)
                    if self._should_stop():
                        break

    def _inference_one_step(self, model: Module, data: Dict, fp: Any):
        """Perform inference on step(batch) data, write embedding to specified file."""
        self._increment_step()
        self._prepare_data(data)

        embedding = self._inference_one_step_internal(model, data)
        self.model.output_embedding(fp, data, embedding)

        if self.args.log_by_steps > 0 and self.step % self.args.log_by_steps == 0:
            duration = self._check_duration()
            self.summary_writer.add_scalar("Inference/Time", duration, self.global_step)
            self.logger.info(
                self._wrap_log(f"step: {self.step:05d}; time: {duration:.4f}s")
            )

    def _inference_one_step_internal(self, model: Module, data: Dict):
        return self.model.get_embedding(data)

    def _increment_step(self):
        """Increment local and global step."""
        self.step += 1
        self.global_step += 1

    def _should_stop(self):
        """Check whether to stop evaluation/inference or current training epoch."""
        return self.max_steps > 0 and self.step >= self.max_steps

    def _init_summary_writer(self, prefix: str):
        """Initialize tensorboard summary writer."""
        self.summary_writer = SummaryWriter(
            os.path.join(self.args.metric_dir, f"{prefix}-{self.rank}")
        )

    def _check_duration(self):
        """Get duration since last check."""
        duration = time.time() - self.start_time
        self.start_time = time.time()
        return duration

    def _prepare_data(self, data: Dict):
        """Prepare one step data."""
        if self.args.gpu:
            to_cuda(data)

    def _wrap_log(self, content: str):
        """Helper function to append world_size and rank in the front of log content."""
        return f"[{self.world_size},{self.rank}] {content}"

    def _create_lr_scheduler(self, optimizer: Optimizer):
        """Setup the learning rate scheduler."""
        num_training_steps = self.max_steps * self.args.num_epochs
        return (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup * num_training_steps,
                num_training_steps=num_training_steps,
            )
            if self.max_steps > 0
            else None
        )

    def _save_checkpoint(self, epoch: int):
        """Save model state with trained epoch and steps in epoch."""
        # Don't save for last step to avoid duplication with ckpt after epoch finished.
        if self.max_steps > 0 and self.step == self.max_steps:
            return

        save_path = os.path.join(
            f"{self.args.save_path}",
            f"{PREFIX_CHECKPOINT}-{epoch:03}-{self.step:06}.pt",
        )
        torch.save(
            {"state_dict": self.model.state_dict(), "epoch": epoch, "step": self.step},
            save_path,
        )
        self.logger.info(self._wrap_log(f"Saved checkpoint to {save_path}."))
        rotate_checkpoints(self.args.save_path, self.args.max_saved_ckpts)

    def _load_checkpoint(self, ckpt_path: str = None):
        """Load model state, trained epochs and steps. Load from specified checkpoint if given,
        or load from latest checkpoint in model_dir.
        """
        if not ckpt_path:
            # Search and sort checkpoints from model path.
            ckpts = get_sorted_checkpoints(self.args.model_dir)
            ckpt_path = ckpts[-1] if len(ckpts) > 0 else None

        if ckpt_path is not None:

            init_ckpt = torch.load(ckpt_path, map_location="cpu")
            self.epochs_trained = init_ckpt["epoch"]
            self.steps_in_epoch_trained = init_ckpt["step"]

            # Only worker with rank 0 loads state dict.
            if self.rank == 0:
                self.model.load_state_dict(init_ckpt["state_dict"])
                self.logger.info(
                    self._wrap_log(
                        f"Loaded initial checkpoint: {ckpt_path},"
                        f" trained epochs: {self.epochs_trained}, steps: {self.steps_in_epoch_trained}"
                    )
                )

            del init_ckpt

    def _init_max_steps(self):
        """Initialize max steps per epoch."""
        self.max_steps = (
            self.args.max_samples // (self.world_size * self.args.batch_size)
            if self.args.max_samples > 0
            else -1
        )
        self.logger.info(self._wrap_log(f"Max steps per epoch:{self.max_steps}"))

    def _get_embedding_writer(self):
        """Get embedding writer, which could be local file pointer or Adl writer."""
        embed_path = os.path.join(
            self.args.save_path, f"{PREFIX_EMBEDDING}-{self.rank}"
        )
        if self.args.enable_adl_uploader:
            uploader = AdlDataWriter(
                process_num=self.args.uploader_process_num,
                threads_per_process=self.args.uploader_threads_num,
                queue_size=200,
                store_name=self.args.uploader_store_name,
                file_path_prefix=embed_path,
            )

            return uploader
        return open(embed_path + ".tsv", "w", encoding="utf-8")

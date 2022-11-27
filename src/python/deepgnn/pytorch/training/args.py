# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Configure command line arguments for pytorch trainers."""

import argparse
import uuid
from deepgnn.pytorch.common.consts import FP16_AMP, FP16_APEX, FP16_NO
from deepgnn import TrainerType, TrainMode, get_current_user
from deepgnn.graph_engine.snark.client import PartitionStorageType


# fmt: off
def init_trainer_args(parser: argparse.ArgumentParser):
    """Configure trainer."""
    group = parser.add_argument_group("Trainer Parameters")
    group.add_argument("--trainer", type=TrainerType, default=TrainerType.BASE, choices=[TrainerType.BASE, TrainerType.HVD, TrainerType.DDP], help="Trainer type.")
    group.add_argument("--user_name", type=str, default=get_current_user(), help="User name when running jobs.")
    group.add_argument("--mode", type=TrainMode, default=TrainMode.TRAIN, choices=[TrainMode.TRAIN, TrainMode.EVALUATE, TrainMode.INFERENCE], help="Run mode.")
    group.add_argument("--num_epochs", default=1, type=int, help="Number of epochs for training.")
    group.add_argument("--model_dir", type=str, default="", help="path to load model checkpoint")
    group.add_argument("--save_path", type=str, default="", help="file path to save embedding or new checkpoints.")
    group.add_argument("--metric_dir", type=str, default="", help="tensorboard metrics dir.")
    group.add_argument("--sort_ckpt_by_mtime", action="store_true", help="Sort model checkpoints by modified time. If not set, sort checkpoints by model name.")
    group.add_argument("--save_ckpt_by_steps", default=0, type=int, help="Number of steps to save model checkpoint. 0 means not saving by step.")
    group.add_argument("--save_ckpt_by_epochs", type=int, default=1, help="Number of epochs to save model checkpoint.")
    group.add_argument("--log_by_steps", type=int, default=20, help="Number of steps to log information.")
    group.add_argument("--eval_during_train_by_steps", default=0, type=int, help="Number of steps to run evaluation during training. 0 means do not run evaluation during training.")
    group.add_argument("--max_saved_ckpts", default=0, type=int, help="If a postitive number is passed, it will delete older checkpoints in save_path.")
    group.add_argument("--gpu", action="store_true", help="Enable gpu training or not.")
    group.add_argument("--job_id", type=str, default=str(uuid.uuid4())[:8], help="job id (uuid).")
    group.add_argument("--local_rank", type=int, default=0, help="During distributed training, the local rank of the process.")
    group.add_argument("--max_samples", type=int, default=0, help="Total number of data to be trained.")
    group.add_argument("--warmup", type=float, default=0.0002, help="Warmup ration of optimizer.")
    group.add_argument("--clip_grad", action="store_true", help="Whether to enable clipping gradient norm.")
    group.add_argument("--grad_max_norm", type=float, default=1.0, help="Max norm of the gradients.")
    group.add_argument("--use_per_step_metrics", action="store_true", help="whether to calculate metric per step.")
    group.add_argument("--enable_adl_uploader", action="store_true", help="Enable collect telemetry data.")
    group.add_argument("--uploader_store_name", type=str, default="", help="Azure data lake gen1 store name for adl uploader.")
    group.add_argument("--uploader_process_num", type=int, default=1, help="total process number for adl uploader.")
    group.add_argument("--uploader_threads_num", type=int, default=12, help="Thread number per process for adl uploader.")
    group.add_argument("--disable_ib", action="store_true", help="Disable Infiniband.")
    group.add_argument("--storage_type", type=lambda type: PartitionStorageType[type], default=PartitionStorageType.memory, choices=list(PartitionStorageType.__members__.keys()) + list(PartitionStorageType), help="Partition storage backing to use, eg memory or disk.")  # type: ignore
    group.add_argument("--config_path", type=str, default="", help="Directory where HDFS or other config files are stored.")
    group.add_argument("--stream", action="store_true", default=False, help="If ADL data path, stream directly to memory or download to disk first.")


def init_fp16_args(parser: argparse.ArgumentParser):
    """Configure arguments for training with half-precision floats."""
    group = parser.add_argument_group("FP16 Parameters")
    group.add_argument("--fp16", type=str, default=FP16_AMP, choices=[FP16_AMP, FP16_APEX, FP16_NO], help="Enable fp16 mix precision training.")
    group.add_argument("--apex_opt_level", type=str, default="O2", help="Apex FP16 mixed precision training opt level.")
# fmt: on

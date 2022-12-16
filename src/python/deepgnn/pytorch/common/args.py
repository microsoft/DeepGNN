# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Command line arguments to configure training."""
from typing import Optional, List, Callable
import argparse
import uuid
from deepgnn.graph_engine import define_param_graph_engine
from deepgnn import TrainerType, TrainMode, get_current_user, get_logger
from deepgnn.graph_engine.snark.client import PartitionStorageType


# fmt: off
def init_model_common_args(parser: argparse.ArgumentParser):
    """Model configuration."""
    group = parser.add_argument_group("Model Common Parameters")
    group.add_argument("--feature_type", default="float", type=str, choices=["float", "uint64", "binary"], help="Feature type.")
    group.add_argument("--feature_idx", default=-1, type=int, help="Feature index.")
    group.add_argument("--feature_dim", default=0, type=int, help="Feature dimension.")
    group.add_argument("--label_idx", default=1, type=int, help="Label index.")
    group.add_argument("--label_dim", default=0, type=int, help="Label dimension.")
    group.add_argument("--dim", default=256, type=int, help="Dimension of embedding.")
    group.add_argument("--num_negs", default=5, type=int, help="Number of negative samplings.")
    group.add_argument("--fanouts", default=[10, 10], nargs="+", type=int, help="Graphsage fanouts.")
    group.add_argument("--neighbor_count", type=int, default=10, help="Number of neighbors to sample of each node")
    group.add_argument("--meta_dir", type=str, default="", help="Local meta data dir.")
    group.add_argument("--featenc_config", default=None, type=str, help="Config file name of feature encoder.")
    group.add_argument("--model_args", type=str, default="", help="Other arguments for model with json format.")
    group.add_argument("--node_type", default=0, type=int, help="Node type to train/evaluate model.")


def init_dataset_args(parser: argparse.ArgumentParser):
    """Dataset configuration."""
    group = parser.add_argument_group("Dataset Parameters")
    group.add_argument("--partitions", type=int, nargs="+", default=[0])
    group.add_argument("--sample_file", type=str, default="", help="File which contains node id to calculate the embedding. It could be filename pattern.")
    group.add_argument("--batch_size", default=512, type=int, help="Mini-batch size.")
    group.add_argument("--prefetch_size", default=16, type=int, help="Number of queries to prefetch.")
    group.add_argument("--num_parallel", default=2, type=int, help="Number of graph queries to run in parallel.")
    group.add_argument("--data_parallel_num", default=2, type=int, help="How many subprocesses to use for data loading.")
    group.add_argument("--prefetch_factor", default=2, type=int, help="Number of samples loaded in advance by each worker.")
    group.add_argument("--max_id", type=int, help="Max node id.")
    group.add_argument("--strategy", type=str, default="RandomWithoutReplacement", help="GraphEngine Sampler: Weighted/Random/RandomWithoutReplacement.")


def init_optimizer_args(parser: argparse.ArgumentParser):
    """Optimizer configuration."""
    group = parser.add_argument_group("Optimizer Parameters")
    group.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")


def init_other_args(parser: argparse.ArgumentParser):
    """Misc configuration."""
    group = parser.add_argument_group("Other Parameters")
    group.add_argument("--seed", type=int, default=None, help="Random seed for initialization")


def init_common_args(parser: argparse.ArgumentParser):
    """Configure all components."""
    define_param_graph_engine(parser)
    init_model_common_args(parser)
    init_dataset_args(parser)
    init_optimizer_args(parser)
    init_other_args(parser)


def init_trainer_args(parser: argparse.ArgumentParser):
    """Configure trainer."""
    group = parser.add_argument_group("Trainer Parameters")
    group.add_argument("--trainer", type=TrainerType, default=TrainerType.BASE, choices=[TrainerType.BASE, TrainerType.HVD], help="Trainer type.")
    group.add_argument("--user_name", type=str, default=get_current_user(), help="User name when running jobs.")
    group.add_argument("--mode", type=TrainMode, default=TrainMode.TRAIN, choices=[TrainMode.TRAIN, TrainMode.EVALUATE, TrainMode.INFERENCE], help="Run mode.")
    group.add_argument("--num_epochs", default=1, type=int, help="Number of epochs for training.")
    group.add_argument("--model_dir", type=str, default="", help="Path to save logs and checkpoints.")
    group.add_argument("--metric_dir", type=str, default="", help="tensorboard metrics dir.")
    group.add_argument("--log_by_steps", type=int, default=20, help="Number of steps to log information.")
    group.add_argument("--gpu", action="store_true", help="Enable gpu training or not.")
    group.add_argument("--job_id", type=str, default=str(uuid.uuid4())[:8], help="job id (uuid).")
    group.add_argument("--local_rank", type=int, default=0, help="During distributed training, the local rank of the process.")
    group.add_argument("--max_samples", type=int, default=0, help="Total number of data to be trained.")
    group.add_argument("--warmup", type=float, default=0.0002, help="Warmup ration of optimizer.")
    group.add_argument("--clip_grad", action="store_true", help="Whether to enable clipping gradient norm.")
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
    group.add_argument("--fp16", action="store_true", default=False, help="Enable fp16 mix precision training.")
# fmt: on


def get_args(init_arg_fn: Optional[Callable] = None, run_args: Optional[List] = None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Initialize common parameters, including model, dataset, optimizer etc.
    init_common_args(parser)

    # Initialize trainer paramaters.
    init_trainer_args(parser)

    # Initialize fp16 related paramaters.
    init_fp16_args(parser)

    if init_arg_fn is not None:
        init_arg_fn(parser)

    args, _ = (
        parser.parse_known_args()
        if run_args is None
        else parser.parse_known_args(run_args)
    )
    for arg in dir(args):
        if not arg.startswith("_"):
            get_logger().info(f"{arg}={getattr(args, arg)}")

    return args

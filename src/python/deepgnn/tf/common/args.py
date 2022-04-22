# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import uuid
from deepgnn.arg_types import str2list_str, str2list_int
from deepgnn.graph_engine import define_param_graph_engine
from deepgnn import TrainMode, TrainerType, get_current_user
from deepgnn import get_logger
from deepgnn.graph_engine.snark.client import PartitionStorageType


def log_all_parameters(args):
    logger = get_logger()

    for arg in vars(args):
        logger.info("{0}:\t{1}".format(arg, getattr(args, arg)))


# fmt:off
def define_param_common(parser):
    group = parser.add_argument_group("Common Parameters")
    group.add_argument("--mode", type=TrainMode, default=TrainMode.TRAIN, choices=list(TrainMode))
    group.add_argument("--user_name", type=str, default=get_current_user(), help="User name when running jobs.")
    group.add_argument("--model_dir", type=str, default="/tmp/model_dir", help="model checkpoint folder.",)
    group.add_argument("--trainer", type=TrainerType, default=TrainerType.PS, choices=[TrainerType.PS, TrainerType.HVD, TrainerType.MULTINODE], help="use PSTrainer, HorovodTrainer or MultiNodeTrainer.",)
    group.add_argument("--eager", action="store_true", default=False, help="Enable eager execution.")
    group.add_argument("--job_id", type=str, default=str(uuid.uuid4())[:8], help="job id (uuid).")
    group.add_argument("--seed", type=int, default=None, help="Random seed for initialization")
    group.add_argument("--gpu", action="store_true", default=False, help="use GPU device")
    group.add_argument("--partitions", type=int, nargs="+", default=[0])
    group.add_argument("--steps_per_epoch", type=int, default=None, help="How many steps will be run.")
    group.add_argument("--storage_type", type=lambda type: PartitionStorageType[type], default=PartitionStorageType.memory, choices=list(PartitionStorageType.__members__.keys()) + list(PartitionStorageType), help="Partition storage backing to use, eg memory or disk.")
    group.add_argument("--config_path", type=str, default="", help="Directory where HDFS or other config files are stored.")
    group.add_argument("--stream", action="store_true", default=False, help="If ADL data path, stream directly to memory or download to disk first.")


def define_param_ps_dist_training(parser):
    group = parser.add_argument_group("PSTrainer Parameters")
    group.add_argument("--job_name", type=str, default=None, choices=["worker", "ps"], help="(PSTrainer Only) Job role.",)
    group.add_argument("--task_index", type=int, default=0, help="(PSTainer Only) job index.")
    group.add_argument("--ps_hosts", type=str2list_str, default="", help="(PSTrainer Only) parameter server hosts.",)
    group.add_argument("--worker_hosts", type=str2list_str, default="", help="(PSTrainer Only) worker hosts.",)


def define_param_tf_hooks(parser):
    group = parser.add_argument_group("TF Hooks Parameters")
    group.add_argument("--checkpoint_save_secs", type=int, default=3600, help="save model checkpoint every N secs.",)
    group.add_argument("--profiler_save_secs", type=int, default=180, help="(TF1Trainer) save profiler every N secs.",)
    group.add_argument("--log_save_steps", type=int, default=20, help="write log every N steps.")
    group.add_argument("--summary_save_steps", type=int, default=100, help="save summary log every N steps.",)
    group.add_argument("--profile_batch", type=str2list_int, default="100,110", help="(TF2Trainer) Profile the batch(es) to trace performance.",)


def define_param_prefetch(parser):
    group = parser.add_argument_group("Query Prefetch Parameters")
    group.add_argument("--prefetch_queue_size", type=int, default=16, help="Number of queries to prefetch",)
    group.add_argument("--prefetch_worker_size", type=int, default=2, help="Num of parallel workers to run graph queries.",)


def define_param_horovod(parser):
    parser.add_argument_group("Horovod TF Parameters")
# fmt:on


def import_default_parameters(parser):
    define_param_common(parser)
    define_param_horovod(parser)
    define_param_ps_dist_training(parser)
    define_param_graph_engine(parser)
    define_param_prefetch(parser)
    define_param_tf_hooks(parser)

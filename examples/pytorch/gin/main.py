import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common import MRR, Accuracy
from deepgnn.pytorch.common.utils import get_feature_type, set_seed
from deepgnn.pytorch.encoding import get_feature_encoder
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.training import run_dist
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import (
    Graph,
    FeatureType,
    CSVNodeSampler,
    GENodeSampler,
    SamplingStrategy,
    GraphEngineBackend,
)

from tqdm import tqdm
from model import GIN

criterion = nn.CrossEntropyLoss()

def init_args(parser: argparse.Namespace):
    group = parser.add_argument_group("GIN Parameters")
    group.add_argument("--algo", type=str, default="supervised")
    group.add_argument("--edge_type", type=np.ndarray, default=np.ndarray([]))


def create_model(args: argparse.Namespace):

    # set seed before instantiating the model
    if args.seed:
        set_seed(args.seed)

    # feature_enc = get_feature_encoder(args)

    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.algo == "supervised":
        get_logger().info(f"Creating GIN model with seed:{args.seed}.")
        return GIN(
            metric=Accuracy(),
            num_layers=1, 
            num_mlp_layers=2, 
            input_dim=1433,
            hidden_dim=128, 
            output_dim=7, 
            final_dropout=0.5, 
            learn_eps=False, 
            edge_type=args.edge_type,
            feature_type=get_feature_type(args.feature_type),
            feature_dim=args.feature_dim,
            feature_idx=args.feature_idx,
            label_idx=args.label_idx,
            label_dim=args.label_dim,
            graph_pooling_type="sum", 
            neighbor_pooling_type="sum", 
            device=device,
        )
    else:
        raise RuntimeError(f"Unknown algo: {args.algo}")


def create_dataset(
    args: argparse.Namespace,
    model: BaseModel,
    rank: int = 0,
    world_size: int = 1,
    backend: GraphEngineBackend = None,
):
    if args.mode == TrainMode.INFERENCE:
        return TorchDeepGNNDataset(
            sampler_class=CSVNodeSampler,
            backend=backend,
            num_workers=world_size,
            worker_index=rank,
            batch_size=args.batch_size,
            sample_file=args.sample_file,
            query_fn=model.query,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
        )
    else:
        return TorchDeepGNNDataset(
            sampler_class=GENodeSampler,
            backend=backend,
            sample_num=args.max_id,
            num_workers=world_size,
            worker_index=rank,
            node_types=np.array([args.node_type], dtype=np.int32),
            batch_size=args.batch_size,
            query_fn=model.query,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
            strategy=SamplingStrategy.RandomWithoutReplacement,
        )


def create_optimizer(args: argparse.Namespace, model: BaseModel, world_size: int):
    return torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate * world_size,
    )


criterion = nn.CrossEntropyLoss()

def _main():
    # setup default logging component.
    setup_default_logging_config(enable_telemetry=True)

    random.seed(42)
    # run_dist is the unified entry for pytorch model distributed training/evaluation/inference.
    # User only needs to prepare initializing function for model, dataset, optimizer and args.
    # reference: `deepgnn/pytorch/training/factory.py`
    run_dist(
        init_model_fn=create_model,
        init_dataset_fn=create_dataset,
        init_optimizer_fn=create_optimizer,
        init_args_fn=init_args,
    )


if __name__ == "__main__":
    _main()

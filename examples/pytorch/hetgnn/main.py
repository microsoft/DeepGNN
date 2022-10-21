# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch
from deepgnn import TrainMode, setup_default_logging_config
from deepgnn import get_logger
from deepgnn.pytorch.common.utils import get_python_type, set_seed
from deepgnn.pytorch.modeling import BaseModel
from deepgnn.pytorch.training import run_dist
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from deepgnn.graph_engine import CSVNodeSampler, GraphEngineBackend
from args import init_args  # type: ignore
from model import HetGnnModel  # type: ignore
from sampler import HetGnnDataSampler  # type: ignore


def create_model(args: argparse.Namespace):
    get_logger().info(f"Creating HetGnnModel with seed:{args.seed}.")
    # set seed before instantiating the model
    if args.seed:
        set_seed(args.seed)

    return HetGnnModel(
        node_type_count=args.node_type_count,
        neighbor_count=args.neighbor_count,
        embed_d=args.feature_dim,  # currently feature dimention is equal to embedding dimention.
        feature_type=get_python_type(args.feature_type),
        feature_idx=args.feature_idx,
        feature_dim=args.feature_dim,
    )

def train_func(config: Dict):
    batch_size = config["batch_size"]
    epochs = config["num_epochs"]
    world_size = session.get_world_size()

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
            query_fn=model.query_inference,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
            batch_size=args.batch_size,
            sample_file=args.sample_file,
        )
    else:
        return TorchDeepGNNDataset(
            sampler_class=HetGnnDataSampler,
            backend=backend,
            query_fn=model.query,
            prefetch_queue_size=10,
            prefetch_worker_size=2,
            num_nodes=args.max_id // world_size,
            batch_size=args.batch_size,
            node_type_count=args.node_type_count,
            walk_length=args.walk_length,
        )

    train_dataloader = train.torch.prepare_data_loader(train_dataloader)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"] * world_size,
        weight_decay=0,
    )
    loss_results = []

    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_dataloader):
            loss, score, label = model(X)
            #loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{num_nodes:>5d}]")
        #session.report(dict(loss=loss))
        torch.save(
            {"state_dict": model_original.state_dict(), "epoch": epoch},
            os.path.join(config["save_path"], f"gnnmodel-{epoch:03}.pt"),
        )

    torch.save(
        model_original.state_dict(),
        os.path.join(config["save_path"], f"gnnmodel.pt"),
    )
    # return required for backwards compatibility with the old API
    # TODO(team-ml) clean up and remove return
    return loss_results


if __name__ == "__main__":
    from deepgnn.pytorch.training.args import get_args
    args = get_args(init_args, run_args=None)

    ray.init()
    trainer = TorchTrainer(
        train_func,
        train_loop_config=vars(args),
        run_config=RunConfig(verbose=1),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )
    result = trainer.fit()

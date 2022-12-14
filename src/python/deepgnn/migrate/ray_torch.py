# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to migrate torch example to use Ray Train."""
import argparse
from pathlib import Path
import pasta
from pasta.augment import rename

train_fn = """
def train_func(config: Dict):
    train.torch.enable_reproducibility(seed=session.get_world_rank())

    model = INIT_MODEL_FN
    model = train.torch.prepare_model(model)
    model.train()

    optimizer = INIT_OPTIMIZER_FN
    optimizer = train.torch.prepare_optimizer(optimizer)

    loss_fn = nn.CrossEntropyLoss()  # TODO loss fn w/ model loss

    SAMPLE_NUM = 152410
    BATCH_SIZE = 512

    cl = Client("/tmp/reddit", [0])
    query_obj = PTGSupervisedGraphSageQuery(
        label_meta=np.array([[0, 50]]),
        feature_meta=np.array([[1, 300]]),
        feature_type=np.float32,
        edge_type=0,
        fanouts=[5, 5],
    )
    N_WORKERS = 2
    dataset = TorchDeepGNNDataset(
        sampler_class=GENodeSampler,
        backend=type("Backend", (object,), {"graph": cl})(),  # type: ignore
        sample_num=SAMPLE_NUM,
        num_workers=1,  # sampler sample count is divided by num_workers and data parralele count
        worker_index=0,
        node_types=np.array([0], dtype=np.int32),
        batch_size=BATCH_SIZE,
        query_fn=query_obj.query,
        strategy=SamplingStrategy.RandomWithoutReplacement,
        prefetch_queue_size=10,
        prefetch_worker_size=N_WORKERS,
    )
    dataset = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=N_WORKERS,
    )
    for epoch in range(5):
        metrics = []
        losses = []
        for i, batch in enumerate(dataset):
            if i == 0:
                print(batch["inputs"])
            continue
            scores = model(batch)[0]
            labels = batch["label"].squeeze().argmax(1).detach()

            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics.append((scores.squeeze().argmax(1) == labels).float().mean().item())
            losses.append(loss.item())

            #if i >= SAMPLE_NUM / BATCH_SIZE / session.get_world_size():
            #    break

        session.report(
            {
                "metric": np.mean(metrics),
                "loss": np.mean(losses),
            }
        )

import ray
ray.init()

trainer = TorchTrainer(
    train_func,
    train_loop_config={},
    run_config=RunConfig(),
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False, resources_per_worker={"CPU": 2}),
)
result = trainer.fit()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate to new versions of DeepgNN.")
    parser.add_argument(
        "--main_path", type=str, required=True, help="Directory to migrate."
    )
    args = parser.parse_args()

    with open(args.main_path, "r") as file:
        raw_input = file.read()

    #raw_input = raw_input.replace("FeatureType.FLOAT", "np.float32")

    tree = pasta.parse(raw_input)

    rename.rename_external(
        tree, "deepgnn.pytorch.training.run_dist", train_fn
    )

    raw_output = pasta.dump(tree)

    #raw_output = raw_output.replace("dtype_temp", "np.dtype")

    with open(args.main_path, "w") as file:
        file.write(raw_output)

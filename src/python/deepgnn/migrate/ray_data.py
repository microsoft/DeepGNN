# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to migrate dataset usage to use Ray Data."""
import os
import argparse
import pasta
from pasta.augment import rename


def ray_dataset(
    sampler_class=FileNodeSampler,
    backend=backend,
    query_fn=model.q.query_training,
    prefetch_queue_size=2,
    prefetch_worker_size=2,
    sample_files=args.sample_file,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    worker_index=rank,
    num_workers=world_size,
    **kwargs
):

    from ray.data import DatasetPipeline
    address = "localhost:9999"
    # TODO needs to be peristents
    s = Server(address, args.data_dir, 0, args.partitions)
    g = DistributedClient([address])

    if sampler_class == "":
        dataset = ray.data.range(TODO).repartition(TODO // TODO)

    pipe = dataset.window(TODO)    
    def transform_batch(idx: list) -> dict:
        return model.query(g, idx)
    pipe = pipe.map_batches(transform_batch)
    return pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate to new versions of DeepgNN.")
    parser.add_argument(
        "--main_path",
        type=str,
        required=True,
        help="Path to main file that contains TorchDeepGNNDataset.",
    )
    args = parser.parse_args()

    with open(args.main_path, "r") as file:
        raw_input = file.read()

    tree = pasta.parse(raw_input)

    rename.rename_external(
        tree, "deepgnn.pytorch.common.dataset.TorchDeepGNNDataset", "ray_dataset"
    )

    raw_output = pasta.dump(tree)

    raw_output = raw_output.replace("set_seed(args.seed)", "")

    with open(args.main_path, "w") as file:
        file.write(raw_output)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to migrate torch example to use Ray Train."""
import os
import argparse
from pathlib import Path
import pasta
from pasta.augment import rename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate to new versions of DeepgNN.")
    parser.add_argument(
        "--main_path",
        type=str,
        required=True,
        help="Path to main file that contains run_dist.",
    )
    parser.add_argument(
        "--hvd",
        action="store_true",
        default=False,
        help="Convert to Horovod trainer instead of standard DDP trainer.",
    )
    args = parser.parse_args()

    dirname = os.path.dirname(args.main_path)
    if dirname == "":
        dirname = "."
    with open(
        os.path.join(os.path.dirname(__file__), "data", "ray_util_tf.py"), "r"
    ) as file:
        ray_util = file.read()
    if args.hvd:
        # https://docs.ray.io/en/latest/_modules/ray/train/horovod/horovod_trainer.html

        ray_util = ray_util.replace(
            "from ray.train.tensorflow import TensorflowTrainer",
            "import horovod.tensorflow as hvd\nimport ray.train.tensorflow\nfrom ray.train.horovod import HorovodTrainer",
        )
        ray_util = ray_util.replace("train.torch.accelerate(args.fp16)", "hvd.init()")
        ray_util = ray_util.replace(
            "model = train.torch.prepare_model(model, move_to_device=args.gpu)",
            "device = train.torch.get_device()\n    model.to(device)",
        )
        ray_util = ray_util.replace(
            "train.torch.prepare_optimizer(optimizer)",
            "hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())",
        )
        ray_util = ray_util.replace("TensorflowTrainer", "HorovodTrainer")

    with open(os.path.join(dirname, "ray_util.py"), "w") as file:
        file.write(ray_util)

    with open(args.main_path, "r") as file:
        raw_input = file.read()

    tree = pasta.parse(raw_input)

    # rename.rename_external(
    #    tree, "deepgnn.pytorch.training.run_dist", "ray_util.run_ray"
    # )

    raw_output = pasta.dump(tree)

    raw_output = raw_output.replace(
        "from deepgnn.tf.common.trainer_factory import get_trainer",
        "from ray_util import run_ray",
    )
    raw_output = raw_output.replace("trainer = get_trainer(param)", "trainer = param")
    raw_output = raw_output.replace("trainer.train", "run_ray")
    raw_output = raw_output.replace("trainer.evaluate", "run_ray")
    raw_output = raw_output.replace("trainer.inference", "run_ray")

    raw_output = raw_output.replace("trainer.worker_size", "trainer.prefetch_worker_size")
    raw_output = raw_output.replace(" * trainer.lr_scaler", "")

    with open(args.main_path, "w") as file:
        file.write(raw_output)

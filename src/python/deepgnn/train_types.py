# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum


class TrainerType(Enum):
    """Trainer types:
    DeepGNN currently support 4 trainer types:
    * BASE: The most basic local trainer, used for simple experiments.
    * PS: Parameter server based distributed trainer, used for tensorflow models only.
    * HVD: Horovod based distributed trainer, supports both tensorflow and pytorch models.
    * DDP: DistributedDataParallel based distributed trainer, used for pytorch models only.
    """

    BASE = "base"
    PS = "ps"
    MULTINODE = "multinode"
    HVD = "hvd"
    DDP = "ddp"

    def __str__(self):
        return self.value


class TrainMode(Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    INFERENCE = "inference"

    def __str__(self):
        return self.value

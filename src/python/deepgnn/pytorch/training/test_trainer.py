# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
import tempfile
from dataclasses import dataclass

from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from random import randint

from deepgnn import TrainMode
from deepgnn.pytorch.common.consts import PREFIX_EMBEDDING, PREFIX_CHECKPOINT
from deepgnn.pytorch.common.utils import get_sorted_checkpoints
from deepgnn.pytorch.training.trainer import Trainer
from deepgnn.pytorch.training.trainer_fp16 import FP16Trainer
from deepgnn.pytorch.training.trainer_ddp import DDPTrainer

hvd_enabled = False
try:
    from deepgnn.pytorch.training.trainer_hvd import HVDTrainer

    hvd_enabled = True
except:
    hvd_enabled = False


@dataclass
class TrainingArguments:
    model: str = "mock"
    user_name: str = "test_user"
    job_id: str = "test_id"
    mode: TrainMode = TrainMode.TRAIN
    batch_size: int = 1
    num_epochs: int = 1
    model_dir: str = "."
    metric_dir: str = "."
    save_path: str = "."
    max_saved_ckpts: int = 0
    save_ckpt_by_steps: int = 0
    save_ckpt_by_epochs: int = 1
    log_by_steps: int = 20
    eval_during_train_by_steps: int = 0
    sort_ckpt_by_mtime: bool = False
    gpu: bool = False
    fp16: bool = False
    ddp: bool = False
    apex: bool = False
    apex_opt_level: str = "O2"
    local_rank: int = 0
    max_samples: int = 0
    warmup: float = 0.0002
    use_per_step_metrics: bool = False
    clip_grad: bool = False
    grad_max_norm: float = 1.0
    enable_adl_uploader: bool = False
    disable_ib: bool = False


class MockModel(torch.nn.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())

    def forward(self, data):
        label = data["label"]
        pred = self.predict(data)
        loss = torch.nn.functional.mse_loss(pred, label)
        return (loss, pred, label)

    def metric_name(self):
        return "metric"

    def compute_metric(self, preds, labels):
        preds = torch.unsqueeze(torch.cat(preds, 0), 1)
        labels = torch.unsqueeze(torch.cat(labels, 0), 1).type(preds.dtype)
        return torch.nn.functional.mse_loss(preds, labels)

    def predict(self, data: dict):
        input = data["inputs"]
        return input * self.a + self.b

    def get_embedding(self, data: dict):
        result = self.predict(data).unsqueeze(-1)
        return torch.cat([result, result], -1)

    # dump embeddings to a file.
    def output_embedding(self, output, context: dict, embeddings):
        embeddings = embeddings.data.cpu().numpy()
        inputs = context["inputs"].squeeze(0)
        for k in range(len(embeddings)):
            output.write(
                str(inputs[k].numpy())
                + " "
                + " ".join([str(embeddings[k][x]) for x in range(len(embeddings[k]))])
                + "\n"
            )


class MockDataset:
    def __init__(self, a=2, b=3, length=64, seed=0):
        np.random.seed(seed)
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.y = (a * self.x + b + np.random.normal(scale=0.1, size=(length,))).astype(
            np.float32
        )

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"inputs": self.x[i], "label": self.y[i]}


def create_mock_dataset(batch_size=8):
    dataset = MockDataset()
    return DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=batch_size
    )


def create_model_optimizer(model=None):
    model = MockModel()
    torch.manual_seed(0)
    return model, torch.optim.SGD(model.parameters(), lr=0.01)


def create_training_args(batch_size=8, num_epochs=1, fp16=False, apex=False):
    args = TrainingArguments()
    temp_dir = tempfile.TemporaryDirectory()
    args.model_dir = temp_dir.name
    args.save_path = temp_dir.name
    args.metric_dir = temp_dir.name
    args.batch_size = batch_size
    args.num_epochs = num_epochs
    args.fp16 = fp16
    args.apex = apex
    return args, temp_dir


def create_trainer(trainer_cls, model=None, fp16=False, apex=False):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{randint(30000, 40000)}"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    args, temp_dir = create_training_args(fp16=fp16, apex=apex)
    return trainer_cls(args), temp_dir


def get_trained_model_without_trainer(batch_size=8, num_epochs=1):
    model, optimizer = create_model_optimizer()
    dataset = create_mock_dataset(batch_size)

    model.train()
    torch.manual_seed(0)

    for epoch in range(num_epochs):
        for step, data in enumerate(dataset):
            optimizer.zero_grad()
            loss, pred, label = model(data)
            loss.backward()
            optimizer.step()

    return model


def get_evaluation_result_without_trainer(model):
    model.eval()

    dataset = create_mock_dataset()

    preds = []
    labels = []
    for step, data in enumerate(dataset):
        loss, pred, label = model(data)
        preds.append(pred)
        labels.append(label)

    metric = model.compute_metric(preds, labels)
    return metric


def get_inference_result_without_trainer(model):
    model.eval()

    dataset = create_mock_dataset()
    temp_dir = tempfile.TemporaryDirectory()

    with open(
        os.path.join(temp_dir.name, f"{PREFIX_EMBEDDING}-0.tsv"), "w+", encoding="utf-8"
    ) as fp:
        for step, data in enumerate(dataset):
            pred = model.get_embedding(data)
            model.output_embedding(fp, data, pred)

        fp.seek(0)
        result = fp.read()

    temp_dir.cleanup()

    return result


def check_trained_model(trained_model, expected_model, equal=True):
    assert equal == torch.allclose(trained_model.a, expected_model.a)
    assert equal == torch.allclose(trained_model.b, expected_model.b)


def check_training_with_trainer(trainer_cls, expected_model):
    trainer, temp_dir = create_trainer(trainer_cls)
    model, optimizer = create_model_optimizer()

    dataset = create_mock_dataset()
    trainer.run(model, dataset, optimizer)
    check_trained_model(trainer.model, expected_model)

    temp_dir.cleanup()


def check_evaluation_with_trainer(trainer_cls, model, expected_result):
    trainer, temp_dir = create_trainer(trainer_cls)
    trainer.args.mode = TrainMode.EVALUATE
    dataset = create_mock_dataset()
    result = trainer.run(model, dataset)
    assert torch.allclose(result.type(expected_result.dtype), expected_result)
    temp_dir.cleanup()


def check_inference_with_trainer(trainer_cls, model, expected_result):
    trainer, temp_dir = create_trainer(trainer_cls)
    trainer.args.mode = TrainMode.INFERENCE
    dataset = create_mock_dataset()
    trainer.run(model, dataset)
    embed_path = os.path.join(
        trainer.args.save_path, f"{PREFIX_EMBEDDING}-{trainer.rank}.tsv"
    )

    with open(embed_path, encoding="utf-8") as fp:
        result = fp.read()
    assert result == expected_result
    temp_dir.cleanup()


def test_training_with_trainer():
    expected_model = get_trained_model_without_trainer()
    check_training_with_trainer(Trainer, expected_model)
    check_training_with_trainer(FP16Trainer, expected_model)
    if hvd_enabled:
        check_training_with_trainer(HVDTrainer, expected_model)
    check_training_with_trainer(DDPTrainer, expected_model)


def test_evaluation_with_trainer():
    model = get_trained_model_without_trainer()
    expected_result = get_evaluation_result_without_trainer(model)
    check_evaluation_with_trainer(Trainer, model, expected_result)
    check_evaluation_with_trainer(FP16Trainer, model, expected_result)
    if hvd_enabled:
        check_evaluation_with_trainer(HVDTrainer, model, expected_result)
    check_evaluation_with_trainer(DDPTrainer, model, expected_result)


def test_inference_with_trainer():
    model = get_trained_model_without_trainer()
    expected_result = get_inference_result_without_trainer(model)
    check_inference_with_trainer(Trainer, model, expected_result)
    check_inference_with_trainer(FP16Trainer, model, expected_result)
    if hvd_enabled:
        check_inference_with_trainer(HVDTrainer, model, expected_result)
    check_inference_with_trainer(DDPTrainer, model, expected_result)


def test_save_rotate_load_checkpoints():
    trainer, temp_dir = create_trainer(Trainer)
    dataset = create_mock_dataset()
    model, optimizer = create_model_optimizer()

    trainer.args.num_epochs = 2
    trainer.args.max_saved_ckpts = 3
    trainer.args.save_ckpt_by_steps = 5
    trainer.run(model, dataset, optimizer)

    # save and rotate checkpoints
    expected_ckpts = [
        os.path.join(temp_dir.name, f"{PREFIX_CHECKPOINT}-{x}.pt")
        for x in ["001-000000", "001-000005", "002-000000"]
    ]
    saved_ckpts = get_sorted_checkpoints(temp_dir.name)
    assert saved_ckpts == expected_ckpts

    expected_model = get_trained_model_without_trainer(num_epochs=2)

    # load specific checkpoint(from 1st epoch)
    trainer._load_checkpoint(ckpt_path=saved_ckpts[0])
    check_trained_model(trainer.model, expected_model, equal=False)

    # load most recent checkpoint(2nd epoch)
    trainer._load_checkpoint()
    check_trained_model(trainer.model, expected_model, equal=True)

    # continous training
    trainer.args.num_epochs = 4
    trainer.run(model, dataset, optimizer)
    expected_ckpts = [
        os.path.join(temp_dir.name, f"{PREFIX_CHECKPOINT}-{x}.pt")
        for x in ["003-000000", "003-000005", "004-000000"]
    ]

    saved_ckpts = get_sorted_checkpoints(temp_dir.name)
    assert saved_ckpts == expected_ckpts

    temp_dir.cleanup()

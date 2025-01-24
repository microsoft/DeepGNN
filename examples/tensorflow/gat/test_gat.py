# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import pytest
import tempfile
import logging
import os
from deepgnn.tf.common.utils import run_commands, get_metrics_from_event_file
from deepgnn.tf.common.test_helper import TestHelper

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def setup_test(main_file):
    tmp_dir = tempfile.TemporaryDirectory()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    mainfile = os.path.join(current_dir, main_file)
    return tmp_dir, tmp_dir.name, tmp_dir.name, mainfile


def get_accuary(model_dir, enable_eager):
    if enable_eager:
        acc = TestHelper.get_tf2_summary_value(
            os.path.join(model_dir, "evaluate/worker_0"), "accuracy"
        )
    else:
        acc = get_metrics_from_event_file(model_dir, "evaluate/worker_0", "accuracy")
    avg_acc = sum(acc) / len(acc)
    return avg_acc


def run_citation_graph_test(
    dataset,
    batch_size,
    hyper_param,
    enable_eager=False,
    run_inference=False,
    desired_acc=None,
    epochs=300,
    attn_drop=0.0,
):
    logger.info(f"run_citation_graph_test -- {dataset.GRAPH_NAME} starts")
    tmp_dir, model_dir, _, mainfile = setup_test("main.py")

    eager_param = " --eager" if enable_eager else ""

    def run_training_job():
        training_cmd = (
            f"python {mainfile} --mode train --seed 123 --model_dir {model_dir} --data_dir {dataset.data_dir()}"
            + eager_param
            + f" --batch_size {batch_size} --learning_rate 0.005 --epochs {epochs}"
            + f" --neighbor_edge_types 0 --attn_drop {attn_drop} --ffd_drop 0.6 {hyper_param}"
            + f" --feature_idx 0 --feature_dim {dataset.FEATURE_DIM}"
            + f" --label_idx 1 --label_dim 1 --num_classes {dataset.NUM_CLASSES} --prefetch_worker_size 1 --log_save_steps 20"
            + " --summary_save_steps 1 --backend snark --converter skip"
        )
        res = run_commands(training_cmd)
        assert res == 0

    def run_eval_job():
        test_file = os.path.join(dataset.data_dir(), "test.nodes")
        eval_cmd = (
            f"python {mainfile} --mode evaluate --seed 123 --model_dir {model_dir} --data_dir {dataset.data_dir()}"
            + eager_param
            + " --batch_size 1000"
            + f" --evaluate_node_files {test_file}"
            + f" --neighbor_edge_types 0 --attn_drop 0.0 --ffd_drop 0.0 {hyper_param}"
            + f" --feature_idx 0 --feature_dim {dataset.FEATURE_DIM}"
            + f" --label_idx 1 --label_dim 1 --num_classes {dataset.NUM_CLASSES} --prefetch_worker_size 1 --log_save_steps 1"
            + " --summary_save_steps 1 --backend snark --converter skip"
        )
        res = run_commands(eval_cmd)
        assert res == 0

    def run_inf_job():
        eval_cmd = (
            f"python {mainfile} --mode inference --seed 123 --model_dir {model_dir} --data_dir {dataset.data_dir()}"
            + eager_param
            + " --batch_size 10"
            + " --inf_min_id 0 --inf_max_id 123"
            + f" --neighbor_edge_types 0 --attn_drop 0.0 --ffd_drop 0.0 {hyper_param}"
            + f" --feature_idx 0 --feature_dim {dataset.FEATURE_DIM}"
            + f" --label_idx 1 --label_dim 1 --num_classes {dataset.NUM_CLASSES} --prefetch_worker_size 1 --log_save_steps 1"
            + " --summary_save_steps 1 --backend snark --converter skip"
        )
        res = run_commands(eval_cmd)
        assert res == 0

    run_training_job()
    run_eval_job()
    acc = get_accuary(model_dir, enable_eager)
    logger.info(
        f"Accuracy ({dataset.GRAPH_NAME} test dataset): {acc}, enable eager: {enable_eager}"
    )

    if run_inference:
        run_inf_job()
    return acc


def test_gat_tf1():
    # fmt: off
    hyper_param = "--head_num 8,1 --l2_coef 0.0005 --hidden_dim 8 --gpu"
    # TF1Trainer
    tmp_dir = tempfile.TemporaryDirectory()
    acc = run_citation_graph_test(CoraFull(output_dir=tmp_dir.name), 140, hyper_param, enable_eager=False, run_inference=True, attn_drop=0.2)
    assert acc > 0.823
    # fmt: on


def test_gat_eager():
    # fmt: off
    hyper_param = "--head_num 8,1 --l2_coef 0.0005 --hidden_dim 8 --gpu"
    # EagerTrainer
    tmp_dir = tempfile.TemporaryDirectory()
    acc = run_citation_graph_test(CoraFull(output_dir=tmp_dir.name), 140, hyper_param, enable_eager=True, run_inference=True, attn_drop=0.2)
    assert acc > 0.823
    # fmt: on


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )

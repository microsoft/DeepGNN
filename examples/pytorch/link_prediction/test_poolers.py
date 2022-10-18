# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import numpy.testing as npt
import os
import sys
import pytest
import random
import argparse
import torch

from deepgnn import get_logger
from deepgnn.graph_engine import (
    FeatureType,
    GraphType,
    BackendType,
    BackendOptions,
    TextFileSampler,
    create_backend,
)
from deepgnn.graph_engine.snark.converter.options import DataConverterType
from examples.pytorch.conftest import MockGraph, load_data  # noqa: F401
from deepgnn.pytorch.common.consts import (
    NODE_FEATURES,
    ENCODER_SEQ,
    ENCODER_MASK,
    ENCODER_TYPES,
    FANOUTS,
    NODE_SRC,
    NODE_DST,
)
from examples.pytorch.link_prediction.consts import (
    ENCODER_LABEL,
    FANOUTS_NAME,
    SIM_TYPE_COSINE,
    SIM_TYPE_MAXFEEDFORWARD,
    SIM_TYPE_RESLAYER,
    SIM_TYPE_MAXRESLAYER,
    SIM_TYPE_SELFATTENTION,
    SIM_TYPE_COSINE_WITH_RNS,
    SIM_TYPE_FEEDFORWARD,
)
from deepgnn.pytorch.encoding import MultiTypeFeatureEncoder
from deepgnn.pytorch.common.dataset import TorchDeepGNNDataset
from model import LinkPredictionModel  # type: ignore
from output_layer import OutputLayer  # type: ignore
from test_model import (  # type: ignore
    train_linkprediction_model_gat,
    train_linkprediction_model_hetgnn,
    lp_mock_graph,
    prepare_params,
    MockBackend,
    prepare_local_twinbert_test_files,
)
from deepgnn.graph_engine.test_adl_reader import IS_ADL_CONFIG_VALID

pytestmark = pytest.mark.skipif(not IS_ADL_CONFIG_VALID, reason="Invalid adl config.")


def get_source_dest_vec(params, config, graph):
    feature_enc = MultiTypeFeatureEncoder(
        FeatureType.INT64, config, ["q", "k", "s"], False
    )

    lp = LinkPredictionModel(
        args=params, feature_type=FeatureType.INT64, feature_enc=feature_enc
    )

    args = argparse.Namespace(
        data_dir="/mock/doesnt/need/physical/path",
        backend=BackendType.CUSTOM,
        graph_type=GraphType.LOCAL,
        converter=DataConverterType.SKIP,
        custom_backendclass=MockBackend,
    )
    backend = create_backend(BackendOptions(args), is_leader=True)
    dataset = TorchDeepGNNDataset(
        sampler_class=TextFileSampler,
        backend=backend,
        query_fn=lp.query,
        prefetch_queue_size=1,
        prefetch_worker_size=1,
        batch_size=32,
        store_name=params.store_name,
        filename=params.filename_pattern,
        shuffle=False,
        drop_last=False,
        worker_index=0,
        num_workers=1,
        epochs=1,
        buffer_size=1024,
    )

    trainloader = torch.utils.data.DataLoader(dataset)
    for i, data in enumerate(trainloader):
        if i == 0:
            x_batch = data[NODE_FEATURES]
            # source nodes
            src_info = lp.get_score(
                {
                    ENCODER_SEQ: x_batch[1][0],
                    ENCODER_MASK: x_batch[1][1],
                    ENCODER_LABEL: x_batch[3],
                    ENCODER_TYPES: params.src_encoders,
                    FANOUTS: params.src_fanouts,
                    FANOUTS_NAME: NODE_SRC,
                }
            )
            # destination nodes
            dst_info = lp.get_score(
                {
                    ENCODER_SEQ: x_batch[2][0],
                    ENCODER_MASK: x_batch[2][1],
                    ENCODER_LABEL: None,
                    ENCODER_TYPES: params.dst_encoders,
                    FANOUTS: params.dst_fanouts,
                    FANOUTS_NAME: NODE_DST,
                }
            )

    return src_info, dst_info


@pytest.fixture(scope="module")
def get_source_dest_vec_gat(train_linkprediction_model_gat):  # noqa: F811
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    src_info, dst_info = get_source_dest_vec(
        train_linkprediction_model_gat["params"],
        train_linkprediction_model_gat["config"],
        train_linkprediction_model_gat["graph"],
    )

    yield {"src": src_info, "dst": dst_info}


@pytest.fixture(scope="module")
def get_source_dest_vec_hetgnn(train_linkprediction_model_hetgnn):  # noqa: F811
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    src_info, dst_info = get_source_dest_vec(
        train_linkprediction_model_hetgnn["params"],
        train_linkprediction_model_hetgnn["config"],
        train_linkprediction_model_hetgnn["graph"],
    )

    yield {"src": src_info, "dst": dst_info}


def do_linkprediction_poolers(test_rootdir, src_info, dst_info):
    results = []

    for i, item in enumerate(
        [
            SIM_TYPE_COSINE,
            SIM_TYPE_COSINE_WITH_RNS,
            SIM_TYPE_FEEDFORWARD,
            SIM_TYPE_MAXFEEDFORWARD,
            SIM_TYPE_RESLAYER,
            SIM_TYPE_MAXRESLAYER,
            SIM_TYPE_SELFATTENTION,
        ]
    ):

        output_layer = OutputLayer(
            input_dim=128,
            featenc_config=os.path.join(
                test_rootdir, "twinbert", "linkprediction.json"
            ),
            sim_type=item,
        )

        scores = output_layer.simpooler(
            src_info[0].float(),
            src_info[1],
            src_info[2],
            dst_info[0].float(),
            dst_info[1],
            dst_info[2],
        )

        get_logger().info(f"comparing scores using {item}...")
        results.append(scores.detach().numpy()[0:10])

    return results


@pytest.mark.flaky(reruns=5)
def test_linkprediction_poolers_gat(
    get_source_dest_vec_gat, prepare_local_twinbert_test_files  # noqa: F811
):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    expected_results = [
        np.array(
            [
                0.53261244,
                0.53252727,
                0.5325831,
                0.5325572,
                0.5330452,
                0.5328803,
                0.5326774,
                0.532783,
                0.53282017,
                0.5329096,
            ]
        ),
        np.array(
            [
                0.51171684,
                0.52309227,
                0.5156336,
                0.5190985,
                0.45392138,
                0.47594112,
                0.50304395,
                0.48894215,
                0.48397413,
                0.4720282,
            ]
        ),
        np.array(
            [
                0.10265648,
                0.10554138,
                0.11915004,
                0.10030035,
                0.10639959,
                0.1330883,
                0.09980031,
                0.11504256,
                0.0993114,
                0.0800672,
            ]
        ),
        np.array(
            [
                -0.00570174,
                -0.00266282,
                -0.01496042,
                -0.00977854,
                -0.00848193,
                -0.00639222,
                -0.02457803,
                -0.00754836,
                -0.00726998,
                -0.01592281,
            ]
        ),
        np.array(
            [
                -0.06188803,
                -0.05852186,
                -0.07559493,
                -0.07554784,
                -0.06622438,
                -0.07241368,
                -0.07695091,
                -0.07141278,
                -0.05849019,
                -0.05835484,
            ]
        ),
        np.array(
            [
                0.03180143,
                0.00130656,
                0.03959438,
                0.01810841,
                0.0106966,
                0.04904991,
                0.01811001,
                0.00093349,
                0.02116175,
                0.01002404,
            ]
        ),
        np.array(
            [
                0.01721961,
                -0.40398604,
                0.2397913,
                0.6257172,
                0.6673481,
                -0.13822797,
                0.09222809,
                0.12950772,
                -0.47402084,
                0.75622725,
            ]
        ),
    ]

    results = do_linkprediction_poolers(
        prepare_local_twinbert_test_files,
        get_source_dest_vec_gat["src"],
        get_source_dest_vec_gat["dst"],
    )

    for i, res in enumerate(results):
        npt.assert_allclose(res, expected_results[i], rtol=1e-4)


@pytest.mark.flaky(reruns=5)
def test_linkprediction_poolers_hetgnn(
    get_source_dest_vec_hetgnn, prepare_local_twinbert_test_files  # noqa: F811
):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    expected_results = [
        np.array(
            [
                0.5367349,
                0.53662753,
                0.53606933,
                0.53662115,
                0.53573495,
                0.53642213,
                0.53615,
                0.535827,
                0.5361027,
                0.5360129,
            ]
        ),
        np.array(
            [
                -0.038907904,
                -0.024566254,
                0.049991798,
                -0.023718711,
                0.09465307,
                0.0028674458,
                0.03921889,
                0.08235607,
                0.0455286,
                0.057531487,
            ]
        ),
        np.array(
            [
                0.051518448,
                -0.059483476,
                0.33157477,
                -0.046736635,
                0.13602942,
                0.15349966,
                0.33470482,
                0.30227128,
                0.36822736,
                0.2925145,
            ]
        ),
        np.array(
            [
                0.17560266,
                -0.1330753,
                0.024855226,
                -0.10239752,
                -0.1412391,
                -0.044689335,
                -0.06855312,
                -0.066341996,
                -0.15280262,
                -0.5014988,
            ]
        ),
        np.array(
            [
                0.09672833,
                0.14576858,
                -0.016803294,
                0.06429329,
                -0.07333496,
                -0.12163665,
                0.034213364,
                0.014596999,
                0.11097679,
                -0.07204538,
            ]
        ),
        np.array(
            [
                -0.12552588,
                0.46151322,
                -0.06319924,
                -0.10635249,
                -0.03906784,
                0.11754773,
                -0.15851876,
                -0.013094959,
                0.060184132,
                0.0718291,
            ]
        ),
        np.array(
            [
                -0.61702114,
                -0.10309171,
                0.34793675,
                0.3406144,
                1.0324205,
                -0.20363998,
                0.10659025,
                0.3297817,
                0.02773913,
                0.4304406,
            ]
        ),
    ]

    results = do_linkprediction_poolers(
        prepare_local_twinbert_test_files,
        get_source_dest_vec_hetgnn["src"],
        get_source_dest_vec_hetgnn["dst"],
    )

    for i, res in enumerate(results):
        npt.assert_allclose(res, expected_results[i], rtol=1e-4)


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "--junitxml", os.environ["XML_OUTPUT_FILE"], *sys.argv[1:]]
        )
    )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest
import tempfile
import os
import torch
import urllib.request
import zipfile

from deepgnn.pytorch.common.consts import (
    EMBEDDING_TYPE,
    TRILETTER,
    ENCODER_SEQ,
    ENCODER_MASK,
)
from deepgnn.graph_engine import FeatureType
from deepgnn.pytorch.encoding.feature_encoder import (
    TwinBERTEncoder,
    TwinBERTFeatureEncoder,
)


@pytest.fixture(scope="module")
def prepare_local_test_files():
    name = "twinbert.zip"
    working_dir = tempfile.TemporaryDirectory()
    zip_file = os.path.join(working_dir.name, name)
    urllib.request.urlretrieve(
        f"https://deepgraphpub.blob.core.windows.net/public/testdata/{name}", zip_file
    )
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(working_dir.name)

    yield working_dir.name
    working_dir.cleanup()


c_test_strings = ["", "\0\0", "hello world", "hello world\0\0"]

c_triletter_input_features = [
    [2754, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [2754, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1121, 1450, 766, 1696, 2878, 684, 685, 1164, 1900, 749, 0, 0, 0, 0, 0, 1, 1, 0],
    [1121, 1450, 766, 1696, 2878, 684, 685, 1164, 1900, 749, 0, 0, 0, 0, 0, 1, 1, 0],
]

c_wordpiece_input_features = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [7592, 2088, 0, 1, 1, 0],
    [7592, 2088, 0, 1, 1, 0],
]

c_triletter_expected_tensors = [
    [
        torch.tensor(
            [[-0.3108, -0.2532, -0.1678, -0.1977, -0.5735, -0.1219, -0.3587, -0.0717]]
            * 3
        ),
        torch.tensor(
            [[0.2644, 0.1836, -0.1318, -0.8181, 0.4117, 0.2877, -0.2865, 0.0205]] * 3
        ),
    ],
    [
        torch.tensor(
            [[-0.3108, -0.2532, -0.1678, -0.1977, -0.5735, -0.1219, -0.3587, -0.0717]]
            * 3
        ),
        torch.tensor(
            [[0.3555, 0.1557, -0.1637, -0.8648, 0.5441, 0.2301, -0.2819, 0.0645]] * 3
        ),
    ],
]

c_wordpiece_expected_tensors = [
    [
        torch.tensor(
            [[-0.0702, 0.1995, 0.7667, -0.0750, -0.0298, -0.5660, -0.3190, 0.0592]] * 3
        ),
        torch.tensor(
            [[0.2627, -0.2284, 0.6007, -0.4606, -0.5257, 0.2273, 0.3187, -0.6435]] * 3
        ),
    ],
    [
        torch.tensor(
            [[-0.0702, 0.1995, 0.7667, -0.0750, -0.0298, -0.5660, -0.3190, 0.0592]] * 3
        ),
        torch.tensor(
            [[0.2905, -0.1703, 0.5723, -0.4159, -0.5098, 0.1325, 0.3189, -0.6137]] * 3
        ),
    ],
]


def get_twinbert_encoder(config_file, feature_type=FeatureType.BINARY):
    torch.manual_seed(0)
    config = TwinBERTEncoder.init_config_from_file(config_file)
    return TwinBERTFeatureEncoder(feature_type, config, pooler_count=2)


def verify_twinbert_encoder_simple_input(config_file, feature_type, pooler_index=0):
    twinbert = get_twinbert_encoder(config_file, feature_type)
    twinbert.eval()

    tokenize_with_triletter = twinbert.config[EMBEDDING_TYPE] == TRILETTER
    expected_tensors = (
        c_triletter_expected_tensors
        if tokenize_with_triletter
        else c_wordpiece_expected_tensors
    )[pooler_index]

    test_input = (
        c_test_strings
        if feature_type == FeatureType.BINARY
        else (
            c_triletter_input_features
            if tokenize_with_triletter
            else c_wordpiece_input_features
        )
    )

    def verify_encode(input, expect):
        context = {}
        if feature_type == FeatureType.BINARY:
            context["feature"] = np.array([bytearray(input, encoding="utf-8")] * 3)
        elif feature_type == FeatureType.INT64:
            context["feature"] = np.array([input] * 3)

        twinbert.transform(context)

        data = context["feature"]
        assert ENCODER_SEQ in data and ENCODER_MASK in data

        if tokenize_with_triletter:
            assert data[ENCODER_SEQ].shape[-1] == 15  # 3 * 5
        else:
            assert data[ENCODER_SEQ].shape[-1] == 3  # 3 * 1

        for key in data:
            data[key] = torch.from_numpy(data[key])

        with torch.no_grad():
            twinbert.forward(context, pooler_index)
            assert torch.allclose(context["feature"], expect, rtol=1e-02)

    verify_encode(test_input[0], expected_tensors[0])
    verify_encode(test_input[1], expected_tensors[0])
    verify_encode(test_input[2], expected_tensors[1])
    verify_encode(test_input[3], expected_tensors[1])


def verify_twinbert_encoder_complex_input(config_file):
    twinbert = get_twinbert_encoder(config_file)
    use_triletter = twinbert.config[EMBEDDING_TYPE] == TRILETTER
    expected_tensors = (
        c_triletter_expected_tensors if use_triletter else c_wordpiece_expected_tensors
    )[0]

    context = {}
    context["feature"] = np.array([bytearray(c_test_strings[0], encoding="utf-8")] * 3)
    context["type1"] = {}
    context["type1"]["feature1"] = np.array(
        [bytearray(c_test_strings[0], encoding="utf-8")] * 3
    )
    context["type1"]["feature2"] = np.array(
        [bytearray(c_test_strings[1], encoding="utf-8")] * 3
    )
    context["type2"] = {}
    context["type2"]["feature1"] = np.array(
        [bytearray(c_test_strings[2], encoding="utf-8")] * 3
    )
    context["type2"]["feature2"] = np.array(
        [bytearray(c_test_strings[3], encoding="utf-8")] * 3
    )
    expect = {}
    expect["feature"] = expected_tensors[0]
    expect["type1"] = {}
    expect["type1"]["feature1"] = expected_tensors[0]
    expect["type1"]["feature2"] = expected_tensors[0]
    expect["type2"] = {}
    expect["type2"]["feature1"] = expected_tensors[1]
    expect["type2"]["feature2"] = expected_tensors[1]

    twinbert.transform(context)

    def totensor(context):
        for key in context:
            data = context[key]
            if isinstance(data, np.ndarray):
                context[key] = torch.from_numpy(data)
            else:
                totensor(context[key])

    def verify_encode(context, expect):
        for key in context:
            data = context[key]
            if isinstance(data, torch.Tensor):
                assert torch.allclose(data, expect[key], rtol=1e-02)
            else:
                verify_encode(context[key], expect[key])

    totensor(context)

    with torch.no_grad():
        twinbert.eval()
        twinbert.forward(context)
        verify_encode(context, expect)


def test_twinbert_encoder_simple_input_string_features_pooler_0(
    prepare_local_test_files,
):
    verify_twinbert_encoder_simple_input(
        os.path.join(prepare_local_test_files, "twinbert", "twinbert_triletter.json"),
        FeatureType.BINARY,
    )
    verify_twinbert_encoder_simple_input(
        os.path.join(prepare_local_test_files, "twinbert", "twinbert_wordpiece.json"),
        FeatureType.BINARY,
    )


def test_twinbert_encoder_simple_input_string_features_pooler_1(
    prepare_local_test_files,
):
    verify_twinbert_encoder_simple_input(
        os.path.join(prepare_local_test_files, "twinbert", "twinbert_triletter.json"),
        FeatureType.BINARY,
        1,
    )
    verify_twinbert_encoder_simple_input(
        os.path.join(prepare_local_test_files, "twinbert", "twinbert_wordpiece.json"),
        FeatureType.BINARY,
        1,
    )


def test_twinbert_encoder_simple_input_int_features(prepare_local_test_files):
    verify_twinbert_encoder_simple_input(
        os.path.join(prepare_local_test_files, "twinbert", "twinbert_triletter.json"),
        FeatureType.INT64,
    )
    verify_twinbert_encoder_simple_input(
        os.path.join(prepare_local_test_files, "twinbert", "twinbert_wordpiece.json"),
        FeatureType.INT64,
    )


def test_twinbert_encoder_complex_input(prepare_local_test_files):
    verify_twinbert_encoder_complex_input(
        os.path.join(prepare_local_test_files, "twinbert", "twinbert_triletter.json")
    )
    verify_twinbert_encoder_complex_input(
        os.path.join(prepare_local_test_files, "twinbert", "twinbert_wordpiece.json")
    )

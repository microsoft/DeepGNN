# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from deepgnn.pytorch.common.utils import get_store_name_and_path


def test_get_store_name_and_path():
    input_path = "adl://grocery.azuredatalakestore.net/local/test/"
    store_name, relative_path = get_store_name_and_path(input_path)
    assert store_name == "grocery"
    assert relative_path == "/local/test/"

    input_path = "/local/test1/"
    store_name, relative_path = get_store_name_and_path(input_path)
    assert store_name == ""
    assert relative_path == "/local/test1/"

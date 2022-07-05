# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import os
import tempfile
import numpy as np
from deepgnn.graph_engine._adl_reader import (
    TextFileSplitIterator,
    TextFileIterator,
    AdlCredentialParser,
)
from azure.datalake.store import core, lib

IS_ADL_CONFIG_VALID = False
try:
    spn = AdlCredentialParser.read_credentials()
    principal_token = lib.auth(
        tenant_id=spn["TENANT_ID"],
        client_secret=spn["CLIENT_SECRET"],
        client_id=spn["CLIENT_ID"],
    )
    adl = core.AzureDLFileSystem(token=principal_token, store_name="snrgnndls")
    # submit a ls request to test if the core-site is correct,
    # if no, skip all the test cases.
    adl.ls()
    IS_ADL_CONFIG_VALID = True
except:  # noqa: E722
    pytestmark = pytest.mark.skip(reason="No valid adl config found.")


def test_adl_file_iterator():
    # sample files:
    # /test_twinbert/test01.txt
    client = TextFileIterator(
        "/test_twinbert/test01.txt", "snrgnndls", batch_size=3, epochs=1
    )
    res = []
    for sp in client:
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9"])
    np.testing.assert_array_equal(res, exp)

    # sample files:
    # /test_twinbert/test01.txt
    # /test_twinbert/test02.txt
    client = TextFileIterator(
        "/test_twinbert/*.txt", "snrgnndls", batch_size=3, epochs=1
    )
    res = []
    for sp in client:
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "20",
            "30",
            "40",
            "50",
            "60",
            "70",
            "80",
            "90",
        ]
    )
    np.testing.assert_array_equal(res, exp)


@pytest.fixture(scope="module")
def prepare_local_test_files():
    working_dir = tempfile.mkdtemp()
    file_count = 3
    file_length = 5
    for i in range(file_count):
        fname = os.path.join(working_dir, f"test_file{i}")
        with open(fname, "w") as w:
            [w.write(f"{i + 1}{x + 1}\n") for x in range(file_length)]

    return working_dir


def test_local_adl_file_iterator(prepare_local_test_files):
    # sample files:
    path = prepare_local_test_files
    client = TextFileIterator(f"{path}/test_file*", "", None, 3, 1)
    res = []
    for _, sp in enumerate(client):
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(
        [
            "11",
            "12",
            "13",
            "14",
            "15",
            "21",
            "22",
            "23",
            "24",
            "25",
            "31",
            "32",
            "33",
            "34",
            "35",
        ]
    )
    np.testing.assert_array_equal(res, exp)


def test_local_adl_file_iterator_drop_last(prepare_local_test_files):
    # sample files:
    path = prepare_local_test_files
    client = TextFileIterator(f"{path}/test_file*", "", None, 4, 1, drop_last=True)
    res = []
    for _, sp in enumerate(client):
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(
        ["11", "12", "13", "14", "15", "21", "22", "23", "24", "25", "31", "32"]
    )
    np.testing.assert_array_equal(res, exp)


def test_local_adl_file_iterator_multi_epoch(prepare_local_test_files):
    # sample files:
    path = prepare_local_test_files
    client = TextFileIterator(f"{path}/test_file*", "", None, 3, 2)
    res = []
    for _, sp in enumerate(client):
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(
        [
            "11",
            "12",
            "13",
            "14",
            "15",
            "21",
            "22",
            "23",
            "24",
            "25",
            "31",
            "32",
            "33",
            "34",
            "35",
            "11",
            "12",
            "13",
            "14",
            "15",
            "21",
            "22",
            "23",
            "24",
            "25",
            "31",
            "32",
            "33",
            "34",
            "35",
        ]
    )
    np.testing.assert_array_equal(res, exp)


def test_local_adl_file_iterator_multi_epoch_drop_last(prepare_local_test_files):
    # sample files:
    path = prepare_local_test_files
    client = TextFileIterator(f"{path}/test_file*", "", None, 4, 2, drop_last=True)
    res = []
    for _, sp in enumerate(client):
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(
        [
            "11",
            "12",
            "13",
            "14",
            "15",
            "21",
            "22",
            "23",
            "24",
            "25",
            "31",
            "32",
            "33",
            "34",
            "35",
            "11",
            "12",
            "13",
            "14",
            "15",
            "21",
            "22",
            "23",
            "24",
            "25",
            "31",
            "32",
            "33",
        ]
    )
    np.testing.assert_array_equal(res, exp)


def test_adl_file_iterator_multi_epoch():
    # sample files:
    # /test_twinbert/test01.txt
    client = TextFileIterator(
        "/test_twinbert/test01.txt", "snrgnndls", batch_size=3, epochs=3
    )
    res = []
    for _, sp in enumerate(client):
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]
    )
    np.testing.assert_array_equal(res, exp)

    # sample files:
    # /test_twinbert/test01.txt
    # /test_twinbert/test02.txt
    client = TextFileIterator(
        "/test_twinbert/*.txt", "snrgnndls", batch_size=3, epochs=2
    )
    res = []
    for _, sp in enumerate(client):
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "20",
            "30",
            "40",
            "50",
            "60",
            "70",
            "80",
            "90",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "20",
            "30",
            "40",
            "50",
            "60",
            "70",
            "80",
            "90",
        ]
    )
    np.testing.assert_array_equal(res, exp)


def test_local_adl_file_iterator_cross_file(prepare_local_test_files):
    # sample files:
    path = prepare_local_test_files
    client = TextFileIterator(f"{path}/test_file*", "", batch_size=4, epochs=1)
    # this is the cross file sample
    exp = np.array(["24", "25", "31", "32"])
    # this is the end of the last file
    exp_end = np.array(["33", "34", "35"])

    index = 0
    for _, sp in enumerate(client):
        if index == 2:
            np.testing.assert_array_equal(sp, exp)
        if index == 3:
            np.testing.assert_array_equal(sp, exp_end)
        index += 1


def test_adl_file_iterator_cross_file():
    # sample files:
    # /test_twinbert/test01.txt contents: 1\n2\n3\n4\n5\n6\n7\n8\n9
    # /test_twinbert/test02.txt contents: 1\n2\n3\n4\n5\n6\n7\n8\n9
    client = TextFileIterator(
        "/test_twinbert/*.txt", "snrgnndls", batch_size=4, epochs=1
    )
    # this is the cross file sample
    exp = np.array(["9", "10", "20", "30"])
    # this is the end of the last file
    exp_end = np.array(["80", "90"])

    index = 0
    for _, sp in enumerate(client):
        if index == 2:
            np.testing.assert_array_equal(sp, exp)
        if index == 4:
            np.testing.assert_array_equal(sp, exp_end)
        index += 1


def test_adl_file_iterator_single_line_cross_block():
    client = iter(
        TextFileIterator(
            "/test_twinbert/folder01/*.json",
            "snrgnndls",
            batch_size=1,
            epochs=1,
            read_block_in_M=0.001,
        )
    )

    exp = '{"node_weight": 1, "uint64_feature": {}, "float_feature": {"0": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], "1": [-0.08551430451309466, -0.08837446087060667, -0.11277995401687406, -0.1719089798201273, -0.07660711300241735, -0.1002706699246255, -0.07512250933978885, -0.11488760015233838, -0.12119106419999719, -0.09936040313660584, 0.0, -0.16990769507399037, -0.04277121835301988, -0.11227260022971815, -0.07601667128230763, -0.1151857054475933, -0.10306549513114831, -0.11196713964718696, -0.14349532776688012, -0.09751538711916201, -0.0874742513123269, -0.1456625048671852, -0.12344163341514039, -0.12418329434086417, -0.09763168290081542, -0.11966855805723346, -0.11607566789018665, -0.07345559012783368, -0.06671734788911636, -0.0873449158087815, -0.17965071163099836, -0.14470299396502243, -0.16059202438451922, -0.15816800997394034, -0.14772183258730107, -0.434953899704262, -0.16168302698696624, -0.1555598998030772, -0.15260625849005044, -0.1396491879758461, -0.12810062023728433, -0.1539000661318612, -0.15934743849874022, -0.15458104173285525, -0.14661625520113497, -0.14494341801198307, -0.1567570459426842, -0.13989759738159854, -0.14936615766369898, -0.1481148513618764]}, "edge": [{"src_id": 0, "weight": 1, "uint64_feature": {}, "float_feature": {}, "dst_id": 372, "edge_type": 0, "binary_feature": {}}, {"src_id": 0, "weight": 1, "uint64_feature": {}, "float_feature": {}, "dst_id": 1101, "edge_type": 0, "binary_feature": {}}, {"src_id": 0, "weight": 1, "uint64_feature": {}, "float_feature": {}, "dst_id": 766, "edge_type": 0, "binary_feature": {}}], "node_type": 0, "node_id": 0, "binary_feature": {}}'
    sp = next(client)
    assert sp[0] == exp


def test_adl_file_iterator_reset():
    client = TextFileIterator(
        "/test_twinbert/*.txt", "snrgnndls", batch_size=3, epochs=1
    )
    res1 = []
    res2 = []

    for sp in client:
        res1.append(sp)

    client.reset()
    for sp in client:
        res2.append(sp)

    np.testing.assert_array_equal(res1, res2)


def test_adl_file_iterator_join():
    client = TextFileIterator(
        "/test_twinbert/*.json",
        "snrgnndls",
        batch_size=1,
        epochs=1,
        read_block_in_M=0.01,
        buffer_queue_size=1,
        thread_count=3,
    )

    for sp in client:
        break

    client.join()

    # second join will not trigger exception.
    client.join()

    # make sure it will not blocked.
    assert True


def get_newline_pos(offset_start, input_path):
    extra_length = 100
    if offset_start != 0:
        with open(input_path, "rb") as fin:
            fin.seek(offset_start)
            bstring = fin.read(extra_length)
            found = False
            while (not found) and (len(bstring) != 0):
                try:
                    pos = bstring.index(b"\n")
                    found = True
                    offset_start += pos
                    break
                except ValueError:
                    offset_start += len(bstring)
                    bstring = fin.read(extra_length)

    return offset_start


def test_local_adl_file_split_iterator(prepare_local_test_files):
    # sample files:
    path = prepare_local_test_files
    start = get_newline_pos(0, f"{path}/test_file1")
    end = get_newline_pos(6, f"{path}/test_file1") + 1
    assert start == 0 and end == 9

    client = TextFileSplitIterator(
        filename=f"{path}/test_file1",
        store_name="",
        batch_size=3,
        worker_offset=start,
        total_read_length=(end - start),
    )
    res = []
    for _, sp in enumerate(client):
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(["21", "22", "23"])
    np.testing.assert_array_equal(res, exp)

    start = get_newline_pos(10, f"{path}/test_file1")
    end = get_newline_pos(15, f"{path}/test_file1") + 1
    assert start == 11 and end == 16

    client = TextFileSplitIterator(
        filename=f"{path}/test_file1",
        store_name="",
        batch_size=3,
        worker_offset=start,
        total_read_length=(end - start),
    )
    res = []
    for _, sp in enumerate(client):
        res.append(sp)

    res = np.concatenate(res)
    exp = np.array(["25"])
    np.testing.assert_array_equal(res, exp)

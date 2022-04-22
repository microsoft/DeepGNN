import os
from deepgnn.pytorch.encoding.twinbert.tokenization import (
    StdBertTokenizer,
    TriLetterTokenizer,
)
import urllib.request
import zipfile
import tempfile
import pytest


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


def test_stdberttokenizer(prepare_local_test_files):
    sentence = "hello world"
    tokenizer = StdBertTokenizer(
        os.path.join(prepare_local_test_files, "twinbert", "uncased_eng_vocab.tsv")
    )
    seq, mask = tokenizer.extract_from_sentence(sentence)
    assert seq == [7592, 2088, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert mask == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_trilettertokenizer(prepare_local_test_files):
    sentence = "hello world"
    tokenizer = TriLetterTokenizer(
        os.path.join(prepare_local_test_files, "twinbert", "l3g.txt")
    )
    seq, mask = tokenizer.extract_from_sentence(
        sentence, max_seq_len=5, max_n_letters=3
    )
    assert seq == [1121, 1450, 766, 684, 685, 1164, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert mask == [1, 1, 0, 0, 0]

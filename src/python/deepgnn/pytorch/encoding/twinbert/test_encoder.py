import torch
import random
import os
import numpy as np
import urllib.request
import zipfile
import pytest
import tempfile
from deepgnn.pytorch.encoding.twinbert.tokenization import (
    TriLetterTokenizer,
    StdBertTokenizer,
)
from deepgnn.pytorch.encoding.twinbert.encoder import TwinBERTEncoder


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


def get_model(
    config_file,
    enable_gpu=False,
    enable_fp16=False,
    transformer_kernel=False,
    init_ckpt_file=None,
):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = TwinBERTEncoder.init_config_from_file(config_file)

    # Update config
    config["enable_fp16"] = enable_fp16
    config["deepspeed"]["transformer_kernel"] = transformer_kernel

    model = TwinBERTEncoder(config, init_ckpt_file)
    model.eval()

    if enable_gpu:
        torch.cuda.set_device(0)
        model = model.cuda()
        if enable_fp16:
            from apex import amp  # type: ignore

            model = amp.initialize(model, None, enabled=True, opt_level="O2")
    return model


def validate_embedding(
    model,
    tokenizer,
    infile,
    embfile,
    enable_gpu=False,
    max_seq_len=32,
    max_n_letters=20,
):
    expect_emb = []
    with open(embfile, "r", encoding="utf-8") as femb:
        for i, emb in enumerate(femb):
            tokens = emb.split("\t")
            emb = np.fromstring(tokens[1], sep=" ")
            expect_emb.append(emb)

    with open(infile, "r", encoding="utf-8") as fin:
        for i, query in enumerate(fin):
            if i > 10:
                break
            query = query.strip("\r\n")
            if isinstance(tokenizer, StdBertTokenizer):
                qseq, qmask = tokenizer.extract_from_sentence(
                    query, max_seq_len=max_seq_len
                )
            else:
                qseq, qmask = tokenizer.extract_from_sentence(
                    query, max_seq_len=max_seq_len, max_n_letters=max_n_letters
                )
            qseq = torch.LongTensor(qseq * 1).view(1, -1)
            qmask = torch.LongTensor(qmask * 1).view(1, -1)
            if enable_gpu:
                qseq, qmask = qseq.cuda(), qmask.cuda()
            qvec = model.forward(qseq, qmask)
            actual_emb = qvec[0].cpu().detach().numpy()
            assert np.allclose(actual_emb, expect_emb[i], atol=1e-02)


def validate_twinbert_with_triletter(
    dir_name,
    infile,
    embfile,
    enable_gpu=False,
    enable_fp16=False,
    transformer_kernel=False,
):
    model = get_model(
        config_file=os.path.join(dir_name, "twinbert", "twinbert_triletter.json"),
        enable_gpu=enable_gpu,
        enable_fp16=enable_fp16,
        transformer_kernel=transformer_kernel,
    )

    tokenizer = TriLetterTokenizer(os.path.join(dir_name, "twinbert", "l3g.txt"))

    validate_embedding(
        model=model,
        tokenizer=tokenizer,
        infile=infile,
        embfile=embfile,
        enable_gpu=enable_gpu,
        max_seq_len=model.config["max_seq_len"],
        max_n_letters=model.config["triletter_max_letters_in_word"],
    )


def validate_twinbert_with_wordpiece(
    dir_name,
    infile,
    embfile,
    enable_gpu=False,
    enable_fp16=False,
    transformer_kernel=False,
):
    model = get_model(
        config_file=os.path.join(dir_name, "twinbert", "twinbert_wordpiece.json"),
        enable_gpu=enable_gpu,
        enable_fp16=enable_fp16,
        transformer_kernel=transformer_kernel,
    )

    tokenizer = StdBertTokenizer(
        os.path.join(dir_name, "twinbert", "uncased_eng_vocab.tsv")
    )

    validate_embedding(
        model=model,
        tokenizer=tokenizer,
        infile=infile,
        embfile=embfile,
        enable_gpu=enable_gpu,
        max_seq_len=model.config["max_seq_len"],
    )


def test_twinbert_with_triletter(prepare_local_test_files):
    infile = os.path.join(prepare_local_test_files, "twinbert", "infile.tsv")
    emb_prefix = os.path.join(
        prepare_local_test_files, "twinbert", "encode_result", "emb_triletter_"
    )
    validate_twinbert_with_triletter(
        prepare_local_test_files, infile, emb_prefix + "std.tsv"
    )


def test_twinbert_with_wordpiece(prepare_local_test_files):
    infile = os.path.join(prepare_local_test_files, "twinbert", "infile.tsv")
    emb_prefix = os.path.join(
        prepare_local_test_files, "twinbert", "encode_result", "emb_wordpiece_"
    )
    validate_twinbert_with_wordpiece(
        prepare_local_test_files, infile, emb_prefix + "std.tsv"
    )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common feature encoders."""
import argparse
import itertools
import operator
import numpy as np
import torch
import os
from typing import Optional, Union, List, Tuple

from deepgnn import get_logger
from .twinbert import TwinBERTEncoder, TriLetterTokenizer, StdBertTokenizer
from deepgnn.graph_engine import FeatureType
from deepgnn.pytorch.common.consts import (
    DOWNSCALE,
    MAX_SEQ_LEN,
    META_DIR,
    VOCAB_FILE,
    EMBEDDING_TYPE,
    TRILETTER,
    TRILETTER_MAX_LETTERS_IN_WORD,
    MAX_SENT_CHARS,
    ENCODER_SEQ,
    ENCODER_MASK,
    OUTPUT,
    TERM_TENSOR,
    FEATURE_ENCODER_STR,
    ENCODER_TYPES,
    EMBS_LIST,
)
from deepgnn.pytorch.common.utils import get_feature_type


class FeatureEncoder(torch.nn.Module):
    """Encoder for raw feature of graph nodes."""

    def __init__(self, feature_type: FeatureType, feature_dim: int, embed_dim: int):
        """Initialize feature encoder.

        Args:
            feature_type: type of raw features.
            feature_dim: dimension of raw features.
            embed_dim: dimension of encoder output.
        """
        super(FeatureEncoder, self).__init__()
        self.feature_type = feature_type
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

    def transform(self, context: dict) -> torch.Tensor:
        """Perform necessary data transformation, e.g. tokenization for text data.

        Result is saved in the context, which will be converted to tensor for
        forward execution. This function will be invoked by prefetch. Args:
            context: nested numpy array dictionary.
        """
        raise NotImplementedError()

    def forward(self, context: dict) -> torch.Tensor:
        """Encode context to get feature embedding."""
        raise NotImplementedError()


class TwinBERTFeatureEncoder(FeatureEncoder):
    """Wrapper for TwinBERTEncoder."""

    def __init__(self, feature_type: FeatureType, config: dict, pooler_count: int = 1):
        """Initialize TwinBERT encoder.

        Args:
            config_file: the file path of the twinber config file.
            config: defined parameters used in twinbert encoder.
        """
        assert config is not None
        self.config = config
        self.tokenize_func = None

        super(TwinBERTFeatureEncoder, self).__init__(
            feature_type,
            self.get_feature_dim(feature_type, config),
            config[DOWNSCALE] if config[DOWNSCALE] > 0 else config["hidden_size"],
        )

        self.max_seq_len = self.config[MAX_SEQ_LEN]
        self.encoder = TwinBERTEncoder(config, pooler_count=pooler_count)

    def _init_tokenize_func(self):
        vocab_file = os.path.join(self.config[META_DIR], self.config[VOCAB_FILE])
        get_logger().info(f"[TwinBERTFeatureEncoder] vocab:{vocab_file}")

        if self.config[EMBEDDING_TYPE] == TRILETTER:
            tokenizer = TriLetterTokenizer(vocab_file)
            self.max_letters_in_word = self.config[TRILETTER_MAX_LETTERS_IN_WORD]
        else:
            tokenizer = StdBertTokenizer(vocab_file)
        self.tokenize_func = tokenizer.extract_from_sentence

    @classmethod
    def get_feature_dim(cls, feature_type: FeatureType, config: dict):
        """Extract feature dimensions."""
        max_seq_len = config[MAX_SEQ_LEN]
        if feature_type == FeatureType.BINARY:
            return config[MAX_SENT_CHARS]
        if feature_type == FeatureType.INT64:
            if config[EMBEDDING_TYPE] == TRILETTER:
                return max_seq_len * (config[TRILETTER_MAX_LETTERS_IN_WORD] + 1)
            else:
                return max_seq_len * 2
        raise RuntimeError("Raw feature dimension is underdetermined.")

    def _tokenize_single_sentence(self, sentence: str) -> torch.Tensor:
        if hasattr(self, "max_letters_in_word"):
            return self.tokenize_func(
                sentence, self.max_seq_len, self.max_letters_in_word
            )  # type: ignore
        return self.tokenize_func(sentence, self.max_seq_len)  # type: ignore

    def _tokenize(self, context: dict):
        """Tokenize binary(string) feature recursively to get sequence ids and mask."""
        # lazy init tokenizer function
        if self.tokenize_func is None:
            self._init_tokenize_func()

        for key in context:
            data = context[key]
            if isinstance(data, dict):
                self._tokenize(data)
            elif isinstance(data, np.ndarray) and (
                data.dtype == np.uint8 or data.dtype == np.int8
            ):
                sentence_len = data.shape[-1]
                sentence_cnt = list(
                    itertools.accumulate(data.shape[:-1], operator.mul)
                )[-1]
                encoder_seq = []
                encoder_mask = []
                if sentence_len != 0:
                    flatten = np.copy(np.reshape(data, [-1, sentence_len]))

                    for sentence in flatten:
                        sentence = (
                            bytearray(sentence)
                            .decode(encoding="utf-8")
                            .strip("\0 \t\n")
                        )
                        seq, mask = self._tokenize_single_sentence(sentence)
                        encoder_seq.append(seq)
                        encoder_mask.append(mask)
                else:
                    seq, mask = self._tokenize_single_sentence("")
                    encoder_seq, encoder_mask = (
                        [seq] * sentence_cnt,
                        [mask] * sentence_cnt,
                    )
                np_seq = np.asarray(encoder_seq)
                np_mask = np.asarray(encoder_mask)
                context[key] = {}
                context[key][ENCODER_SEQ] = np.reshape(
                    np_seq, list(data.shape[:-1]) + list(np_seq.shape[1:])
                )
                context[key][ENCODER_MASK] = np.reshape(
                    np_mask, list(data.shape[:-1]) + list(np_mask.shape[1:])
                )

    def _extract_sequence_id_and_mask(self, context: dict):
        # If tokenization has been done offline, raw feature would be
        # corresponding sequence ids and mask. Here just need to extract them from
        # the combined vector.
        for key in context:
            data = context[key]
            if isinstance(data, dict):
                self._extract_sequence_id_and_mask(data)
            elif isinstance(data, np.ndarray) and (
                data.dtype == np.uint64 or data.dtype == np.int64
            ):
                context[key] = {}
                assert data.shape[-1] >= self.max_seq_len
                encoder_seq = data[:, : -self.max_seq_len]
                encoder_mask = data[:, -self.max_seq_len :]
                context[key][ENCODER_SEQ] = np.reshape(
                    encoder_seq, list(data.shape[:-1]) + list(encoder_seq.shape[1:])
                )
                context[key][ENCODER_MASK] = np.reshape(
                    encoder_mask, list(data.shape[:-1]) + list(encoder_mask.shape[1:])
                )

    def transform(self, context: dict):
        """Transform binary or int64 features."""
        if self.feature_type == FeatureType.BINARY:
            self._tokenize(context)
        elif self.feature_type == FeatureType.INT64:
            self._extract_sequence_id_and_mask(context)
        else:
            raise RuntimeError(
                f"Raw feature with type {self.feature_type} is not supported."
            )

    def forward(
        self, context: dict, pooler_index: int = 0, output_encoded_layer: bool = False
    ):
        """Encode context recursively to get binary(string) feature embedding."""
        for key in context:
            data = context[key]
            if isinstance(data, dict):
                if ENCODER_SEQ in data and ENCODER_MASK in data:
                    seq = data[ENCODER_SEQ]
                    mask = data[ENCODER_MASK]
                    new_seq = seq.view(-1, seq.shape[-1]).type(torch.int64)
                    new_mask = mask.view(-1, mask.shape[-1]).type(torch.int64)
                    output_value = self.encoder.forward(
                        new_seq, new_mask, pooler_index, output_encoded_layer
                    )
                    # encoder_seq and encoder_mask is not needed anymore.
                    if output_encoded_layer:
                        embed = output_value[0]
                        encoded_layer = output_value[1]
                        context[key] = {
                            OUTPUT: embed.view(
                                tuple(list(seq.shape)[:-1] + [embed.shape[-1]])
                            ),
                            TERM_TENSOR: encoded_layer,
                        }
                    else:
                        context[key] = output_value.view(
                            tuple(list(seq.shape)[:-1] + [output_value.shape[-1]])
                        )
                else:
                    self.forward(context[key], pooler_index)


class MultiTypeFeatureEncoder(FeatureEncoder):
    """Encoder for multiple node types.

    This encoder contains a list of TwinBERTFeatureEncoders, if the
    share_encoder parameter is False, each node type has its own TwinBERTFeatureEncoders instance,
    otherwise they share the same encoder but the dedicated pooler.
    """

    def __init__(
        self,
        feature_type: FeatureType,
        config: dict,
        encoder_types: List[str],
        share_encoder: bool = False,
    ):
        """Initialize MultiType feature encoder."""
        super(MultiTypeFeatureEncoder, self).__init__(
            feature_type=feature_type,
            feature_dim=TwinBERTFeatureEncoder.get_feature_dim(feature_type, config),
            embed_dim=config[DOWNSCALE],
        )

        self.share_encoder = share_encoder
        self.encoder_types = encoder_types

        if share_encoder:
            self.add_module(
                FEATURE_ENCODER_STR,
                TwinBERTFeatureEncoder(
                    feature_type=self.feature_type,
                    config=config,
                    pooler_count=len(self.encoder_types),
                ),
            )
        else:
            # e.g. encoder_types = ['q', 'k', 's']
            for encoder_type in self.encoder_types:
                self.add_module(
                    encoder_type + FEATURE_ENCODER_STR,
                    TwinBERTFeatureEncoder(
                        feature_type=self.feature_type, config=config, pooler_count=1
                    ),
                )

    def get_feature_encoder(self, encoder_type: str) -> TwinBERTFeatureEncoder:
        """Get encoder by type."""
        if self.share_encoder:
            return getattr(self, FEATURE_ENCODER_STR)  # type: ignore
        return getattr(self, encoder_type + FEATURE_ENCODER_STR)  # type: ignore

    def forward(self, context: dict):
        """Compute embeddings and save in context."""
        seq_samples = context[ENCODER_SEQ]
        mask_samples = context[ENCODER_MASK]
        all_encoder_types = context[ENCODER_TYPES]

        embs_list = []
        term_tensors = None
        key_str = "data"

        for i, encoder_types in enumerate(all_encoder_types):
            if isinstance(encoder_types, list):
                embs = []
                for j, encoder_type in enumerate(encoder_types):
                    ctx = {
                        key_str: {
                            ENCODER_SEQ: seq_samples[i][j].squeeze(0),
                            ENCODER_MASK: mask_samples[i][j].squeeze(0),
                        }
                    }
                    self.get_feature_encoder(encoder_type)(
                        ctx,
                        pooler_index=self.encoder_types.index(encoder_type)
                        if self.share_encoder
                        else 0,
                        output_encoded_layer=True,
                    )
                    embs.append(ctx[key_str][OUTPUT])
                if i == 0:
                    term_tensors = ctx[key_str][TERM_TENSOR]

                embs_list.append(embs)

        context[EMBS_LIST] = embs_list
        context[TERM_TENSOR] = term_tensors
        context[ENCODER_MASK] = mask_samples[0][0].squeeze(0)

    def transform(self, context: dict):
        """Skip transform."""
        pass


def get_feature_encoder(
    args: argparse.Namespace,
) -> Optional[Union[TwinBERTFeatureEncoder, Tuple[MultiTypeFeatureEncoder, dict]]]:
    """Create feature encoder from command line arguments."""
    if (
        args.meta_dir is not None
        and len(args.meta_dir) > 0
        and args.featenc_config is not None
        and len(args.featenc_config) > 0
    ):
        config = TwinBERTEncoder.init_config_from_file(
            os.path.join(args.meta_dir, args.featenc_config)
        )
        config["enable_fp16"] = args.fp16
        if hasattr(args, "src_encoders") and hasattr(args, "dst_encoders"):
            encoders = list(
                set([t for i in args.src_encoders for t in i]).union(
                    set([t for i in args.dst_encoders for t in i])
                )
            )
            encoders.sort()

            return (
                MultiTypeFeatureEncoder(
                    get_feature_type(args.feature_type),
                    config,
                    encoders,
                    args.share_encoder,
                ),
                config,
            )  # here we also return the config object because model will use it.
        else:
            return TwinBERTFeatureEncoder(get_feature_type(args.feature_type), config)

    return None

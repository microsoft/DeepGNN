import os
import json
import logging
import collections
import torch
import torch.nn as nn

from .configuration import DeepSpeedArgs

from .embedding import TriletterEmbeddings
from .pooler import WeightPooler


class TwinBERTEncoder(nn.Module):
    def __init__(self, config, init_ckpt_file=None, pooler_count=1, local_rank=0):
        """TwinBERT feature encoder.

        Args:
            config: parameter dictionary.
            init_ckpt_file: start checkpoint, could be standard bert checkpoint or
                            TwinBERT pre-trained checkpoint.
            pooler_count: support multiple poolers sharing the same bert encoder.
                          this is to specify the number of poolers.
            local_rank: used by DeepSpeed to initialize transformer kernel.
        """  # noqa: D403
        super(TwinBERTEncoder, self).__init__()
        self.config = config
        self.deepspeed = False

        # if init_ckpt_file is not specified in param, check config.
        if not init_ckpt_file and "init_ckpt_file" in config:
            init_ckpt_file = config["init_ckpt_file"]

        if "deepspeed" in config:
            deepspeed_args = DeepSpeedArgs(config, local_rank)

            if deepspeed_args.apex:
                from .deepspeed.nvidia_modeling import BertModel, BertConfig
            else:
                from .deepspeed.nvidia_modeling_no_apex import BertModel, BertConfig

            self.bert_encoder = BertModel(BertConfig.from_dict(config), deepspeed_args)
            self.deepspeed = True
        else:
            from transformers.models.bert.modeling_bert import (  # type: ignore
                BertModel,
                BertConfig,
            )

            # from .huggingface.modeling_bert import BertModel, BertConfig
            self.bert_encoder = BertModel(BertConfig.from_dict(config))

        # Padding for divisibility by 8
        # vocab_size = config["vocab_size"]
        # if vocab_size % 8 != 0:
        #     vocab_diff = 8 - (vocab_size % 8)
        #     vocab_size += vocab_diff
        # config['vocab_size'] = vocab_size
        # logging.info(f"vocab_size = {vocab_size}")

        if config["embedding_type"] == "triletter":
            self.bert_encoder.embeddings = TriletterEmbeddings(config)
            self.bert_encoder.init_bert_weights(self.bert_encoder.embeddings)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if config["pooler_type"] == "weightpooler":
            for _ in range(pooler_count):
                self.poolers.append(WeightPooler(config))

        bert_name = "DeepSpeed" if self.deepspeed else "HuggingFace"
        logging.info(
            f"This model is using {bert_name} BERT. It has {len(self.poolers)} {config['pooler_type']}."
        )

        # downscale dense layer
        self.downscale_size = self.config["downscale"]
        if self.downscale_size > 0:
            self.downscale = nn.Linear(self.config["hidden_size"], self.downscale_size)

        self._load_from_checkpoint(init_ckpt_file)

    def forward(self, seq, mask, pooler_index=0, output_encoded_layer=False):

        # bert encoding
        if self.deepspeed:
            encoded_layer, _ = self.bert_encoder(
                input_ids=seq.type(torch.int64),
                attention_mask=mask.type(torch.int64),
                output_all_encoded_layers=False,
            )
        else:  # huggingface
            encoded_layer = self.bert_encoder(
                input_ids=seq.type(torch.int64), attention_mask=mask.type(torch.int64)
            )[0]

        # pooling
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](encoded_layer, mask)

        # downscaling
        if self.downscale_size > 0:
            output = torch.tanh(self.downscale(output))

        # output term embedding
        if output_encoded_layer:
            return output, encoded_layer

        return output

    def _load_from_checkpoint(self, init_ckpt_file):
        if init_ckpt_file and os.path.isfile(init_ckpt_file):
            if "init_ckpt_prefix" in self.config:
                self._load_from_other_checkpoint(init_ckpt_file)
            else:
                self.load_state_dict(torch.load(init_ckpt_file))

    def _load_from_other_checkpoint(self, init_ckpt_file):
        """Initialize twinbert encoder with twinbert/bert pre-trained checkpoint."""
        init_ckpt_prefix = self.config["init_ckpt_prefix"]

        state_dict = torch.load(init_ckpt_file, map_location="cpu")

        if init_ckpt_prefix.find("twinbert") >= 0:
            prefix = init_ckpt_prefix

            # twinbert query checkpoint
            if self.config["is_query"]:
                state_dict = self.extract_bert_by_prefix(
                    state_dict,
                    f"{prefix}.qbert_sentencoder|{prefix}.qpooler|{prefix}.qdownscale",
                    "bert_encoder|poolers.0|downscale",
                )
            else:  # twinbert query checkpoint
                state_dict = self.extract_bert_by_prefix(
                    state_dict,
                    f"{prefix}.kbert_sentencoder|{prefix}.kpooler|{prefix}.kdownscale",
                    "bert_encoder|poolers.0|downscale",
                )
        else:  # bert standard checkpoint
            state_dict = self.extract_bert_by_prefix(
                state_dict,
                f"{init_ckpt_prefix}|qpooler|qdownscale",
                "bert_encoder|poolers.0|downscale",
            )

            logging.info(
                "Only load bert state dict and leave rest parameters: "
                "poolers,downscale the same as initialized."
            )

        self.load_state_dict(state_dict)  # bert is the only module

    def extract_bert_by_prefix(self, state_dict, prefix="bert.bert", target_prefix=""):
        res = {}
        src_prefix = prefix.split("|")
        dst_prefix = target_prefix.split("|")
        exist_in_src = [False] * len(src_prefix)
        for key in state_dict:
            for i in range(len(src_prefix)):
                if src_prefix[i] in key:
                    exist_in_src[i] = True
                    idx = key.find(src_prefix[i]) + len(src_prefix[i])
                    name = key[idx:]
                    res[dst_prefix[i] + name] = state_dict[key]

        # Copy self state dict to res for key does not exist in src state dict.
        self_state_dict = self.state_dict()
        for key in self_state_dict:
            for i in range(len(src_prefix)):
                if not exist_in_src[i]:
                    if key.startswith(dst_prefix[i]):
                        res[key] = self_state_dict[key]

        return collections.OrderedDict(res)

    @classmethod
    def init_config_from_file(cls, filename: str) -> dict:
        config = {}
        with open(filename, "r", encoding="utf-8") as fin:
            config = json.load(fin)
        config["metadir"] = os.path.dirname(filename)
        return config

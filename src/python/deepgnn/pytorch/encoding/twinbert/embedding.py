import torch
import torch.nn as nn


class TriletterEmbeddings(nn.Module):
    """Taken from TwinBERT Author's implementation."""

    def __init__(self, config):
        super(TriletterEmbeddings, self).__init__()
        self.max_letters_in_word = config["triletter_max_letters_in_word"]
        self.triletter_embeddings = nn.Embedding(
            config["vocab_size"] + 1, config["hidden_size"], padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config["max_position_embeddings"] + 1, config["hidden_size"], padding_idx=0
        )

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.shape[1] // self.max_letters_in_word

        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device) + 1
        )
        position_ids = position_ids.unsqueeze(0).repeat(input_ids.shape[0], 1)

        # below two lines may be useful when we want to convert to onnx.
        # position_ids[attention_mask == 0] = 0
        # position_ids = position_ids.type(torch.float).masked_fill(attention_mask==0, 0.0).type(torch.int64)

        position_embeddings = self.position_embeddings(position_ids)

        # [batch_size, max_seq_len*max_letters_in_word, hidden_size]
        embeddings = self.triletter_embeddings(input_ids)

        # [batch_size, max_seq_len, max_letters_in_word, hidden_size]
        embeddings = embeddings.view(
            -1, seq_len, self.max_letters_in_word, embeddings.shape[-1]
        )
        embeddings = embeddings.sum(dim=2).view(-1, seq_len, embeddings.shape[-1])
        embeddings = embeddings + position_embeddings

        return embeddings

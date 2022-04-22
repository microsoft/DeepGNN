import torch.nn as nn
import torch.nn.functional as fun


class WeightPooler(nn.Module):
    """Taken from TwinBERT Author's implementation."""

    def __init__(self, config):
        super(WeightPooler, self).__init__()
        self.weighting = nn.Linear(config["hidden_size"], 1)
        # scale down to 1e-4 to avoid fp16 overflow.
        self.weight_factor = (
            1e-4 if "enable_fp16" in config and config["enable_fp16"] else 1e-8
        )

    def forward(self, term_tensor, mask):
        weights = self.weighting(term_tensor)
        weights = (
            weights + (mask - 1).type(weights.dtype).unsqueeze(2) / self.weight_factor
        )
        weights = fun.softmax(weights, dim=1)
        return (term_tensor * weights).sum(dim=1)

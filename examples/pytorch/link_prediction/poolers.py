# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn


def feedforwardpooler(src_vec, dst_vec, with_rng, ff_denses, random_negative):
    """Compute the similarity score by the dense layers.
    src_vec and dst_vec will be concat and passed to layers.
    If with_rng is true, negative samples will be calculated using random_negative.
    """
    assert isinstance(ff_denses, list) and len(ff_denses) == 2
    concat_vec = torch.cat([src_vec, dst_vec], dim=1)
    if not with_rng:
        simscore = ff_denses[1](torch.relu(ff_denses[0](concat_vec)) + concat_vec)
        simscore = simscore.squeeze()
        return simscore
    else:
        batch = [ff_denses[1](torch.relu(ff_denses[0](concat_vec)) + concat_vec)]
        for i in range(random_negative):
            batch_idx = torch.randperm(batch[0].shape[0]).to(batch[0].device)
            concat_vec = torch.cat([src_vec, dst_vec[batch_idx]], dim=1)
            batch.append(
                ff_denses[1](torch.relu(ff_denses[0](concat_vec)) + concat_vec)
            )
        batch = torch.cat(batch, dim=0)
        batch = batch.squeeze()
        return batch


def maxfeedforwardpooler(src_vec, dst_vec, with_rng, ff_denses, random_negative):
    """Compute the similarity score by the dense layers.
    Max value in src_vec and dst_vec will be retrieved and passed to layers.
    If with_rng is true, negative samples will be calculated using random_negative.
    """
    assert isinstance(ff_denses, list) and len(ff_denses) == 2
    concat_vec = torch.max(src_vec, dst_vec)
    if not with_rng:
        simscore = ff_denses[1](torch.relu(ff_denses[0](concat_vec)) + concat_vec)
        simscore = simscore.squeeze()
        return simscore
    else:
        batch = [ff_denses[1](torch.relu(ff_denses[0](concat_vec)) + concat_vec)]
        for i in range(random_negative):
            batch_idx = torch.randperm(batch[0].shape[0]).to(batch[0].device)
            concat_vec = torch.max(src_vec, dst_vec[batch_idx])
            batch.append(
                ff_denses[1](torch.relu(ff_denses[0](concat_vec)) + concat_vec)
            )
        batch = torch.cat(batch, dim=0)
        batch = batch.squeeze()
        return batch


def cosinepooler(src_vec, dst_vec, with_rng, ff_denses, random_negative):
    """Compute the similarity score using cosine similarity. This is usually used when
    doing recall action.
    If with_rng is true, negative samples will be calculated using random_negative.
    """
    assert isinstance(ff_denses, list) and len(ff_denses) == 1

    if not with_rng:
        simscore = ff_denses[0](
            nn.CosineSimilarity(dim=1, eps=1e-6)(src_vec, dst_vec).view(-1, 1)
        )
        simscore = simscore.squeeze()
        return simscore
    else:
        batch = [
            ff_denses[0](
                nn.CosineSimilarity(dim=1, eps=1e-6)(src_vec, dst_vec).view(-1, 1)
            )
        ]
        for i in range(random_negative):
            batch_idx = torch.randperm(batch[0].shape[0]).to(batch[0].device)
            batch.append(
                ff_denses[0](
                    nn.CosineSimilarity(dim=1, eps=1e-6)(
                        src_vec, dst_vec[batch_idx]
                    ).view(-1, 1)
                )
            )
        batch = torch.cat(batch, dim=0)
        batch = batch.squeeze()
        return batch


def cosinepooler_with_rns(src_vec, dst_vec, with_rng, nsp_gamma, random_negative):
    """Compute the similarity score using cosine similarity with random negative sampling.
    https://dl.acm.org/doi/pdf/10.1145/2661829.2661935
    """
    if not with_rng:
        simscore = nn.CosineSimilarity(dim=1, eps=1e-6)(src_vec, dst_vec).view(-1, 1)
        simscore = simscore.squeeze()
        return simscore
    else:
        batch = [
            (nn.CosineSimilarity(dim=1, eps=1e-6)(src_vec, dst_vec).view(-1, 1))
            * nsp_gamma
        ]
        # bs * 1
        for i in range(random_negative):
            batch_idx = torch.randperm(batch[0].shape[0]).to(batch[0].device)
            batch.append(
                (
                    nn.CosineSimilarity(dim=1, eps=1e-6)(
                        src_vec, dst_vec[batch_idx]
                    ).view(-1, 1)
                )
                * nsp_gamma
            )
        batch = torch.cat(batch, -1)
        batch = nn.functional.log_softmax(batch, -1)[:, 0]
        return batch


def res_layer(input, ff_denses):
    output = ff_denses[0](input)
    if len(ff_denses) == 4 and ff_denses[3] is not None:
        output = ff_denses[3](output)
    output = torch.relu(ff_denses[1](torch.relu(output)) + input)
    return output


def reslayerpooler(src_vec, dst_vec, with_rng, ff_denses, random_negative):
    """Compute similarity score using residual layers.
    src_vec and dst_vec will be concat and passed to res layer.
    """
    assert isinstance(ff_denses, list) and len(ff_denses) >= 3
    concat_vec = torch.cat([src_vec, dst_vec], dim=1)
    if not with_rng:
        simscore = ff_denses[2](res_layer(concat_vec, ff_denses))
        simscore = simscore.squeeze()
        return simscore
    else:
        batch = [ff_denses[2](res_layer(concat_vec, ff_denses))]
        for i in range(random_negative):
            batch_idx = torch.randperm(batch[0].shape[0]).to(batch[0].device)
            concat_vec = torch.cat([src_vec, dst_vec[batch_idx]], dim=1)
            batch.append(ff_denses[2](res_layer(concat_vec, ff_denses)))
        batch = torch.cat(batch, dim=0)
        batch = batch.squeeze()
        return batch


def maxreslayerpooler(src_vec, dst_vec, with_rng, ff_denses, random_negative):
    """Compute similarity score using residual layers.
    Max value of src_vec and dst_vec will be retrieved and passed to res layer.
    """
    assert isinstance(ff_denses, list) and len(ff_denses) >= 3
    concat_vec = torch.max(src_vec, dst_vec)
    if not with_rng:
        simscore = ff_denses[2](res_layer(concat_vec, ff_denses))
        simscore = simscore.squeeze()
        return simscore
    else:
        batch = [ff_denses[2](res_layer(concat_vec, ff_denses))]
        for i in range(random_negative):
            batch_idx = torch.randperm(batch[0].shape[0]).to(batch[0].device)
            concat_vec = torch.max(src_vec, dst_vec[batch_idx])
            batch.append(ff_denses[2](res_layer(concat_vec, ff_denses)))
        batch = torch.cat(batch, dim=0)
        batch = batch.squeeze()
        return batch


def selfattentionpooler(
    src_vec,
    src_term_tensor,
    src_mask,
    dst_vec,
    dst_term_tensor,
    dst_mask,
    ff_denses,
    random_negative,
    with_rng=False,
):
    """Compute the similarity score using attention layer."""
    if not with_rng:
        hidden_states = torch.cat([src_term_tensor, dst_term_tensor], dim=1)
        attention_mask = torch.cat([src_mask, dst_mask], dim=1)
        tensors = ff_denses[0](
            hidden_states,
            attention_mask.unsqueeze(1).unsqueeze(1).type(hidden_states.dtype),
        )
        cls_tensor = ff_denses[1](tensors[-1], attention_mask)
        simscore = ff_denses[2](cls_tensor).squeeze()
        return simscore
    else:
        # 4 random negatives at most
        hidden_states = torch.cat([src_term_tensor, dst_term_tensor], dim=1)
        attention_mask = torch.cat([src_mask, dst_mask], dim=1)
        tensors = ff_denses[0](
            hidden_states,
            attention_mask.unsqueeze(1).unsqueeze(1).type(hidden_states.dtype),
        )
        cls_tensor = ff_denses[1](None, tensors[-1], attention_mask)
        batch = [ff_denses[2](cls_tensor)]
        for i in range(random_negative):
            batch_idx = torch.randperm(batch[0].shape[0]).to(batch[0].device)
            hidden_states = torch.cat(
                [src_term_tensor, dst_term_tensor[batch_idx, :, :]], dim=1
            )
            attention_mask = torch.cat([src_mask, dst_mask[batch_idx, :]], dim=1)
            tensors = ff_denses[0](
                hidden_states,
                attention_mask.unsqueeze(1).unsqueeze(1).type(hidden_states.dtype),
            )
            cls_tensor = ff_denses[1](None, tensors[-1], attention_mask)
            batch.append(ff_denses[2](cls_tensor))
        batch = torch.cat(batch, dim=1)
        return batch

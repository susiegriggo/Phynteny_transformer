"""
Per-genome circular relative attention: fixes a batch-padding artifact identified
during architecture review of CircularRelativePositionAttention
(src/model_onehot.py). The original computes the circular wraparound fold using
the padded BATCH sequence length (query.size(1)) as the modulus for every genome
in the batch - so the wraparound point is an accident of which other, randomly
batched genomes a genome happens to share a training step with, not the genome's
own true length. Since batches are reshuffled every epoch, the same genome gets a
different, inconsistent fold point across training.

This module subclasses the existing classes and overrides ONLY the circular
index computation, using each genome's own true (unpadded) length - derived from
src_key_padding_mask - as its own modulus. Nothing in src/model_onehot.py is
modified; this is purely additive, so every existing trained model/result stays
exactly reproducible.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_onehot import (
    CircularRelativePositionAttention,
    CircularTransformerEncoderLayer,
    TransformerClassifierCircularRelativeAttention,
)


class CircularRelativePositionAttentionPerGenome(CircularRelativePositionAttention):
    """Same as CircularRelativePositionAttention, but the wraparound fold modulus
    is each sample's own true (unpadded) length, not the padded batch seq_len."""

    def forward(self, query, key, value, attn_mask=None, src_key_padding_mask=None,
                is_causal=False, output_dir=None, batch_idx=None, return_attn_weights=False):
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        batch_size, seq_len, d_model = query.size()
        device = query.device

        q = query.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        k = key.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        v = value.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_model // self.num_heads)

        # --- the fix: per-sample modulus instead of the shared padded seq_len ---
        if src_key_padding_mask is not None:
            true_len = src_key_padding_mask.sum(dim=1).clamp(min=1)  # (batch,)
        else:
            true_len = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)

        idx_range = torch.arange(seq_len, device=device)
        diff = idx_range.view(1, seq_len, 1) - idx_range.view(1, 1, seq_len)  # (1, seq_len, seq_len): [0,i,j] = i - j
        diff = diff.expand(batch_size, -1, -1)  # (batch, seq_len, seq_len)

        true_len_ = true_len.view(batch_size, 1, 1)
        circular_indices = (diff + true_len_) % true_len_
        circular_indices = torch.min(circular_indices, true_len_ - circular_indices)
        circular_indices = circular_indices.clamp(max=self.max_len - 1).long()
        # -------------------------------------------------------------------

        rel_positions_k = self.relative_position_k[circular_indices]  # (batch, seq_len, seq_len, head_dim)
        scores = scores + torch.einsum("bhqd,bqkd->bhqk", q, rel_positions_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(3) == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        if src_key_padding_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_output = torch.matmul(attn_weights, v)

        rel_positions_v = self.relative_position_v[circular_indices]  # (batch, seq_len, seq_len, head_dim)
        attn_output = attn_output + torch.einsum("bhqk,bqkd->bhqd", attn_weights, rel_positions_v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if output_dir is not None and batch_idx is not None:
            attn_weights_path = os.path.join(output_dir, f"attention_weights_batch_{batch_idx}.pkl")
            with open(attn_weights_path, "wb") as f:
                pickle.dump(attn_weights.cpu().detach().numpy(), f)

        if return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output


class CircularTransformerEncoderLayerPerGenome(CircularTransformerEncoderLayer):
    """Same as CircularTransformerEncoderLayer, but wired to the per-genome
    attention module above instead of the original batch-padded one."""

    def __init__(self, d_model, num_heads, dim_feedforward=512, dropout=0.1, max_len=1500,
                 intialisation='random', pre_norm=False):
        super().__init__(d_model, num_heads, dim_feedforward, dropout, max_len, intialisation, pre_norm)
        self.self_attn = CircularRelativePositionAttentionPerGenome(
            d_model, num_heads, max_len=max_len, batch_first=True, intialisation=intialisation
        )


class TransformerClassifierCircularRelativeAttentionPerGenome(TransformerClassifierCircularRelativeAttention):
    """Same as TransformerClassifierCircularRelativeAttention, but its transformer
    encoder layers use the per-genome circular attention fix. All constructor
    arguments must be passed as keywords (matching how this project always
    constructs these models), since __init__ reads a few of them back out of
    kwargs to rebuild the encoder after the parent constructor runs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_heads = kwargs.get("num_heads", 4)
        dropout = kwargs.get("dropout", 0.1)
        max_len = kwargs.get("max_len", 1500)
        intialisation = kwargs.get("intialisation", "random")
        pre_norm = kwargs.get("pre_norm", False)

        if self.num_layers > 0:
            hidden_dim = self.embedding_layer.out_features + self.gene_feature_dim
            device = next(self.parameters()).device
            encoder_layers = CircularTransformerEncoderLayerPerGenome(
                d_model=hidden_dim, num_heads=num_heads, dropout=dropout, max_len=max_len,
                intialisation=intialisation, pre_norm=pre_norm,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers).to(device)

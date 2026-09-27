"""
Two diagnostics for whether genomic context helps the model generalise to genes
that are hard from sequence alone, without relying on phold's calls as ground
truth (phold is itself an imperfect computational method, not gold-standard
truth - see the phold-recovery analysis for why that comparison alone can't
settle the question).

1. Confidence-stratified real-accuracy comparison (standard masked-prediction
   task, TRUE curated labels): bucket validation positions by the baseline
   model's own prediction confidence (a proxy for "how ambiguous is this gene
   from sequence alone"), then compare circular_fixed vs baseline TRUE accuracy
   within each bucket. If circular_fixed pulls ahead specifically where
   baseline is least confident, that's direct, ground-truth-backed evidence
   context helps generalisation in hard cases.

2. Calibration on genuinely novel genes (unknown in BOTH the curated labels and
   phold_data.y.pkl - no ground truth exists anywhere for these): compare each
   model's own confidence/entropy on these genes against its confidence on
   ordinary known genes. A model that drops confidence appropriately on truly
   novel genes is better calibrated - it "knows what it doesn't know" - which
   is meaningful even without any label to check correctness against.

This compares the "circular_fixed" (per-genome wraparound fix - see
src/model_onehot_circular_fixed.py and train_transformer/train_circular_fixed.py)
and "baseline" checkpoints for one fold in a single pass, reusing the
checkpoint-loading approach from evaluate_phold_recovery.py. This is a
deliberately different question from circular_fixed-vs-original (which is
about whether the batch-padding fix changes raw accuracy): here we're asking
whether the batch-padding-corrected model shows more evidence of leveraging
gene order/context than a context-free baseline, independent of that.
"""
import os
import pickle
import json

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.model_onehot import (
    EmbeddingDataset,
    collate_fn,
    CircularRelativePositionAttention,
    CircularTransformerEncoderLayer,
    TransformerClassifierCircularRelativeAttention,
    fourier_positional_encoding,
)

NUM_CLASSES = 9
CONFIDENCE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# The three classes below are also defined in src/model_onehot_circular_fixed.py
# and train_transformer/train_circular_fixed.py. Duplicated here rather than
# imported for the same reason BaselineClassifier is duplicated (see below):
# `src` resolves to an installed copy in some environments rather than this
# repo's checkout, which breaks cross-module imports of newly-added files.


class CircularRelativePositionAttentionPerGenome(CircularRelativePositionAttention):
    """Same as CircularRelativePositionAttention, but the wraparound fold modulus
    is each sample's own true (unpadded) length - from src_key_padding_mask -
    instead of the padded batch seq_len."""

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
    def __init__(self, d_model, num_heads, dim_feedforward=512, dropout=0.1, max_len=1500,
                 intialisation='random', pre_norm=False):
        super().__init__(d_model, num_heads, dim_feedforward, dropout, max_len, intialisation, pre_norm)
        self.self_attn = CircularRelativePositionAttentionPerGenome(
            d_model, num_heads, max_len=max_len, batch_first=True, intialisation=intialisation
        )


class TransformerClassifierCircularRelativeAttentionPerGenome(TransformerClassifierCircularRelativeAttention):
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


class BaselineClassifier(nn.Module):
    """Duplicated from train_transformer/train_baseline.py - see the note in
    evaluate_phold_recovery.py for why (train_transformer/ isn't a package)."""

    def __init__(self, protein_dim, num_classes, hidden_dim=256, dropout=0.1,
                 strand_embedding_dim=2, length_embedding_dim=8):
        super().__init__()
        self.num_classes = num_classes
        self.strand_embedding = nn.Linear(2, strand_embedding_dim)
        self.length_embedding = nn.Linear(1, length_embedding_dim)
        protein_embedding_dim = hidden_dim - strand_embedding_dim - length_embedding_dim
        self.embedding_layer = nn.Linear(protein_dim, protein_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        strand_ids = x[:, :, self.num_classes:self.num_classes + 2]
        gene_length = x[:, :, self.num_classes + 2:self.num_classes + 3]
        protein_embeds = x[:, :, self.num_classes + 3:]
        strand_embeds = self.strand_embedding(strand_ids.float())
        length_embeds = self.length_embedding(gene_length)
        protein_embeds = self.embedding_layer(protein_embeds)
        combined = torch.cat([strand_embeds, length_embeds, protein_embeds], dim=-1)
        return self.mlp(combined)


def load_circular_fixed_model(checkpoint_path, input_dim, hidden_dim, num_heads, num_layers, dropout, device):
    model = TransformerClassifierCircularRelativeAttentionPerGenome(
        input_dim=input_dim, num_classes=NUM_CLASSES, num_heads=num_heads, num_layers=num_layers,
        hidden_dim=hidden_dim, lstm_hidden_dim=512, dropout=dropout, use_lstm=False,
        positional_encoding=fourier_positional_encoding, use_positional_encoding=True,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()
    return model


def load_baseline_model(checkpoint_path, protein_dim, hidden_dim, dropout, device):
    model = BaselineClassifier(protein_dim=protein_dim, num_classes=NUM_CLASSES, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()
    return model


def normalize_idx(idx):
    """In EmbeddingDataset's no-masking mode (set_training(False)), each sample's
    idx comes back as a plain empty Python list rather than a LongTensor, which
    the model's internal protein_feature_dropout doesn't handle (it calls
    idx[b].numel()). Convert to empty/real LongTensors without touching model_onehot.py."""
    return [ii if torch.is_tensor(ii) else torch.tensor(ii, dtype=torch.long) for ii in idx]


def softmax_confidence_and_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    confidence, _ = probs.max(dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return confidence, entropy


def bin_index(confidence):
    return min(int(confidence * len(CONFIDENCE_BINS[:-1])), len(CONFIDENCE_BINS) - 2)


def find_unknown_in_both(y, phold_y):
    """(genome_key, position) pairs where both the curated label and phold's
    call are -1 - no ground truth exists anywhere for these."""
    pairs = []
    for key, true_cat in y.items():
        phold_cat = phold_y.get(key)
        if phold_cat is None or phold_cat.shape != true_cat.shape:
            continue
        both_unknown = (true_cat == -1) & (phold_cat == -1)
        for pos in torch.nonzero(both_unknown).flatten().tolist():
            pairs.append((key, pos))
    return pairs


@click.command()
@click.option("--x_path", required=True, type=click.Path(exists=True))
@click.option("--y_path", required=True, type=click.Path(exists=True))
@click.option("--phold_y_path", required=True, type=click.Path(exists=True))
@click.option("--circular_fixed_checkpoint", required=True, type=click.Path(exists=True))
@click.option("--baseline_checkpoint", required=True, type=click.Path(exists=True))
@click.option("--fold_index", required=True, type=int)
@click.option("--n_splits", default=10, type=int)
@click.option("--random_seed", default=42, type=int)
@click.option("--mask_portion", default=0.3, type=float)
@click.option("--n_passes", default=5, type=int, help="Repeated random-mask draws over the validation set, for more complete position coverage on the standard task.")
@click.option("--hidden_dim", default=256, type=int)
@click.option("--num_heads", default=4, type=int)
@click.option("--num_layers", default=2, type=int)
@click.option("--dropout", default=0.05, type=float)
@click.option("--batch_size", default=64, type=int)
@click.option("--device", default="cuda", type=str)
@click.option("--out", required=True, type=click.Path())
def main(x_path, y_path, phold_y_path, circular_fixed_checkpoint, baseline_checkpoint, fold_index,
         n_splits, random_seed, mask_portion, n_passes, hidden_dim, num_heads, num_layers,
         dropout, batch_size, device, out):
    logger.info("Reading in data")
    X = pickle.load(open(x_path, "rb"))
    y = pickle.load(open(y_path, "rb"))
    phold_y = pickle.load(open(phold_y_path, "rb"))
    protein_dim = list(X.values())[0].shape[1] - 3
    logger.info(f"protein_dim={protein_dim}")

    unknown_both = find_unknown_in_both(y, phold_y)
    unknown_both_by_key = {}
    for key, pos in unknown_both:
        unknown_both_by_key.setdefault(key, []).append(pos)
    logger.info(f"Genuinely unknown-in-both positions: {len(unknown_both)}")

    keys = list(y.keys())
    dataset_std = EmbeddingDataset(list(X.values()), list(y.values()), keys, mask_portion=mask_portion)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    val_index = None
    for fold, (_, v_idx) in enumerate(kf.split(dataset_std), 1):
        if fold == fold_index:
            val_index = v_idx
            break
    logger.info(f"Fold {fold_index}: {len(val_index)} validation genomes")

    circular_fixed_model = load_circular_fixed_model(circular_fixed_checkpoint, protein_dim, hidden_dim, num_heads, num_layers, dropout, device)
    baseline_model = load_baseline_model(baseline_checkpoint, protein_dim, hidden_dim, dropout, device)

    # --- Part 1: confidence-stratified real accuracy on the standard task ---
    dataset_std.set_training(True)  # same random category masking used during training
    bin_correct = {name: [0] * (len(CONFIDENCE_BINS) - 1) for name in ("circular_fixed", "baseline")}
    bin_total = [0] * (len(CONFIDENCE_BINS) - 1)

    for p in range(n_passes):
        loader = DataLoader(dataset_std, batch_size=batch_size, sampler=SubsetRandomSampler(val_index), collate_fn=collate_fn, pin_memory=True)
        with torch.no_grad():
            for embeddings, categories, masks, idx in loader:
                embeddings = embeddings.to(device).float()
                categories = categories.to(device).long()
                src_key_padding_mask = (masks.to(device) != -2).bool()

                circular_fixed_logits = circular_fixed_model(embeddings, idx=idx, src_key_padding_mask=src_key_padding_mask)
                baseline_logits = baseline_model(embeddings)

                circular_fixed_preds = circular_fixed_logits.argmax(dim=-1)
                baseline_preds = baseline_logits.argmax(dim=-1)
                baseline_conf, _ = softmax_confidence_and_entropy(baseline_logits)

                for b, sample_idx in enumerate(idx):
                    for pos in sample_idx.tolist():
                        true_cat = categories[b, pos].item()
                        conf = baseline_conf[b, pos].item()
                        bi = bin_index(conf)
                        bin_total[bi] += 1
                        bin_correct["circular_fixed"][bi] += int(circular_fixed_preds[b, pos].item() == true_cat)
                        bin_correct["baseline"][bi] += int(baseline_preds[b, pos].item() == true_cat)
        logger.info(f"Standard-task pass {p + 1}/{n_passes} done")

    standard_task_results = {
        "bin_edges": CONFIDENCE_BINS,
        "bin_total": bin_total,
        "bin_correct_circular_fixed": bin_correct["circular_fixed"],
        "bin_correct_baseline": bin_correct["baseline"],
    }
    for i in range(len(CONFIDENCE_BINS) - 1):
        if bin_total[i] > 0:
            logger.info(
                f"baseline_conf in [{CONFIDENCE_BINS[i]:.1f},{CONFIDENCE_BINS[i+1]:.1f}): n={bin_total[i]} "
                f"circular_fixed_acc={bin_correct['circular_fixed'][i]/bin_total[i]:.4f} "
                f"baseline_acc={bin_correct['baseline'][i]/bin_total[i]:.4f}"
            )

    # --- Part 2: calibration on genes unknown in both, vs. ordinary known genes ---
    dataset_std.set_training(False)  # no masking - every position reflects its true known category (or zero if -1)

    def collect_confidence(genome_keys_and_positions):
        """genome_keys_and_positions: list of (key, position) restricted to this fold's val genomes only."""
        by_key = {}
        for key, pos in genome_keys_and_positions:
            by_key.setdefault(key, []).append(pos)
        val_keys = {keys[i] for i in val_index}
        target_keys = [k for k in by_key if k in val_keys]
        if not target_keys:
            return {"circular_fixed": [], "baseline": []}

        key_to_dataset_idx = {keys[i]: i for i in val_index}
        confidences = {"circular_fixed": [], "baseline": []}
        loader = DataLoader(dataset_std, batch_size=batch_size, sampler=[key_to_dataset_idx[k] for k in target_keys], collate_fn=collate_fn)
        with torch.no_grad():
            row = 0
            for embeddings, categories, masks, idx in loader:
                embeddings = embeddings.to(device).float()
                src_key_padding_mask = (masks.to(device) != -2).bool()
                idx = normalize_idx(idx)
                circular_fixed_logits = circular_fixed_model(embeddings, idx=idx, src_key_padding_mask=src_key_padding_mask)
                baseline_logits = baseline_model(embeddings)
                cf_conf, _ = softmax_confidence_and_entropy(circular_fixed_logits)
                base_conf, _ = softmax_confidence_and_entropy(baseline_logits)
                batch_keys = target_keys[row:row + embeddings.shape[0]]
                for b, key in enumerate(batch_keys):
                    for pos in by_key[key]:
                        if pos < cf_conf.shape[1]:
                            confidences["circular_fixed"].append(cf_conf[b, pos].item())
                            confidences["baseline"].append(base_conf[b, pos].item())
                row += embeddings.shape[0]
        return confidences

    novel_confidence = collect_confidence(unknown_both)

    # reference: confidence on ordinary KNOWN (real, non -1) positions in this fold's val genomes
    known_pairs = []
    val_keys_set = {keys[i] for i in val_index}
    for key in val_keys_set:
        cat = y[key]
        for pos in torch.nonzero(cat != -1).flatten().tolist():
            known_pairs.append((key, pos))
    known_confidence = collect_confidence(known_pairs)

    for name in ("circular_fixed", "baseline"):
        nv, kn = novel_confidence[name], known_confidence[name]
        logger.info(
            f"{name}: mean confidence on novel(unknown-in-both)={np.mean(nv) if nv else float('nan'):.4f} (n={len(nv)}), "
            f"mean confidence on known={np.mean(kn) if kn else float('nan'):.4f} (n={len(kn)})"
        )

    calibration_results = {
        "novel_confidence_circular_fixed": novel_confidence["circular_fixed"],
        "novel_confidence_baseline": novel_confidence["baseline"],
        "known_confidence_circular_fixed_summary": {"mean": float(np.mean(known_confidence["circular_fixed"])) if known_confidence["circular_fixed"] else None, "n": len(known_confidence["circular_fixed"])},
        "known_confidence_baseline_summary": {"mean": float(np.mean(known_confidence["baseline"])) if known_confidence["baseline"] else None, "n": len(known_confidence["baseline"])},
    }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"fold": fold_index, "standard_task": standard_task_results, "calibration": calibration_results}, f, indent=2)
    logger.info(f"Saved {out}")
    logger.info("FINISHED! :D")


if __name__ == "__main__":
    main()

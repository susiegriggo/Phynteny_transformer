"""
Two diagnostics for whether genomic context helps the model generalise to genes
that are hard from sequence alone, without relying on phold's calls as ground
truth (phold is itself an imperfect computational method, not gold-standard
truth - see the phold-recovery analysis for why that comparison alone can't
settle the question).

1. Confidence-stratified real-accuracy comparison (standard masked-prediction
   task, TRUE curated labels): bucket validation positions by the baseline
   model's own prediction confidence (a proxy for "how ambiguous is this gene
   from sequence alone"), then compare original vs baseline TRUE accuracy
   within each bucket. If original pulls ahead specifically where baseline is
   least confident, that's direct, ground-truth-backed evidence context helps
   generalisation in hard cases.

2. Calibration on genuinely novel genes (unknown in BOTH the curated labels and
   phold_data.y.pkl - no ground truth exists anywhere for these): compare each
   model's own confidence/entropy on these genes against its confidence on
   ordinary known genes. A model that drops confidence appropriately on truly
   novel genes is better calibrated - it "knows what it doesn't know" - which
   is meaningful even without any label to check correctness against.

Evaluates both the "original" and "baseline" checkpoints for one fold in a
single pass, reusing the checkpoint-loading approach from
evaluate_phold_recovery.py.
"""
import os
import pickle
import json

import click
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.model_onehot import (
    EmbeddingDataset,
    collate_fn,
    TransformerClassifierCircularRelativeAttention,
    fourier_positional_encoding,
)

NUM_CLASSES = 9
CONFIDENCE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


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


def load_original_model(checkpoint_path, input_dim, hidden_dim, num_heads, num_layers, dropout, device):
    model = TransformerClassifierCircularRelativeAttention(
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
@click.option("--original_checkpoint", required=True, type=click.Path(exists=True))
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
def main(x_path, y_path, phold_y_path, original_checkpoint, baseline_checkpoint, fold_index,
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

    original_model = load_original_model(original_checkpoint, protein_dim, hidden_dim, num_heads, num_layers, dropout, device)
    baseline_model = load_baseline_model(baseline_checkpoint, protein_dim, hidden_dim, dropout, device)

    # --- Part 1: confidence-stratified real accuracy on the standard task ---
    dataset_std.set_training(True)  # same random category masking used during training
    bin_correct = {name: [0] * (len(CONFIDENCE_BINS) - 1) for name in ("original", "baseline")}
    bin_total = [0] * (len(CONFIDENCE_BINS) - 1)

    for p in range(n_passes):
        loader = DataLoader(dataset_std, batch_size=batch_size, sampler=SubsetRandomSampler(val_index), collate_fn=collate_fn, pin_memory=True)
        with torch.no_grad():
            for embeddings, categories, masks, idx in loader:
                embeddings = embeddings.to(device).float()
                categories = categories.to(device).long()
                src_key_padding_mask = (masks.to(device) != -2).bool()

                original_logits = original_model(embeddings, idx=idx, src_key_padding_mask=src_key_padding_mask)
                baseline_logits = baseline_model(embeddings)

                original_preds = original_logits.argmax(dim=-1)
                baseline_preds = baseline_logits.argmax(dim=-1)
                baseline_conf, _ = softmax_confidence_and_entropy(baseline_logits)

                for b, sample_idx in enumerate(idx):
                    for pos in sample_idx.tolist():
                        true_cat = categories[b, pos].item()
                        conf = baseline_conf[b, pos].item()
                        bi = bin_index(conf)
                        bin_total[bi] += 1
                        bin_correct["original"][bi] += int(original_preds[b, pos].item() == true_cat)
                        bin_correct["baseline"][bi] += int(baseline_preds[b, pos].item() == true_cat)
        logger.info(f"Standard-task pass {p + 1}/{n_passes} done")

    standard_task_results = {
        "bin_edges": CONFIDENCE_BINS,
        "bin_total": bin_total,
        "bin_correct_original": bin_correct["original"],
        "bin_correct_baseline": bin_correct["baseline"],
    }
    for i in range(len(CONFIDENCE_BINS) - 1):
        if bin_total[i] > 0:
            logger.info(
                f"baseline_conf in [{CONFIDENCE_BINS[i]:.1f},{CONFIDENCE_BINS[i+1]:.1f}): n={bin_total[i]} "
                f"original_acc={bin_correct['original'][i]/bin_total[i]:.4f} "
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
            return {"original": [], "baseline": []}

        key_to_dataset_idx = {keys[i]: i for i in val_index}
        confidences = {"original": [], "baseline": []}
        loader = DataLoader(dataset_std, batch_size=batch_size, sampler=[key_to_dataset_idx[k] for k in target_keys], collate_fn=collate_fn)
        with torch.no_grad():
            row = 0
            for embeddings, categories, masks, idx in loader:
                embeddings = embeddings.to(device).float()
                src_key_padding_mask = (masks.to(device) != -2).bool()
                idx = normalize_idx(idx)
                original_logits = original_model(embeddings, idx=idx, src_key_padding_mask=src_key_padding_mask)
                baseline_logits = baseline_model(embeddings)
                orig_conf, _ = softmax_confidence_and_entropy(original_logits)
                base_conf, _ = softmax_confidence_and_entropy(baseline_logits)
                batch_keys = target_keys[row:row + embeddings.shape[0]]
                for b, key in enumerate(batch_keys):
                    for pos in by_key[key]:
                        if pos < orig_conf.shape[1]:
                            confidences["original"].append(orig_conf[b, pos].item())
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

    for name in ("original", "baseline"):
        nv, kn = novel_confidence[name], known_confidence[name]
        logger.info(
            f"{name}: mean confidence on novel(unknown-in-both)={np.mean(nv) if nv else float('nan'):.4f} (n={len(nv)}), "
            f"mean confidence on known={np.mean(kn) if kn else float('nan'):.4f} (n={len(kn)})"
        )

    calibration_results = {
        "novel_confidence_original": novel_confidence["original"],
        "novel_confidence_baseline": novel_confidence["baseline"],
        "known_confidence_original_summary": {"mean": float(np.mean(known_confidence["original"])) if known_confidence["original"] else None, "n": len(known_confidence["original"])},
        "known_confidence_baseline_summary": {"mean": float(np.mean(known_confidence["baseline"])) if known_confidence["baseline"] else None, "n": len(known_confidence["baseline"])},
    }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"fold": fold_index, "standard_task": standard_task_results, "calibration": calibration_results}, f, indent=2)
    logger.info(f"Saved {out}")
    logger.info("FINISHED! :D")


if __name__ == "__main__":
    main()

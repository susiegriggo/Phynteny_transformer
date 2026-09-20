"""
Does genomic context let Phynteny recover a correct functional category specifically
where sequence-based evidence alone found nothing?

phold_data.y.pkl is data.y.pkl with every position that was unknown (-1) in the
curated/PHROG-based labels filled in wherever phold's structure-based comparison
found a confident category (never overwriting an existing curated label). This
script evaluates, for the exact positions where that fill-in happened, whether the
"original" (context-aware, attention) model and the "baseline" (own-features-only,
no-context) model predict a category matching phold's call - using each gene's
own fold's checkpoint, so every gene is scored by a model that never saw it during
training (proper out-of-fold evaluation), and both models are compared on the
IDENTICAL held-out positions per fold.

Uses EmbeddingDataset's existing set_validation() mechanism: idx ends up being
exactly the positions where our constructed validation_categories differs from the
true training labels - which, by construction, is exactly the phold-filled set.

Run this once per fold (it evaluates BOTH model types for that fold in one go, to
avoid paying the ~6 minute data-load cost twice).
"""
import os
import pickle
import json

import click
import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn

from src.model_onehot import (
    EmbeddingDataset,
    collate_fn,
    TransformerClassifierCircularRelativeAttention,
    fourier_positional_encoding,
)

NUM_CLASSES = 9


class BaselineClassifier(nn.Module):
    """Same architecture as train_transformer/train_baseline.py's BaselineClassifier -
    duplicated here rather than imported, since train_transformer/ isn't a proper
    Python package (no __init__.py) and its import path isn't reliable across how
    this repo gets invoked on Setonix."""

    def __init__(self, protein_dim, num_classes, hidden_dim=256, dropout=0.1,
                 strand_embedding_dim=2, length_embedding_dim=8):
        super().__init__()
        self.num_classes = num_classes
        self.strand_embedding = nn.Linear(2, strand_embedding_dim)
        self.length_embedding = nn.Linear(1, length_embedding_dim)
        protein_embedding_dim = hidden_dim - strand_embedding_dim - length_embedding_dim
        self.embedding_layer = nn.Linear(protein_dim, protein_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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


def build_validation_categories(y, phold_y):
    """
    For each genome, start from the true (curated) categories, and fill in ONLY the
    positions that were -1 (unknown) there and have a real category in phold_y. All
    other positions keep the original true label untouched.
    """
    validation_categories = []
    for key in y.keys():
        true_cat = y[key].clone()
        phold_cat = phold_y.get(key)
        if phold_cat is not None and phold_cat.shape == true_cat.shape:
            fill = (true_cat == -1) & (phold_cat != -1)
            true_cat[fill] = phold_cat[fill]
        validation_categories.append(true_cat)
    return validation_categories


def accumulate_correct(outputs, categories, idx):
    """outputs: (batch, seq, num_classes) logits. idx: list of 1D LongTensors, one
    per sample, giving the positions to score - here, exactly the phold-filled ones
    for genomes that have any (empty for genomes with none)."""
    correct, total = 0, 0
    preds = outputs.argmax(dim=-1)
    for b, sample_idx in enumerate(idx):
        if len(sample_idx) == 0:
            continue
        p = preds[b, sample_idx]
        t = categories[b, sample_idx]
        correct += (p == t).sum().item()
        total += len(sample_idx)
    return correct, total


def load_original_model(checkpoint_path, input_dim, hidden_dim, num_heads, num_layers, dropout, device):
    model = TransformerClassifierCircularRelativeAttention(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        lstm_hidden_dim=512,
        dropout=dropout,
        use_lstm=False,
        positional_encoding=fourier_positional_encoding,
        use_positional_encoding=True,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_baseline_model(checkpoint_path, protein_dim, hidden_dim, dropout, device):
    model = BaselineClassifier(protein_dim=protein_dim, num_classes=NUM_CLASSES, hidden_dim=hidden_dim, dropout=dropout).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@click.command()
@click.option("--x_path", required=True, type=click.Path(exists=True), help="Path to data.X.pkl")
@click.option("--y_path", required=True, type=click.Path(exists=True), help="Path to data.y.pkl (curated labels)")
@click.option("--phold_y_path", required=True, type=click.Path(exists=True), help="Path to phold_data.y.pkl")
@click.option("--original_checkpoint", required=True, type=click.Path(exists=True), help="transformer_state_dict.pth for this fold's original model")
@click.option("--baseline_checkpoint", required=True, type=click.Path(exists=True), help="transformer_state_dict.pth for this fold's baseline model")
@click.option("--fold_index", required=True, type=int, help="Which fold (1-10) these checkpoints correspond to")
@click.option("--n_splits", default=10, type=int)
@click.option("--random_seed", default=42, type=int)
@click.option("--hidden_dim", default=256, type=int)
@click.option("--num_heads", default=4, type=int)
@click.option("--num_layers", default=2, type=int)
@click.option("--dropout", default=0.05, type=float)
@click.option("--batch_size", default=64, type=int)
@click.option("--device", default="cuda", type=str)
@click.option("--out", required=True, type=click.Path(), help="Output .json path for this fold's result")
def main(x_path, y_path, phold_y_path, original_checkpoint, baseline_checkpoint, fold_index,
         n_splits, random_seed, hidden_dim, num_heads, num_layers, dropout, batch_size, device, out):
    logger.info("Reading in data")
    X = pickle.load(open(x_path, "rb"))
    y = pickle.load(open(y_path, "rb"))
    phold_y = pickle.load(open(phold_y_path, "rb"))
    protein_dim = list(X.values())[0].shape[1] - 3
    logger.info(f"protein_dim={protein_dim}")

    validation_categories = build_validation_categories(y, phold_y)
    n_filled = sum(int((vc != tc).sum()) for vc, tc in zip(validation_categories, y.values()))
    logger.info(f"Total phold-filled positions across whole dataset: {n_filled}")

    dataset = EmbeddingDataset(list(X.values()), list(y.values()), list(y.keys()))
    dataset.set_validation(validation_categories, validation=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
        if fold != fold_index:
            continue
        logger.info(f"Fold {fold}: {len(val_index)} validation genomes")
        val_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_index),
            collate_fn=collate_fn, pin_memory=True,
        )

        original_model = load_original_model(original_checkpoint, protein_dim, hidden_dim, num_heads, num_layers, dropout, device)
        baseline_model = load_baseline_model(baseline_checkpoint, protein_dim, hidden_dim, dropout, device)

        results = {}
        for name, model in [("original", original_model), ("baseline", baseline_model)]:
            correct_total, positions_total = 0, 0
            with torch.no_grad():
                for embeddings, categories, masks, idx in val_loader:
                    embeddings = embeddings.to(device).float()
                    categories = categories.to(device).long()
                    if name == "original":
                        src_key_padding_mask = (masks.to(device) != -2).bool()
                        outputs = model(embeddings, idx=idx, src_key_padding_mask=src_key_padding_mask)
                    else:
                        outputs = model(embeddings)
                    c, t = accumulate_correct(outputs, categories, idx)
                    correct_total += c
                    positions_total += t

            acc = correct_total / positions_total if positions_total > 0 else float("nan")
            logger.info(f"{name}: {correct_total}/{positions_total} = {acc:.4f} agreement with phold's call")
            results[name] = {"correct": correct_total, "total": positions_total, "accuracy": acc}

        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump({"fold": fold, "n_val_genomes": len(val_index), **results}, f, indent=2)
        logger.info(f"Saved {out}")
        break

    logger.info("FINISHED! :D")


if __name__ == "__main__":
    main()

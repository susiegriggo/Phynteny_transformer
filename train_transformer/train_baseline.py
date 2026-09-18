"""
Context-free baseline for the Reviewer 3 / Reviewer 2 comparisons: predicts each
gene's masked functional category from its OWN features only (ESM protein
embedding + strand + gene length) via a simple MLP - no attention, no BiLSTM, no
positional information, and no information about any other gene in the genome at
all. Trained on the exact same K-fold splits (same random_state=42), masking
scheme, and data as train_transformer.py, so its accuracy is directly comparable
fold-for-fold to the "original"/"no_sinusoidal"/"untrained" results already
produced there.

This answers two things at once:
  - Reviewer 3's request for "a simple baseline (e.g. ESM embeddings + MLP) to
    quantify how much the complex architecture actually contributes beyond the
    protein language model features".
  - Whether the model uses genomic context/gene order at all: if the full
    contextual model beats this baseline by a real margin, that's direct evidence
    context matters - independent of what the (separately shown to be partly
    artifactual) attention periodicity looks like.
"""
import os
import pickle

import click
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.model_onehot import (
    EmbeddingDataset,
    collate_fn,
    calculate_accuracy,
    cosine_lr_scheduler,
)


class BaselineClassifier(nn.Module):
    """Per-gene MLP: protein embedding + strand + length -> class logits. Every
    gene is scored completely independently of every other gene in the genome -
    no attention, no recurrence, no positional information of any kind."""

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
        # x layout matches what EmbeddingDataset/collate_fn produce for the
        # attention models: [one_hot(masked category) | strand | length | protein
        # embedding]. We deliberately skip the leading num_classes columns - that
        # channel carries (possibly-known) category information, which is exactly
        # the "self/neighbour label" signal this baseline must not have access to.
        strand_ids = x[:, :, self.num_classes:self.num_classes + 2]
        gene_length = x[:, :, self.num_classes + 2:self.num_classes + 3]
        protein_embeds = x[:, :, self.num_classes + 3:]

        strand_embeds = self.strand_embedding(strand_ids.float())
        length_embeds = self.length_embedding(gene_length)
        protein_embeds = self.embedding_layer(protein_embeds)

        combined = torch.cat([strand_embeds, length_embeds, protein_embeds], dim=-1)
        return self.mlp(combined)


def masked_ce_loss(outputs, categories, idx, criterion):
    """Cross-entropy computed only at the deliberately-masked positions (idx),
    matching the same masked-prediction objective train_transformer.py uses -
    otherwise this baseline would get free extra supervision the attention models
    never see, which would make the comparison unfair rather than a clean ablation."""
    preds, targets = [], []
    for b, sample_idx in enumerate(idx):
        if len(sample_idx) == 0:
            continue
        preds.append(outputs[b, sample_idx])
        targets.append(categories[b, sample_idx])
    if not preds:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    return criterion(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def train_one_fold(fold, train_index, val_index, dataset, num_classes, hidden_dim,
                    dropout, batch_size, epochs, lr, min_lr_ratio, device, save_path,
                    protein_dim, checkpoint_interval):
    output_dir = os.path.join(save_path, f"fold_{fold}")
    os.makedirs(output_dir, exist_ok=True)
    fold_logger = logger.bind(fold=fold)
    fold_logger.add(os.path.join(output_dir, "trainer.log"), level="DEBUG")

    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn, pin_memory=True)

    with open(os.path.join(output_dir, "val_kfold_loader.pkl"), "wb") as f:
        pickle.dump(val_loader, f)

    model = BaselineClassifier(protein_dim=protein_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = epochs / 4
    scheduler = cosine_lr_scheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, min_lr_ratio=min_lr_ratio)
    criterion = nn.CrossEntropyLoss()

    metrics = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_model.pth")

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0.0, 0
        for embeddings, categories, masks, idx in train_loader:
            embeddings = embeddings.to(device).float()
            categories = categories.to(device).long()

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = masked_ce_loss(outputs, categories, idx, criterion)
            loss.backward()
            optimizer.step()
            scheduler.step()

            accuracy = calculate_accuracy(outputs, categories, masks.to(device).float(), idx)
            total_loss += loss.item()
            total_correct += accuracy * len(idx)
            total_samples += len(idx)

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples

        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for embeddings, categories, masks, idx in val_loader:
                embeddings = embeddings.to(device).float()
                categories = categories.to(device).long()

                outputs = model(embeddings)
                loss = masked_ce_loss(outputs, categories, idx, criterion)
                accuracy = calculate_accuracy(outputs, categories, masks.to(device).float(), idx)

                total_val_loss += loss.item()
                total_val_correct += accuracy * len(idx)
                total_val_samples += len(idx)

        val_loss = total_val_loss / len(val_loader)
        val_acc = total_val_correct / total_val_samples

        fold_logger.info(
            f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        metrics.append({
            "epoch": epoch,
            "training losses": train_loss,
            "validation losses": val_loss,
            "training accuracies": train_acc,
            "validation accuracies": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            fold_logger.info(f"New best validation loss: {best_val_loss:.4f}. Model saved to {best_model_path}")

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(output_dir, "transformer_state_dict.pth"))
    fold_logger.info(f"Fold {fold} done. Best val loss: {best_val_loss:.4f}")


@click.command()
@click.option("--x_path", "-x", required=True, type=click.Path(exists=True), help="File path to X training data")
@click.option("--y_path", "-y", required=True, type=click.Path(exists=True), help="File path to y training data")
@click.option("--mask_portion", default=0.3, type=float, help="Portion of knowns to mask during training")
@click.option("--epochs", default=50, type=int, help="Number of training epochs")
@click.option("--lr", default=1e-5, type=float, help="Learning rate")
@click.option("--min_lr_ratio", default=0.1, type=float, help="Minimum learning rate ratio for the cosine scheduler")
@click.option("--hidden_dim", default=256, type=int, help="Hidden dimension size (matches the attention models for a fair comparison)")
@click.option("--dropout", default=0.05, type=float, help="Dropout value")
@click.option("--batch_size", default=64, type=int, help="Batch size")
@click.option("--fold_index", default=None, type=int, help="Specify a single fold index to train (matches the same 10-fold split as train_transformer.py)")
@click.option("--checkpoint_interval", default=20, type=int, help="Epochs between checkpoints")
@click.option("-o", "--out", default="baseline_out", type=str, help="Path to save the output")
@click.option("-f", "--force", is_flag=True, default=False, help="Overwrite output directory if it exists")
@click.option("--device", default="cuda", type=str, help="cuda or cpu")
def main(x_path, y_path, mask_portion, epochs, lr, min_lr_ratio, hidden_dim, dropout,
         batch_size, fold_index, checkpoint_interval, out, force, device):
    if os.path.exists(out) and not force:
        raise Exception(f"Directory {out} already exists.")
    os.makedirs(out, exist_ok=True)
    logger.add(os.path.join(out, "trainer.log"), level="DEBUG")

    logger.info(f"Parameters: x_path={x_path}, y_path={y_path}, mask_portion={mask_portion}, epochs={epochs}, lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}, batch_size={batch_size}, fold_index={fold_index}")

    logger.info("Reading in data")
    X = pickle.load(open(x_path, "rb"))
    y = pickle.load(open(y_path, "rb"))
    protein_dim = list(X.values())[0].shape[1] - 3  # 3 = strand(2) + length(1)
    logger.info(f"Computed protein embedding dim: {protein_dim}")

    dataset = EmbeddingDataset(list(X.values()), list(y.values()), list(y.keys()), mask_portion=mask_portion)
    dataset.set_training(True)
    logger.info(f"Total dataset size: {len(dataset)} samples")
    num_classes = 9

    # Same n_splits/random_state as train_crossValidation in src/model_onehot.py,
    # so --fold_index N here is the exact same train/val genomes as fold N there.
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
        if fold_index is not None and fold != fold_index:
            continue
        logger.info(f"Starting fold {fold}")
        train_one_fold(
            fold, train_index, val_index, dataset, num_classes, hidden_dim, dropout,
            batch_size, epochs, lr, min_lr_ratio, device, out, protein_dim, checkpoint_interval,
        )

    logger.info("FINISHED! :D")


if __name__ == "__main__":
    main()

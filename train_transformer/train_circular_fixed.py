"""
Trains the "original" architecture but with the batch-padding circular-attention
fix (src/model_onehot_circular_fixed.py) instead of the original
CircularRelativePositionAttention. Everything else - sinusoidal PE, diagonal
attention penalty, hyperparameters, masking, K-fold split - is identical to
train_transformer.py's "original" runs, so this is directly comparable
fold-for-fold against sinusoidal_comparison_original_fold_N.

Self-contained (does not call train()/train_fold()/train_crossValidation() in
src/model_onehot.py) so the existing training pipeline and all its already-
completed results stay untouched and reproducible.
"""
import os
import pickle

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.model_onehot import (
    EmbeddingDataset,
    collate_fn,
    fourier_positional_encoding,
    combined_loss,
    calculate_accuracy,
    cosine_lr_scheduler,
    CircularRelativePositionAttention,
    CircularTransformerEncoderLayer,
    TransformerClassifierCircularRelativeAttention,
)

NUM_CLASSES = 9

# The three classes below are also defined in src/model_onehot_circular_fixed.py
# (kept there as the documented, standalone version of the fix). They're
# duplicated here rather than imported, the same way BaselineClassifier is
# duplicated in evaluate_phold_recovery.py/evaluate_generalization.py: `src`
# resolves to an installed copy in some environments rather than this repo's
# checkout, which breaks cross-module imports of newly-added files even though
# `from src.model_onehot import ...` (an already-existing, already-installed
# module) works fine. Inlining here guarantees this script doesn't depend on
# that resolution succeeding.


class CircularRelativePositionAttentionPerGenome(CircularRelativePositionAttention):
    """Same as CircularRelativePositionAttention, but the wraparound fold modulus
    is each sample's own true (unpadded) length - from src_key_padding_mask -
    instead of the padded batch seq_len. See src/model_onehot_circular_fixed.py
    for the full explanation of the batch-padding artifact this fixes."""

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


def train_one_fold(fold, train_index, val_index, dataset, hidden_dim, num_heads, num_layers,
                    dropout, batch_size, epochs, lr, min_lr_ratio, lambda_penalty, device,
                    save_path, protein_dim, checkpoint_interval):
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

    model = TransformerClassifierCircularRelativeAttentionPerGenome(
        input_dim=protein_dim, num_classes=NUM_CLASSES, num_heads=num_heads, num_layers=num_layers,
        hidden_dim=hidden_dim, lstm_hidden_dim=512, dropout=dropout, use_lstm=False,
        positional_encoding=fourier_positional_encoding, use_positional_encoding=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = epochs / 4
    scheduler = cosine_lr_scheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, min_lr_ratio=min_lr_ratio)
    scaler = GradScaler()

    metrics = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_model.pth")
    final_validation_attention, final_validation_weights = [], []
    final_validation_categories, final_validation_masks = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0.0, 0
        for embeddings, categories, masks, idx in train_loader:
            embeddings = embeddings.to(device).float()
            categories = categories.to(device).long()
            masks_dev = masks.to(device).float()
            src_key_padding_mask = (masks_dev != -2).bool()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                outputs, attn_weights = model(embeddings, src_key_padding_mask=src_key_padding_mask, idx=idx, return_attn_weights=True)
                loss, _ = combined_loss(outputs, categories, masks_dev, attn_weights, idx, src_key_padding_mask, lambda_penalty=lambda_penalty)
                accuracy = calculate_accuracy(outputs, categories, masks_dev, idx)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_correct += accuracy * len(idx)
            total_samples += len(idx)

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples

        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0.0, 0.0, 0
        epoch_categories, epoch_masks, epoch_weights, epoch_attn = [], [], [], []
        with torch.no_grad():
            for embeddings, categories, masks, idx in val_loader:
                embeddings = embeddings.to(device).float()
                categories = categories.to(device).long()
                masks_dev = masks.to(device).float()
                src_key_padding_mask = (masks_dev != -2).bool()

                with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                    outputs, attn_weights = model(embeddings, src_key_padding_mask=src_key_padding_mask, idx=idx, return_attn_weights=True)
                    loss, _ = combined_loss(outputs, categories, masks_dev, attn_weights, idx, src_key_padding_mask, lambda_penalty=lambda_penalty)
                    accuracy = calculate_accuracy(outputs, categories, masks_dev, idx)

                total_val_loss += loss.item()
                total_val_correct += accuracy * len(idx)
                total_val_samples += len(idx)

                if epoch == epochs - 1:
                    epoch_weights.append(outputs.float().cpu().detach().numpy())
                    epoch_attn.append(attn_weights.float().cpu().detach().numpy())
                    epoch_categories.append(categories.cpu().detach().numpy())
                    epoch_masks.append(masks_dev.cpu().detach().numpy())

        val_loss = total_val_loss / len(val_loader)
        val_acc = total_val_correct / total_val_samples

        fold_logger.info(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        metrics.append({"epoch": epoch, "training losses": train_loss, "validation losses": val_loss, "training accuracies": train_acc, "validation accuracies": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            fold_logger.info(f"New best validation loss: {best_val_loss:.4f}. Model saved to {best_model_path}")

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

        if epoch == epochs - 1:
            final_validation_weights = epoch_weights
            final_validation_attention = epoch_attn
            final_validation_categories = epoch_categories
            final_validation_masks = epoch_masks

    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(output_dir, "transformer_state_dict.pth"))

    with open(os.path.join(output_dir, "final_validation_weights.pkl"), "wb") as f:
        pickle.dump(final_validation_weights, f)
    with open(os.path.join(output_dir, "final_validation_attention.pkl"), "wb") as f:
        pickle.dump(final_validation_attention, f)
    with open(os.path.join(output_dir, "final_validation_categories.pkl"), "wb") as f:
        pickle.dump(final_validation_categories, f)
    with open(os.path.join(output_dir, "final_validation_masks.pkl"), "wb") as f:
        pickle.dump(final_validation_masks, f)

    fold_logger.info(f"Fold {fold} done. Best val loss: {best_val_loss:.4f}")


@click.command()
@click.option("--x_path", "-x", required=True, type=click.Path(exists=True))
@click.option("--y_path", "-y", required=True, type=click.Path(exists=True))
@click.option("--mask_portion", default=0.3, type=float)
@click.option("--epochs", default=50, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--min_lr_ratio", default=0.1, type=float)
@click.option("--hidden_dim", default=256, type=int)
@click.option("--num_heads", default=4, type=int)
@click.option("--num_layers", default=2, type=int)
@click.option("--dropout", default=0.05, type=float)
@click.option("--batch_size", default=64, type=int)
@click.option("--lambda_penalty", default=100.0, type=float)
@click.option("--fold_index", default=None, type=int)
@click.option("--checkpoint_interval", default=20, type=int)
@click.option("-o", "--out", default="circular_fixed_out", type=str)
@click.option("-f", "--force", is_flag=True, default=False)
@click.option("--device", default="cuda", type=str)
def main(x_path, y_path, mask_portion, epochs, lr, min_lr_ratio, hidden_dim, num_heads, num_layers,
         dropout, batch_size, lambda_penalty, fold_index, checkpoint_interval, out, force, device):
    if os.path.exists(out) and not force:
        raise Exception(f"Directory {out} already exists.")
    os.makedirs(out, exist_ok=True)
    logger.add(os.path.join(out, "trainer.log"), level="DEBUG")

    logger.info(f"Parameters: x_path={x_path}, y_path={y_path}, mask_portion={mask_portion}, epochs={epochs}, lr={lr}, hidden_dim={hidden_dim}, num_heads={num_heads}, num_layers={num_layers}, dropout={dropout}, batch_size={batch_size}, lambda_penalty={lambda_penalty}, fold_index={fold_index}")

    logger.info("Reading in data")
    X = pickle.load(open(x_path, "rb"))
    y = pickle.load(open(y_path, "rb"))
    protein_dim = list(X.values())[0].shape[1] - 3
    logger.info(f"Computed protein embedding dim: {protein_dim}")

    dataset = EmbeddingDataset(list(X.values()), list(y.values()), list(y.keys()), mask_portion=mask_portion)
    dataset.set_training(True)
    logger.info(f"Total dataset size: {len(dataset)} samples")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
        if fold_index is not None and fold != fold_index:
            continue
        logger.info(f"Starting fold {fold}")
        train_one_fold(
            fold, train_index, val_index, dataset, hidden_dim, num_heads, num_layers, dropout,
            batch_size, epochs, lr, min_lr_ratio, lambda_penalty, device, out, protein_dim, checkpoint_interval,
        )

    logger.info("FINISHED! :D")


if __name__ == "__main__":
    main()

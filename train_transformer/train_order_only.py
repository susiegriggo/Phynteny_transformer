"""
Order-only ablation: the exact same architecture as the "original" model
(TransformerClassifierCircularRelativeAttention, same circular attention,
same sinusoidal PE, same diagonal-attention-penalty loss) but with every
gene's protein embedding (ESM features) zeroed out before it reaches the
model. What's left for the model to use: gene order/position (via the
circular relative attention and sinusoidal PE), each neighbouring gene's
known category (via the one-hot context channel), and strand/length.

This is the mirror-image ablation to train_baseline.py: baseline removes
context and keeps the embedding; this removes the embedding and keeps
context/order. If this model predicts function meaningfully above chance,
that's direct evidence gene order/synteny carries real signal on its own,
independent of any argument about whether attention-weight visualisations or
agreement with phold are trustworthy.

Architecturally unchanged from src/model_onehot.py - this script only zeroes
input values before they reach the model, it does not modify the model
definition at all.
"""
import os
import pickle

import click
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.model_onehot import (
    EmbeddingDataset,
    collate_fn,
    TransformerClassifierCircularRelativeAttention,
    fourier_positional_encoding,
    combined_loss,
    calculate_accuracy,
    cosine_lr_scheduler,
)

NUM_CLASSES = 9


def zero_protein_embedding(embeddings, num_classes):
    """embeddings layout: [one_hot(category) | strand(2) | length(1) | protein_embedding].
    Zero out only the protein embedding columns, in place on a clone so the
    original batch tensor from the DataLoader is untouched."""
    embeddings = embeddings.clone()
    embeddings[:, :, num_classes + 3:] = 0.0
    return embeddings


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

    model = TransformerClassifierCircularRelativeAttention(
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
            embeddings = zero_protein_embedding(embeddings, NUM_CLASSES).to(device).float()
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
                embeddings = zero_protein_embedding(embeddings, NUM_CLASSES).to(device).float()
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
@click.option("-o", "--out", default="order_only_out", type=str)
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
    logger.info(f"Computed protein embedding dim: {protein_dim} (will be zeroed before every forward pass)")

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

"""
Modules for building transformer model 
"""

import torch.nn as nn
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import numpy as np
from loguru import logger
import pandas as pd
import random
import os
import gc


class VariableSeq2SeqEmbeddingDataset(Dataset):
    def __init__(self, embeddings, categories, mask_token=-1, mask_portion=0.15):
        self.embeddings = [embedding.float() for embedding in embeddings]
        self.categories = [category.long() for category in categories]
        self.mask_token = mask_token
        self.num_classes = 10  # hard coded in for now
        self.mask_portion = mask_portion

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        category = self.categories[idx]
        mask = (
            category != self.mask_token
        ).float()  # 1 for valid, 0 for missing - this indicates which rows to consider when training

        # select a random cateogory to mask if training
        if self.training:
            masked_category, idx = self.category_mask(category, mask)
        else:
            masked_category = category

        # use the masked category to generate a one-hot encoding
        one_hot = self.custom_one_hot_encode(masked_category)

        # append the one_hot encoding to the front of the embedding
        embedding_one_hot = torch.tensor(np.hstack((one_hot, embedding)))

        self.embedding = embedding_one_hot

        return embedding_one_hot, category, mask, idx

    def set_training(self, training=True):
        self.training = training

    def category_mask(self, category, mask):
        # Define a probability distribution over maskable tokens
        probability_distribution = mask.float() / mask.sum()

        # Calculate the number of tokens to mask based on input length
        num_maskable_tokens = mask.sum().item()  # Count of maskable tokens
        num_tokens_to_mask = max(
            1, int(self.mask_portion * num_maskable_tokens)
        )  # Ensure at least 1 token is masked

        # Sample tokens to mask
        idx = torch.multinomial(
            probability_distribution, num_samples=num_tokens_to_mask, replacement=False
        )

        # generate masked versions
        masked_category = category.clone()
        masked_category[idx] = self.mask_token

        return masked_category, idx

    def shuffle_rows(self):
        """
        Shuffle rows within each instance in the dataset.
        This modifies the code in place
        """

        zipped_data = list(zip(self.embeddings, self.categories))
        random.shuffle(zipped_data)
        self.embeddings, self.categories = zip(*zipped_data)
        self.embeddings = list(self.embeddings)
        self.categories = list(self.categories)

    def custom_one_hot_encode(self, data):
        # Create the one-hot encoded array
        one_hot_encoded = []
        for value in data:
            if value == -1:
                one_hot_encoded.append([0] * self.num_classes)
            else:
                one_hot_row = [0] * self.num_classes
                one_hot_row[value] = 1
                one_hot_encoded.append(one_hot_row)

        return np.array(one_hot_encoded)


class Seq2SeqTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        lstm_hidden_dim=512,
        dropout=0.1,
    ):
        super(Seq2SeqTransformerClassifier, self).__init__()
        """
        Difference is between the position of the LSTM layer and the the type of positional encoding that is used
        """

        # check if cuda is available 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Embedding layer
        self.embedding_layer = nn.Linear(input_dim, hidden_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)  # think about where this layer goes

        # Positional Encoding (now learnable) -  could try using the fixed sinusoidal embeddings instead
        self.positional_encoding = nn.Parameter(
            torch.zeros(1000, hidden_dim)
        ).to(device)  # is zeroes good

        # normalisation layer??

        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
        ).to(device)

        # Transformer Encoder
        # encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True).cuda()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2, nhead=num_heads, batch_first=True
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        ).to(device)

        # Final Classification Layer
        self.fc = nn.Linear(2 * lstm_hidden_dim, num_classes).to(device)

    def forward(self, x, src_key_padding_mask=None, return_attn_weights=False):
        x = x.float()
        x = self.embedding_layer(x)
        x = x + self.positional_encoding[: x.size(1), :]

        x = self.dropout(x)
        x, _ = self.lstm(x)  # LSTM layer
        if return_attn_weights:
            x, attn_weights = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True)
            x = self.fc(x)
            return x, attn_weights
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            return x


class RelativePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1000, batch_first=True):
        super(RelativePositionAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first

        # Relative position encodings
        self.relative_position_k = nn.Parameter(
            torch.randn(max_len, d_model // num_heads)
        )
        self.relative_position_v = nn.Parameter(
            torch.randn(max_len, d_model // num_heads)
        )

    def forward(self, query, key, value, attn_mask=None):
        if not self.batch_first:
            # If batch is not first, transpose the batch and sequence dimensions
            query, key, value = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )

        batch_size, seq_len, d_model = query.size()

        # Reshape for multi-head attention
        q = query.view(
            batch_size, seq_len, self.num_heads, d_model // self.num_heads
        ).transpose(1, 2)
        k = key.view(
            batch_size, seq_len, self.num_heads, d_model // self.num_heads
        ).transpose(1, 2)
        v = value.view(
            batch_size, seq_len, self.num_heads, d_model // self.num_heads
        ).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
            d_model // self.num_heads
        )

        # Add relative position biases
        rel_positions = self.relative_position_k[:seq_len, :]
        scores = scores + torch.matmul(q, rel_positions.transpose(-2, -1))

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        rel_positions_v = self.relative_position_v[:seq_len, :]
        attn_output = attn_output + torch.matmul(attn_weights, rel_positions_v)

        # Reshape back to the original shape
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        if not self.batch_first:
            # If batch is not first, transpose back the batch and sequence dimensions
            attn_output = attn_output.transpose(0, 1)

        return attn_output


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, num_heads, dim_feedforward=512, dropout=0.1, max_len=1000
    ):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = RelativePositionAttention(
            d_model, num_heads, max_len=max_len, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, return_attn_weights=False):
        if return_attn_weights:
            src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src, attn_weights
        else:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src


class Seq2SeqTransformerClassifierRelativeAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        lstm_hidden_dim=512,
        dropout=0.1,
        max_len=1000,
    ):
        super(Seq2SeqTransformerClassifierRelativeAttention, self).__init__()


        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.embedding_layer = nn.Linear(input_dim, hidden_dim).to(device)
        self.dropout = nn.Dropout(
            dropout
        ).to(device)  # not sure if this is best spot and is a normalisation layer might be helpful
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim)).to(device)
        self.lstm = nn.LSTM(
            hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
        ).to(device)

        encoder_layers = CustomTransformerEncoderLayer(
            d_model=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        ).to(device)

        self.fc = nn.Linear(2 * lstm_hidden_dim, num_classes).to(device)

    def forward(self, x, src_key_padding_mask=None, return_attn_weights=False):
        x = x.float()
        x = self.embedding_layer(x)
        x = x + self.positional_encoding[: x.size(1), :]

        x = self.dropout(x)
        x, _ = self.lstm(x)
        if return_attn_weights:
            x, attn_weights = self.transformer_encoder.layers[0](x, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True)
            for layer in self.transformer_encoder.layers[1:]:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            return x, attn_weights
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            return x


class CircularRelativePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1000, batch_first=True):
        super(CircularRelativePositionAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first

        # Relative position encodings # this to intialise with random numbers 
        #self.relative_position_k = nn.Parameter(
        #    torch.randn(max_len, d_model // num_heads)
        #)
        #self.relative_position_v = nn.Parameter(
        #    torch.randn(max_len, d_model // num_heads)
        #)

        # Relative position encodings - intialise at zero - better to zero inialise or one intialise 
        self.relative_position_k = nn.Parameter(
            torch.zeros(max_len, d_model // num_heads)
        )
        self.relative_position_v = nn.Parameter(
            torch.zeros(max_len, d_model // num_heads)
        )

    def forward(self, query, key, value, attn_mask=None, is_causal=False, output_dir=None, batch_idx=None, return_attn_weights=False):
        if not self.batch_first:
            query, key, value = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )

        batch_size, seq_len, d_model = query.size()

        # Reshape for multi-head attention
        q = query.view(
            batch_size, seq_len, self.num_heads, d_model // self.num_heads
        ).transpose(1, 2)
        k = key.view(
            batch_size, seq_len, self.num_heads, d_model // self.num_heads
        ).transpose(1, 2)
        v = value.view(
            batch_size, seq_len, self.num_heads, d_model // self.num_heads
        ).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
            d_model // self.num_heads
        )

        # Circular relative position
        circular_indices = torch.arange(seq_len).unsqueeze(1) - torch.arange(
            seq_len
        ).unsqueeze(0)
        circular_indices = (circular_indices + seq_len) % seq_len  # Circular difference
        circular_indices = torch.min(
            circular_indices, seq_len - circular_indices
        )  # Min of direct and wrap-around distance

        # Adjust dimensions for broadcasting
        rel_positions_k = (
            self.relative_position_k[circular_indices]
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
        )
        scores += torch.einsum("bhqd,bqkd->bhqk", q, rel_positions_k[0])  # error here

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Adjust dimensions for broadcasting
        rel_positions_v = (
            self.relative_position_v[circular_indices]
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
        )
        attn_output += torch.einsum("bhqk,bqkd->bhqd", attn_weights, rel_positions_v[0])

        # Reshape back to the original shape
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # Save attention weights to a file if output_dir is provided
        if output_dir is not None and batch_idx is not None:
            attn_weights_path = os.path.join(output_dir, f"attention_weights_batch_{batch_idx}.pkl")
            with open(attn_weights_path, "wb") as f:
                pickle.dump(attn_weights.cpu().detach().numpy(), f)

        if return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output


class CircularTransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, num_heads, dim_feedforward=512, dropout=0.1, max_len=1000
    ):
        super(CircularTransformerEncoderLayer, self).__init__()
        self.self_attn= CircularRelativePositionAttention(
            d_model, num_heads, max_len=max_len, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, return_attn_weights=False):
        if return_attn_weights: 
            src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, is_causal=is_causal, return_attn_weights=return_attn_weights)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src, attn_weights
        else: 
            src2 = self.self_attn(src, src, src, attn_mask=src_mask, is_causal=is_causal)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src
            

class Seq2SeqTransformerClassifierCircularRelativeAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        lstm_hidden_dim=512,
        dropout=0.1,
        max_len=1000,
    ):
        super(Seq2SeqTransformerClassifierCircularRelativeAttention, self).__init__()

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.embedding_layer = nn.Linear(input_dim, hidden_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim, device=device)).to(device) # I think this here is an abosolution position 
        self.lstm = nn.LSTM(
            hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
        ).to(device)

        encoder_layers = CircularTransformerEncoderLayer(
            d_model=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        ).to(device)

        self.fc = nn.Linear(2 * lstm_hidden_dim, num_classes).to(device)

    def forward(self, x, src_key_padding_mask=None, return_attn_weights=False):
        x = x.float()
        x = self.embedding_layer(x)
        x = x + self.positional_encoding[: x.size(1), :]
        x = self.dropout(x)
        x, _ = self.lstm(x)
        if return_attn_weights:
            x, attn_weights = self.transformer_encoder.layers[0](x, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True)
            for layer in self.transformer_encoder.layers[1:]:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            return x, attn_weights
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            return x

def masked_loss(output, target, mask, idx, ignore_index=-1):
    # Loss function focused on predicting the specific category
    # Temp that doesn't work with batched data. Come back to this later 

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = target.clone()
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none").to(device)
    loss = loss_fct(output.view(-1, output.size(-1)), target.view(-1))
    return loss[idx].sum() / len(idx[0])


def masked_loss(output, target, mask, idx, ignore_index=-1):
    # Assuming `output` is of shape [batch_size, seq_len, num_classes]
    batch_size, seq_len, num_classes = output.shape

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Flatten the batch and sequence dimensions
    output_flat = output.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
    target_flat = target.view(-1)  # [batch_size * seq_len]
    mask_flat = mask.view(-1)  # [batch_size * seq_len]

    # Only consider the elements at the specific `idx` positions
    idx_overall = [ii + val * seq_len for val, ii in enumerate(idx)] # adjust index for the flatten batch
    idx_flat = torch.cat([t.flatten() for t in idx_overall]) # flatten indexes 

    # Apply the mask (where mask == 0, set ignore_index in target)
    target_flat[mask_flat == 0] = ignore_index

    # Calculate loss using CrossEntropyLoss (ignoring masked positions)
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none").to(device)
    loss = loss_fct(output_flat, target_flat)

    # Return the loss only for the specific indices
    return loss[idx_flat].sum() / len(idx_flat)


def collate_fn(batch):
    embeddings, categories, masks, idx = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    categories_padded = pad_sequence(categories, batch_first=True, padding_value=-1)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=-2)
    return embeddings_padded, categories_padded, masks_padded, idx

def loss(output, target, mask, ignore_index=-1):
    # this loss function should be looking at the predicting gene not everything
    target = target.clone()
    target[mask == 0] = ignore_index
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none").to(device)
    loss = loss_fct(output.view(-1, output.size(-1)), target.view(-1))
    loss = loss * mask.view(-1)
    return loss.sum() / mask.sum()


def cosine_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """ 
    From glm2 paper - thanks George for sharing! 
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            logger.info('warmup step')
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps))))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train(
    model,
    train_dataloader,
    test_dataloader,
    epochs=10,
    step_size=10, 
    gamma=0.1,
    lr=1e-5,
    save_path="model",
    device="cuda",
    checkpoint_interval=1
):
    logger.info("Training on " + str(device))
    model.to(device)

    #optimizer = optim.Adam(model.parameters(), lr=lr) 
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1) #adamW optimizer from the glm2 paper 

    # original scheduler 
    #print(f'Using Step LR', flush=True)
    #print(f'Learning rate: {lr}', flush=True)
    #print(f'Step size: {step_size}', flush=True)
    #print(f'Gamma: {gamma}', flush=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # original scheduler 


    #scheduler = optim.lr_scheduler.OneCycleLR( # this scheudler uses a warmup - might be a good comparison 
    #    optimizer, 
    #    max_lr=lr, 
    #    total_steps=len(train_dataloader)*epochs,
    #    epochs=epochs, 
    #    pct_start=0.3
    #)

    # alternative scheduler - from glm2 
    num_training_steps = len(train_dataloader)*epochs,
    num_training_steps = epochs 
    num_warmup_steps = epochs/4
    print(f'num_training_steps {num_training_steps}', flush=True )
    scheduler = cosine_lr_scheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    final_validation_weights = []
    final_validation_attention = [] 

    logger.info("Beginning training loop")
    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode
        total_loss = 0
        for embeddings, categories, masks, idx in train_dataloader:
            embeddings, categories, masks = (
                embeddings.to(device).float(),
                categories.to(device).long(),
                masks.to(device).float(),
            )
            optimizer.zero_grad()
            
            # Mask the padding from the transformer 
            src_key_padding_mask = (masks == -2).bool()
           
            with autocast():
                outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                loss = masked_loss(
                    outputs, categories, masks, idx
                )  # need to specify which index to reference in the loss function

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss}")

        # Clear cache after training epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Validation loss
        model.eval()  # Ensure model is in evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for embeddings, categories, masks, idx in test_dataloader:
                embeddings, categories, masks = (
                    embeddings.to(device).float(),
                    categories.to(device).long(),
                    masks.to(device).float(),
                )
               
                src_key_padding_mask = (masks == -2).bool()  # Mask the padding from the transformer 
                
                with autocast():
                    if epoch == epochs - 1:
                        outputs, attn_weights = model(
                            embeddings, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True
                        )
                        final_validation_weights.append(outputs.cpu().detach().numpy())
                        final_validation_attention.append(attn_weights.cpu().detach().numpy())
                    else:
                        outputs = model(
                            embeddings, src_key_padding_mask=src_key_padding_mask
                        )
                    val_loss = masked_loss(
                        outputs, categories, masks, idx
                    )  # need to include idx in the loss function
                total_val_loss += val_loss.item()



        avg_val_loss = total_val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Clear cache after validation epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Update scheduler
        scheduler.step()
        logger.info(f'learning rate {scheduler.get_last_lr()}', flush=True)

        # save checkpoint every 'checkpoint interval' epochs 
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save the model
    torch.save(model.state_dict(), save_path + "transformer.model")

    # Save the training and validation loss to CSV
    loss_df = pd.DataFrame(
        {
            "epoch": [i for i in range(epochs)],
            "training losses": train_losses,
            "validation losses": val_losses,
        }
    )
    output_dir = f"{save_path}"
    loss_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # Save the final validation weights and attention weights
    with open(os.path.join(save_path, "final_validation_weights.pkl"), "wb") as f:
        pickle.dump(final_validation_weights, f)

    with open(os.path.join(save_path, "final_validation_attention.pkl"), "wb") as f:
        pickle.dump(final_validation_attention, f)

def train_crossValidation(
    dataset,
    attention,
    n_splits=10,
    batch_size=16,
    epochs=10,
    lr=1e-5,
    step_size=10, 
    gamma=0.1,
    save_path="out",
    num_heads=4,
    hidden_dim=512,
    device="cuda",
    dropout=0.1,
    checkpoint_interval=1
):
    # access the logger object
    logger.add(save_path + "trainer.log", level="DEBUG")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    logger.info(
        "Training with K-Fold crossvalidation with " + str(n_splits) + " folds..."
    )

    for train_index, val_index in kf.split(dataset):
        logger.info(f"FOLD: {fold}")

        # Create output directory for the current fold
        logger.info("Creating output directory")
        output_dir = f"{save_path}/fold_{fold}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Directory created: {output_dir}")
        else:
            logger.info(f"Warning: Directory {output_dir} already exists.")

        # Use SubsetRandomSampler for efficient subsetting
        logger.info("generating subsamples")
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)

        # Create data loaders with samplers
        logger.info("Saving datasets")
        train_kfold_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        val_kfold_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # save the validation data object as well as the keys used in validtaiokn 
        pickle.dump(val_kfold_loader, open(output_dir + "/val_kfold_loader.pkl", "wb"))
        with open(output_dir + "/val_kfold_keys.txt", "w") as file: 
            for k in val_index: 
                file.write(f"{k}\n")

        # Initialize model
        input_dim = dataset[0][0].shape[1]
        if attention == "circular":
            kfold_transformer_model = (
                Seq2SeqTransformerClassifierCircularRelativeAttention(
                    input_dim=input_dim,
                    num_classes=9,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            )
        elif attention == "relative":
            kfold_transformer_model = Seq2SeqTransformerClassifierRelativeAttention(
                input_dim=input_dim,
                num_classes=9,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif attention == "absolute":
            kfold_transformer_model = Seq2SeqTransformerClassifier(
                input_dim=input_dim,
                num_classes=9,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            logger.error("invalid attention type specified")

        # kfold_transformer_model = ImprovedSeq2SeqTransformerClassifier(input_dim=input_dim, num_classes=9, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
        kfold_transformer_model.to(device)

        # Train model
        logger.info("Training the model")
        train(
            kfold_transformer_model,
            train_kfold_loader,
            val_kfold_loader,
            epochs=epochs,
            lr=lr,
            step_size=step_size, 
            gamma=gamma, 
            save_path=output_dir,
            device=device,
        )

        # Clear cache after training each fold
        gc.collect()
        torch.cuda.empty_cache()

        fold += 1


def evaluate(model, dataloader, phrog_integer, device, output_dir="metrics_output"):
    ## This here needs work. Could be done outside of this function
    model.to(device)
    model.eval()
    labels = list(phrog_integer.keys())
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for embeddings, categories, masks, idx in dataloader:
            embeddings, categories, masks = (
                embeddings.to(device).float(),
                categories.to(device).long(),
                masks.to(device).float(),
            )
            src_key_padding_mask = masks == 0
            outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)

            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=2).detach().cpu().numpy()
            categories = categories.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            for i in range(probs.shape[0]):  # Iterate over the batch
                valid_indices = np.where(masks[i] == 1)[0]
                for idx in valid_indices:
                    all_labels.append(categories[i][idx])
                    all_probs.append(probs[i][idx])

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    for ii, label in enumerate(labels):
        roc_probs = all_probs[:, ii]
        roc_labels = (all_labels == ii).astype(int)
        fpr, tpr, thresholds = roc_curve(roc_labels, roc_probs)
        roc_auc = auc(fpr, tpr)
        print("AUC for category " + phrog_integer[label] + ": " + str(roc_auc))

        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Thresholds": thresholds})
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        roc_df.to_csv(
            os.path.join(output_dir, f"roc_curve_category_{phrog_integer[label]}.csv"),
            index=False,
        )

    # Compute F1 score, precision, recall for each category
    all_preds = np.argmax(all_probs, axis=1)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=1)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=1)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=1)

    print("f1")
    print


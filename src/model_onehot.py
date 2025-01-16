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
import torch.multiprocessing as mp
import resource

# Increase the limit of open files
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft * 2, hard), hard))

# Set the sharing strategy to file_system
torch.multiprocessing.set_sharing_strategy('file_system')


class VariableSeq2SeqEmbeddingDataset(Dataset):
    def __init__(self, embeddings, categories, mask_token=-1, mask_portion=0.15):
        """
        Initialize the dataset with embeddings and categories.

        Parameters:
        embeddings (list of torch.Tensor): List of embedding tensors.
        categories (list of torch.Tensor): List of category tensors.
        mask_token (int): Token used for masking.
        mask_portion (float): Portion of tokens to mask during training.
        """
        try:
            self.embeddings = [embedding.float() for embedding in embeddings]
            self.categories = [category.long() for category in categories]
            self.mask_token = mask_token
            self.num_classes = 10  # hard coded in for now
            self.mask_portion = mask_portion
        except Exception as e:
            logger.error(f"Error initializing VariableSeq2SeqEmbeddingDataset: {e}")
            raise

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
        int: Number of samples.
        """
        try:
            return len(self.embeddings)
        except Exception as e:
            logger.error(f"Error getting length of dataset: {e}")
            raise

    def __getitem__(self, idx):
        """
        Get a single viruses from the dataset at the specified index.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        tuple: Tuple containing the embedding, category, mask, and masked indices.
        """
        try:
            embedding = self.embeddings[idx]
            category = self.categories[idx]
            mask = (
                category != self.mask_token
            ).float()  # 1 for valid, 0 for missing - this indicates which rows to consider when training

            # select a random category to mask if training
            if self.training:
                masked_category, idx = self.category_mask(category, mask)
            else:
                masked_category = category

            # use the masked category to generate a one-hot encoding
            one_hot = self.custom_one_hot_encode(masked_category)

            # append the one_hot encoding to the front of the embedding
            embedding_one_hot = torch.tensor(np.hstack((one_hot, embedding)))

            return embedding_one_hot, category, mask, idx
        except Exception as e:
            logger.error(f"Error getting item from dataset at index {idx}: {e}")
            raise

    def set_training(self, training=True):
        """
        Set the training mode for the dataset.

        Parameters:
        training (bool): If True, the dataset is in training mode.
        """
        self.training = training

    def category_mask(self, category, mask):
        """
        Mask a portion of the category tokens.

        Parameters:
        category (torch.Tensor): Category tensor.
        mask (torch.Tensor): Mask tensor.

        Returns:
        tuple: Tuple containing the masked category and masked indices.
        """
        try:
            # Check for NaN or infinite values in the mask tensor
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                logger.error(f"NaN or infinite values found in mask tensor at index: {idx}")

            # Define a probability distribution over maskable tokens
            #logger.info(f'mask.sum: {mask.sum()}')
            #logger.info(f'mask.float: {mask.float()}')
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
        except Exception as e:
            logger.error(f"Error in category_mask: {e}")
            raise

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
        """
        Generate a one-hot encoding for the given data.

        Parameters:
        data (torch.Tensor): Data tensor to encode.

        Returns:
        np.array: One-hot encoded array.
        """
        try:
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
        except Exception as e:
            logger.error(f"Error in custom_one_hot_encode: {e}")
            raise


class Seq2SeqTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        dropout=0.1,
        intialisation='random',
        output_dim=None  # Add output_dim parameter
    ):
        """
        Initialize the Seq2Seq Transformer Classifier.

        Parameters:
        input_dim (int): Input dimension size.
        num_classes (int): Number of output classes.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        hidden_dim (int): Hidden dimension size.
        dropout (float): Dropout rate.
        intialisation (str): Initialization method for positional encoding ('random' or 'zero').
        """
        super(Seq2SeqTransformerClassifier, self).__init__()
        """
        Difference is between the position of the LSTM layer and the the type of positional encoding that is used
        """

        # check if cuda is available 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layers
        self.func_embedding = nn.Embedding(10, 16).to(device)
        self.strand_embedding = nn.Embedding(2, 4).to(device)
        self.length_embedding = nn.Linear(1, 8).to(device)
        self.embedding_layer = nn.Linear(1280, hidden_dim - 28).to(device)

        self.dropout = nn.Dropout(dropout).to(device)  

        # Positional Encoding (now learnable) -  could try using the fixed sinusoidal embeddings instead
        logger.info(f"Initialising positional encoding with {intialisation} values")
        if intialisation == 'random':
            self.positional_encoding = nn.Parameter(
                torch.randn(1000, hidden_dim)
            ).to(device)
        elif intialisation == 'zeros':
            self.positional_encoding = nn.Parameter(
                torch.zeros(1000, hidden_dim) 
            ).to(device)
        else: 
            ValueError(f"Invalid initialization value: {intialisation}. Must be 'random' or 'zeros'.")

        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        ).to(device)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2, nhead=num_heads, batch_first=True
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        ).to(device)

        # Final Classification Layer
        self.output_dim = output_dim if output_dim else num_classes  # Set output_dim
        self.fc = nn.Linear(2 * hidden_dim, self.output_dim).to(device)  # Use output_dim

    def forward(self, x, src_key_padding_mask=None, return_attn_weights=False):
        """
        Forward pass of the model.

        Parameters:
        x (torch.Tensor): Input tensor.
        src_key_padding_mask (torch.Tensor, optional): Mask tensor for padding.
        return_attn_weights (bool, optional): If True, return attention weights.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = x.float()
        func_ids, strand_ids, gene_length, protein_embeds = x[:,:,:10], x[:,:,10:12], x[:,:,12:13], x[:,:,13:]
        func_embeds = self.func_embedding(func_ids.argmax(-1))
        strand_embeds = self.strand_embedding(strand_ids.argmax(-1))
        length_embeds = self.length_embedding(gene_length)
        protein_embeds = self.embedding_layer(protein_embeds)

        x = torch.cat([func_embeds, strand_embeds, length_embeds, protein_embeds], dim=-1)
        x = x + self.positional_encoding[: x.size(1), :].to(x.device)

        x = self.dropout(x)
        x, _ = self.lstm(x)  # LSTM layer
        if return_attn_weights:
            x, attn_weights = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True)
            x = self.fc(x)
            if self.output_dim != 9:  # Apply softmax if output_dim is not 9
                x = F.softmax(x, dim=-1)
            return x, attn_weights
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            if self.output_dim != 9:  # Apply softmax if output_dim is not 9
                x = F.softmax(x, dim=-1)
            return x

# Add logging to check the device of various components
def log_model_devices(model):
    logger.info(f"func_embedding is on device: {model.func_embedding.weight.device}")
    logger.info(f"strand_embedding is on device: {model.strand_embedding.weight.device}")
    logger.info(f"length_embedding is on device: {model.length_embedding.weight.device}")
    logger.info(f"embedding_layer is on device: {model.embedding_layer.weight.device}")
    logger.info(f"positional_encoding is on device: {model.positional_encoding.device}")
    logger.info(f"lstm is on device: {next(model.lstm.parameters()).device}")
    logger.info(f"transformer_encoder is on device: {next(model.transformer_encoder.parameters()).device}")
    logger.info(f"fc is on device: {model.fc.weight.device}")

class RelativePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1000, batch_first=True, intialisation = 'random'):
        """
        Initialize the Relative Position Attention module.

        Parameters:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length.
        batch_first (bool): If True, the batch dimension is the first dimension.
        intialisation (str): Initialization method for relative position encodings ('random' or 'zero').
        """
        super(RelativePositionAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first

        # Relative position encodings
        if intialisation == 'random':
            self.relative_position_k = nn.Parameter(
                torch.randn(max_len, d_model // num_heads)
            )
            self.relative_position_v = nn.Parameter(
                torch.randn(max_len, d_model // num_heads)
            )
        elif intialisation == 'zeros':
            self.relative_position_k = nn.Parameter(
                torch.zeros(max_len, d_model // num_heads)
            )               
            self.relative_position_v = nn.Parameter(
                torch.zeros(max_len, d_model // num_heads)
            )           
        else:       
            ValueError(f"Invalid initialization value: {intialisation}. Must be 'random' or 'zeros'.")

    def forward(self, query, key, value, attn_mask=None, return_attn_weights=False, src_key_padding_mask=None):
        """
        Forward pass of the relative position attention.

        Parameters:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        attn_mask (torch.Tensor, optional): Attention mask tensor.
        return_attn_weights (bool, optional): If True, return attention weights.

        Returns:
        torch.Tensor: Output tensor.
        """
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
        
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(3) == 0, float("-inf"))

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

        if return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, num_heads, dim_feedforward=512, dropout=0.1, max_len=1000, intialisation='random'
    ):
        """
        Initialize the Custom Transformer Encoder Layer.

        Parameters:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        max_len (int): Maximum sequence length.
        intialisation (str): Initialization method for relative position encodings ('random' or 'zero').
        """
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = RelativePositionAttention( # is this called by other methods? 
            d_model, num_heads, max_len=max_len, batch_first=True, intialisation=intialisation
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, return_attn_weights=False):
        """
        Forward pass of the custom transformer encoder layer.

        Parameters:
        src (torch.Tensor): Source tensor.
        src_mask (torch.Tensor, optional): Source mask tensor.
        src_key_padding_mask (torch.Tensor, optional): Source key padding mask tensor.
        is_causal (bool, optional): If True, use causal masking.
        return_attn_weights (bool, optional): If True, return attention weights.

        Returns:
        torch.Tensor: Output tensor.
        """
        if return_attn_weights:
            src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, src_key_padding_mask=src_key_padding_mask, return_attn_weights=return_attn_weights) # this is a change here that I made 
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src, attn_weights
        else:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
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
        dropout=0.1,
        max_len=1000,
        intialisation='random',
        output_dim=None  # Add output_dim parameter
    ):
        """
        Initialize the Seq2Seq Transformer Classifier with Relative Attention.

        Parameters:
        input_dim (int): Input dimension size.
        num_classes (int): Number of output classes.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        hidden_dim (int): Hidden dimension size.
        dropout (float): Dropout rate.
        max_len (int): Maximum sequence length.
        intialisation (str): Initialization method for positional encoding ('random' or 'zero').
        """
        super(Seq2SeqTransformerClassifierRelativeAttention, self).__init__()

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layers
        self.func_embedding = nn.Embedding(10, 16).to(device)
        self.strand_embedding = nn.Embedding(2, 4).to(device)
        self.length_embedding = nn.Linear(1, 8).to(device)
        self.embedding_layer = nn.Linear(1280, hidden_dim - 28).to(device) #maybe this minus 28 is causing problems 

        self.dropout = nn.Dropout(dropout).to(device)
        self.positional_encoding = sinusoidal_positional_encoding(max_len, hidden_dim, device).to(device)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        ).to(device)

        encoder_layers = CustomTransformerEncoderLayer(
            d_model=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len,
            intialisation=intialisation
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        ).to(device)

        self.output_dim = output_dim if output_dim else num_classes  # Set output_dim
        self.fc = nn.Linear(2 * hidden_dim, self.output_dim).to(device)  # Use output_dim

    def forward(self, x, src_key_padding_mask=None, return_attn_weights=False):
        """
        Forward pass of the model.

        Parameters:
        x (torch.Tensor): Input tensor.
        src_key_padding_mask (torch.Tensor, optional): Mask tensor for padding.
        return_attn_weights (bool, optional): If True, return attention weights.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = x.float()
        func_ids, strand_ids, gene_length, protein_embeds = x[:,:,:10], x[:,:,10:12], x[:,:,12:13], x[:,:,13:]
        func_embeds = self.func_embedding(func_ids.argmax(-1))
        strand_embeds = self.strand_embedding(strand_ids.argmax(-1))
        length_embeds = self.length_embedding(gene_length)
        protein_embeds = self.embedding_layer(protein_embeds)

        x = torch.cat([func_embeds, strand_embeds, length_embeds, protein_embeds], dim=-1)
        x = x + self.positional_encoding[: x.size(1), :].to(x.device)

        x = self.dropout(x)
        x, _ = self.lstm(x)
        if return_attn_weights:
            x, attn_weights = self.transformer_encoder.layers[0](x, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True)
            for layer in self.transformer_encoder.layers[1:]:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            if self.output_dim != 9:  # Apply softmax if output_dim is not 9
                x = F.softmax(x, dim=-1)
            return x, attn_weights
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            if self.output_dim != 9:  # Apply softmax if output_dim is not 9
                x = F.softmax(x, dim=-1)
            return x


class CircularRelativePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1000, batch_first=True, intialisation = 'random'):
        """
        Initialize the Circular Relative Position Attention module.

        Parameters:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length.
        batch_first (bool): If True, the batch dimension is the first dimension.
        intialisation (str): Initialization method for relative position encodings ('random' or 'zero').
        """
        super(CircularRelativePositionAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first

        # Relative position encodings # this to intialise with random numbers 
        if intialisation == 'random':
            self.relative_position_k = nn.Parameter(
                torch.randn(max_len, d_model // num_heads)
            ) 
            self.relative_position_v = nn.Parameter(
                torch.randn(max_len, d_model // num_heads)
            )  

        elif intialisation == 'zeros':
            self.relative_position_k = nn.Parameter(
                torch.zeros(max_len, d_model // num_heads)
            ) 
            self.relative_position_v = nn.Parameter(
                torch.zeros(max_len, d_model // num_heads)
            )
        else:
            raise ValueError(f"Invalid initialization value: {intialisation}. Must be 'random' or 'zeros'.")
            
    def forward(self, query, key, value, attn_mask=None, src_key_padding_mask=None, is_causal=False, output_dir=None, batch_idx=None, return_attn_weights=False):
        """
        Forward pass of the circular relative position attention.

        Parameters:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        attn_mask (torch.Tensor, optional): Attention mask tensor.
        is_causal (bool, optional): If True, use causal masking.
        output_dir (str, optional): Directory to save attention weights.
        batch_idx (int, optional): Batch index for saving attention weights.
        return_attn_weights (bool, optional): If True, return attention weights.

        Returns:
        torch.Tensor: Output tensor.
        """
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
        scores += torch.einsum("bhqd,bqkd->bhqk", q, rel_positions_k[0])

        if attn_mask is not None:
            logger.info(f"attn_mask: {attn_mask}")  
            scores = scores.masked_fill(attn_mask == 0, float("-inf")) 

        if src_key_padding_mask is not None:
            # apply the mask in both direction 
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf")) #mask the padded rows 
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(3) == 0, float("-inf")) #mask the padded columns

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Mask the resulting attention weights to ensure padded positions have zero attention
        if src_key_padding_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0) # this is a bit cheeky 

        attn_output = torch.matmul(attn_weights, v)
        #logger.info(f"attn_weights shape: {attn_weights.shape}")
        #logger.info(f"attn_output shape: {attn_output.shape}") 
        #logger.info(f"attn_output: {attn_output}")
        #logger.info(f"attn_weights: {attn_weights}")
      

        # Adjust dimensions for broadcasting
        rel_positions_v = (
            self.relative_position_v[circular_indices]
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
        )
        attn_output += torch.einsum("bhqk,bqkd->bhqd", attn_weights, rel_positions_v[0])

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

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
        self, d_model, num_heads, dim_feedforward=512, dropout=0.1, max_len=1000, initialisation='random'
    ):
        """
        Initialize the Circular Transformer Encoder Layer.

        Parameters:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        max_len (int): Maximum sequence length.
        initialisation (str): Initialization method for relative position encodings ('random' or 'zero').
        """
        super(CircularTransformerEncoderLayer, self).__init__()
        self.self_attn= CircularRelativePositionAttention(
            d_model, num_heads, max_len=max_len, batch_first=True,intialisation=initialisation
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, return_attn_weights=False):
        """
        Forward pass of the circular transformer encoder layer.

        Parameters:
        src (torch.Tensor): Source tensor.
        src_mask (torch.Tensor, optional): Source mask tensor.
        src_key_padding_mask (torch.Tensor, optional): Source key padding mask tensor.
        is_causal (bool, optional): If True, use causal masking.
        return_attn_weights (bool, optional): If True, return attention weights.

        Returns:
        torch.Tensor: Output tensor.
        """
        if return_attn_weights: 
            src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask, return_attn_weights=return_attn_weights)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src, attn_weights
        else: 
            src2 = self.self_attn(src, src, src, attn_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
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
        dropout=0.1,
        max_len=1000,
        intialisation='random',
        output_dim=None  # Add output_dim parameter
    ):
        """
        Initialize the Seq2Seq Transformer Classifier with Circular Relative Attention.

        Parameters:
        input_dim (int): Input dimension size.
        num_classes (int): Number of output classes.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        hidden_dim (int): Hidden dimension size.
        dropout (float): Dropout rate.
        max_len (int): Maximum sequence length.
        intialisation (str): Initialization method for positional encoding ('random' or 'zero').
        """
        super(Seq2SeqTransformerClassifierCircularRelativeAttention, self).__init__()

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # try adding some embedding layers 
        self.func_embedding = nn.Embedding(10, 16).to(device)
        self.strand_embedding = nn.Embedding(2, 4).to(device)
        self.length_embedding = nn.Linear(1,8).to(device) # linear embedding for the protein embedding

        # linear embedding for the protein embedding
        self.embedding_layer = nn.Linear(1280, hidden_dim -28).to(device)

        self.dropout = nn.Dropout(dropout).to(device)
        self.positional_encoding = sinusoidal_positional_encoding(max_len, hidden_dim, device).to(device)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        ).to(device)

        encoder_layers = CircularTransformerEncoderLayer(
            d_model=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len,
            initialisation=intialisation
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        ).to(device)

        self.output_dim = output_dim if output_dim else num_classes  # Set output_dim
        self.fc = nn.Linear(2 * hidden_dim, self.output_dim).to(device)  # Use output_dim

    def forward(self, x, src_key_padding_mask=None, return_attn_weights=False):
        """
        Forward pass of the model.

        Parameters:
        x (torch.Tensor): Input tensor.
        src_key_padding_mask (torch.Tensor, optional): Mask tensor for padding.
        return_attn_weights (bool, optional): If True, return attention weights.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = x.float()
        func_ids, strand_ids, gene_length, protein_embeds = x[:,:,:10], x[:,:,10:12], x[:,:,12:13], x[:,:,13:]
        func_embeds = self.func_embedding(func_ids.argmax(-1))
        strand_embeds = self.strand_embedding(strand_ids.argmax(-1))
        length_embeds = self.length_embedding(gene_length)
        protein_embeds = self.embedding_layer(protein_embeds)

        x = torch.cat([func_embeds, strand_embeds, length_embeds, protein_embeds], dim=-1)
        x = x + self.positional_encoding[: x.size(1), :].to(x.device)
        x = self.dropout(x)
        x, _ = self.lstm(x)

        if return_attn_weights:
            x, attn_weights = self.transformer_encoder.layers[0](x, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True)
            for layer in self.transformer_encoder.layers[1:]:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            if self.output_dim != 9:  # Apply softmax if output_dim is not 9
                x = F.softmax(x, dim=-1)
            return x, attn_weights
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            x = self.fc(x)
            if self.output_dim != 9:  # Apply softmax if output_dim is not 9
                x = F.softmax(x, dim=-1)
            return x


def masked_loss(output, target, mask, idx, ignore_index=-1):
    """
    Calculate the masked loss.

    Parameters:
    output (torch.Tensor): Output tensor from the model.
    target (torch.Tensor): Target tensor.
    mask (torch.Tensor): Mask tensor.
    idx (list of torch.Tensor): Indices of masked tokens.
    ignore_index (int, optional): Index to ignore in the loss calculation.

    Returns:
    torch.Tensor: Calculated loss.
    """
    # Assuming `output` is of shape [batch_size, seq_len, num_classes]
    batch_size, seq_len, num_classes = output.shape

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #logger.info(f"Output is on device: {output.device}")
    #logger.info(f"Target is on device: {target.device}")
    #logger.info(f"Mask is on device: {mask.device}")

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
    """
    Collate function for the DataLoader.

    Parameters:
    batch (list of tuples): Batch of data.

    Returns:
    tuple: Padded embeddings, categories, masks, and indices.
    """
    embeddings, categories, masks, idx = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    categories_padded = pad_sequence(categories, batch_first=True, padding_value=-1)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=-2)
    return embeddings_padded, categories_padded, masks_padded, idx

def combined_loss(output, target, mask, attn_weights, idx, src_key_padding_mask, lambda_penalty=0.1, ignore_index=-1):
    """
    Combine classification loss with a diagonal attention penalty.

    Parameters:
    output (torch.Tensor): Output tensor from the model.
    target (torch.Tensor): Target tensor.
    mask (torch.Tensor): Mask tensor.
    attn_weights (torch.Tensor): Attention weights tensor.
    idx (list of torch.Tensor): Indices of masked tokens.
    lambda_penalty (float, optional): Penalty for diagonal attention.
    ignore_index (int, optional): Index to ignore in the loss calculation.

    Returns:
    torch.Tensor: Combined loss.
    """
    # Standard masked loss
    batch_size, seq_len, num_classes = output.shape
    device = output.device

    #logger.info(f"Output is on device: {output.device}")
    #logger.info(f"Target is on device: {target.device}")
    #logger.info(f"Mask is on device: {mask.device}")
    #logger.info(f"Attention weights are on device: {attn_weights.device}")

    output_flat = output.view(-1, num_classes)
    target_flat = target.view(-1)
    mask_flat = mask.view(-1)

    # Adjust indices for flattened batch and sequence
    idx_overall = [ii + val * seq_len for val, ii in enumerate(idx)]
    idx_flat = torch.cat([t.flatten() for t in idx_overall])

    # Apply the mask
    target_flat[mask_flat == 0] = ignore_index

    # Compute classification loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none").to(device)
    classification_loss = loss_fct(output_flat, target_flat)
    #logger.info(f'classification_loss: {classification_loss}') 
    classification_loss = classification_loss[idx_flat].sum() / len(idx_flat)
    #logger.info(f'classification_loss: {classification_loss}')

    # potentially using src_key_padding_mask for something here 

    # Compute diagonal penalty
    diagonal_penalty = diagonal_attention_penalty(attn_weights, src_key_padding_mask=src_key_padding_mask)
    #logger.info(f'diagonal_penalty: {diagonal_penalty}')

    # Combined loss
    total_loss = classification_loss + lambda_penalty * diagonal_penalty
    return total_loss

def diagonal_attention_penalty(attn_weights, src_key_padding_mask):
    """
    Calculate the diagonal attention penalty.

    Parameters:
    attn_weights (torch.Tensor): Attention weights tensor.

    Returns:
    torch.Tensor: Calculated penalty.
    """
    # attn_weights: [batch_size, num_heads, seq_len, seq_len]
    # seq_len is the largest seqence length in the batch 
    batch_size, num_heads, seq_len, _ = attn_weights.size()
    device = attn_weights.device

    # Create a diagonal mask
    diag_mask = torch.eye(seq_len, device=device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len]
    
    # Create a mask for valid positions (non-padded positions)
    valid_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, num_heads, seq_len, -1)  # Shape: [batch_size, num_heads, seq_len, seq_len]

    # Apply the valid mask to the attention weights
    attn_weights = attn_weights * valid_mask

    # Extract diagonal attention weights
    diag_values = attn_weights * diag_mask  # Retain only diagonal values

     # Calculate the valid sequence lengths
    valid_lengths = src_key_padding_mask.sum(dim=1).unsqueeze(1).unsqueeze(2).expand(-1, num_heads, seq_len)  # Shape: [batch_size, num_heads, seq_len]

    # Penalize the diagonal attention weights
    #penalty = diag_values.sum(dim=(-2, -1)) #/ valid_lengths.sum(dim=-1) # I think this here is broken 
    #return penalty.mean()  # Return average penalty over batch and heads

    penalty = torch.nansum(diag_values, dim=(-2,-1)) / valid_lengths.sum(dim=-1)
    #logger.info(f'penalty: {penalty.mean()}')

    return penalty.mean()


def cosine_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """
    Create a cosine learning rate scheduler with warmup.

    Parameters:
    optimizer (torch.optim.Optimizer): Optimizer.
    num_warmup_steps (int): Number of warmup steps.
    num_training_steps (int): Total number of training steps.
    min_lr_ratio (float, optional): Minimum learning rate ratio.
    last_epoch (int, optional): The index of the last epoch.

    Returns:
    torch.optim.lr_scheduler.LambdaLR: Learning rate scheduler.
    """ 
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            logger.info('warmup step')
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps))))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def sinusoidal_positional_encoding(seq_len, d_model, device):
    """
    Generate sinusoidal positional encoding.

    Parameters:
    seq_len (int): Sequence length.
    d_model (int): Dimension of the model.
    device (torch.device): Device to create the tensor on.

    Returns:
    torch.Tensor: Sinusoidal positional encoding tensor.
    """
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(np.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def train(
    model,
    train_dataloader,
    test_dataloader,
    epochs=10,
    step_size=10, 
    gamma=0.1,
    lr=1e-5,
    min_lr_ratio=0.1,
    save_path="model",
    device="cuda",
    checkpoint_interval=1, 
    diagonal_loss=True, 
    lambda_penalty=0.1
):
    """
    Train the model.

    Parameters:
    model (nn.Module): Model to train.
    train_dataloader (DataLoader): DataLoader for training data.
    test_dataloader (DataLoader): DataLoader for test data.
    epochs (int, optional): Number of training epochs.
    step_size (int, optional): Step size for the learning rate scheduler.
    gamma (float, optional): Gamma for the learning rate scheduler.
    lr (float, optional): Learning rate.
    min_lr_ratio (float, optional): Minimum learning rate ratio.
    save_path (str, optional): Path to save the model.
    device (str, optional): Device to train on ('cuda' or 'cpu').
    checkpoint_interval (int, optional): Interval for saving checkpoints.
    diagonal_loss (bool, optional): If True, use diagonal loss function.
    lambda_penalty (float, optional): Penalty for diagonal attention.

    Returns:
    None
    """
    logger.info("Training on " + str(device)) 
    model.to(device)
    logger.info(f"Model is on device: {next(model.parameters()).device}")

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)

    # Initialize learning rate scheduler
    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = epochs / 4
    scheduler = cosine_lr_scheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, min_lr_ratio=min_lr_ratio)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    final_validation_weights = []
    final_validation_attention = [] 
    final_validation_masks = []  # Add this line

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
            src_key_padding_mask = (masks != -2).bool().to(device) #NOTE I HAVE EDITED THIS LINE 
     
           
            with autocast():
                # Use the diagonal loss function if specified
                if diagonal_loss: 
                    #logger.info(f"Using diagonal loss with lambda_penalty: {lambda_penalty}")
                    outputs, attn_weights = model(embeddings, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True)
                    loss = combined_loss(
                        outputs, categories, masks, attn_weights, idx, src_key_padding_mask, lambda_penalty=lambda_penalty
                    ) 
                else:  
                    outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                    #logger.info(f"Outputs are on device: {outputs.device}")
                    loss = masked_loss(
                        outputs, categories, masks, idx
                    )
                #logger.info(f"Loss is on device: {loss.device}")

            # Backpropagation and optimization
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
        final_validation_categories = []  # List to store categories for each validation instance
        with torch.no_grad():
            for embeddings, categories, masks, idx in test_dataloader:
                embeddings, categories, masks = (
                    embeddings.to(device).float(),
                    categories.to(device).long(),
                    masks.to(device).float(),
                )
               
                src_key_padding_mask = (masks != -2).bool().to(device)  # Mask the padding from the transformer 
                
                with autocast():
                    outputs, attn_weights = model(
                        embeddings, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True
                    )
                    if diagonal_loss:
                        val_loss = combined_loss(
                            outputs, categories, masks, attn_weights, idx, src_key_padding_mask, lambda_penalty=lambda_penalty
                        )
                    else:
                        val_loss = masked_loss(
                            outputs, categories, masks, idx
                        )
                total_val_loss += val_loss.item()

                if epoch == epochs - 1:
                    final_validation_weights.append(outputs.cpu().detach().numpy())
                    final_validation_attention.append(attn_weights.cpu().detach().numpy())
                    final_validation_categories.append(categories.cpu().detach().numpy())  # Store categories
                    final_validation_masks.append(masks.cpu().detach().numpy())  # Store masks

        avg_val_loss = total_val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Save the final validation categories and masks
        with open(os.path.join(save_path, "final_validation_categories.pkl"), "wb") as f:
            pickle.dump(final_validation_categories, f)
        with open(os.path.join(save_path, "final_validation_masks.pkl"), "wb") as f:  # Add this block
            pickle.dump(final_validation_masks, f)

        # Clear cache after validation epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Update scheduler
        scheduler.step()
        logger.info(f'learning rate {scheduler.get_last_lr()}', flush=True)

        # Save checkpoint every 'checkpoint_interval' epochs 
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

    # Clear cache after saving the model
    gc.collect()
    torch.cuda.empty_cache()



def train_fold(fold, train_index, val_index, device_id, dataset, attention, batch_size, epochs, lr, step_size, gamma, min_lr_ratio, save_path, num_heads, hidden_dim, dropout, checkpoint_interval, intialisation, diagonal_loss, lambda_penalty, num_layers, output_dim):
    fold_logger = None
    try:
        device = torch.device(device_id)
        output_dir = f"{save_path}/fold_{fold}"
        
        # Set up a new logger for each fold
        fold_logger = logger.bind(fold=fold)
        fold_logger.add(os.path.join(output_dir, "trainer.log"), level="DEBUG")

        fold_logger.info(f"FOLD: {fold} - Starting training")

        # Create output directory for the current fold
        try:
            fold_logger.info("Creating output directory")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                fold_logger.info(f"Directory created: {output_dir}")
            else:
                fold_logger.info(f"Warning: Directory {output_dir} already exists.")
        except Exception as e:
            fold_logger.error(f"Error creating output directory: {e}")
            raise

        # Use SubsetRandomSampler for efficient subsetting
        fold_logger.info("Generating subsamples")
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)

        # Create data loaders with samplers
        fold_logger.info("Saving datasets")
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

        # Log the size of training and validation data
        fold_logger.info(f"Training data size: {len(train_sampler)} samples")
        fold_logger.info(f"Validation data size: {len(val_sampler)} samples")

        # save the validation data object as well as the keys used in validation 
        pickle.dump(val_kfold_loader, open(output_dir + "/val_kfold_loader.pkl", "wb"))
        with open(output_dir + "/val_kfold_keys.txt", "w") as file: 
            for k in val_index: 
                file.write(f"{k}\n")

        # Initialize model
        try:
            fold_logger.info("Initializing model")
            input_dim = dataset[0][0].shape[1]
            num_classes = 9
            fold_logger.info(f"Model parameters: input_dim={input_dim}, num_classes={num_classes}, output_dim={output_dim}")  # Log input_dim, num_classes, and output_dim
            if attention == "circular":
                kfold_transformer_model = (
                    Seq2SeqTransformerClassifierCircularRelativeAttention(
                        input_dim=input_dim,
                        num_classes=num_classes,
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        intialisation=intialisation,
                        num_layers=num_layers,
                        output_dim=output_dim  # Pass output_dim
                    ).to(device)
                )
            elif attention == "relative":
                kfold_transformer_model = Seq2SeqTransformerClassifierRelativeAttention(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    intialisation=intialisation,
                    num_layers=num_layers,
                    output_dim=output_dim  # Pass output_dim
                ).to(device)
            elif attention == "absolute":
                kfold_transformer_model = Seq2SeqTransformerClassifier(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    intialisation=intialisation,
                    num_layers=num_layers,
                    output_dim=output_dim  # Pass output_dim
                ).to(device)
            else:
                fold_logger.error("Invalid attention type specified")
                raise ValueError("Invalid attention type specified")
        except Exception as e:
            fold_logger.error(f"Error initializing model: {e}")
            raise

        # Log the devices of various components of the model
        log_model_devices(kfold_transformer_model)

        # Train model
        fold_logger.info("Training the model")
        train(
            kfold_transformer_model,
            train_kfold_loader,
            val_kfold_loader,
            epochs=epochs,
            lr=lr,
            step_size=step_size, 
            gamma=gamma, 
            min_lr_ratio=min_lr_ratio,
            save_path=output_dir,
            device=device,
            checkpoint_interval=checkpoint_interval,
            diagonal_loss=diagonal_loss,
            lambda_penalty=lambda_penalty
        )

        fold_logger.info(f"FOLD: {fold} - Training completed")

        # Clear cache after training each fold
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        if fold_logger: 
            fold_logger.error(f"Error in fold {fold}: {e}")
        else:
            fold_logger.error(f"Error in fold {fold}: {e}")

def train_crossValidation(
    dataset,
    attention,
    n_splits=10,
    batch_size=16,
    epochs=10,
    lr=1e-5,
    step_size=10, 
    gamma=0.1,
    min_lr_ratio=0.1,
    save_path="out",
    num_heads=4,
    hidden_dim=512,
    device="cuda",
    dropout=0.1,
    checkpoint_interval=1, 
    intialisation='random',
    diagonal_loss=True,
    lambda_penalty=0.1,
    parallel_kfolds=False,
    num_layers=2, 
    random_seed=42,
    single_fold=None,
    output_dim=None  # Add output_dim parameter
):
    """
    Train the model using K-Fold cross-validation.

    Parameters:
    dataset (Dataset): Dataset to train on.
    attention (str): Type of attention ('absolute', 'relative', 'circular').
    n_splits (int, optional): Number of folds for cross-validation.
    batch_size (int, optional): Batch size.
    epochs (int, optional): Number of training epochs.
    lr (float, optional): Learning rate.
    step_size (int, optional): Step size for the learning rate scheduler.
    gamma (float, optional): Gamma for the learning rate scheduler.
    min_lr_ratio (float, optional): Minimum learning rate ratio.
    save_path (str, optional): Path to save the model.
    num_heads (int, optional): Number of attention heads.
    hidden_dim (int, optional): Hidden dimension size.
    device (str, optional): Device to train on ('cuda' or 'cpu').
    dropout (float, optional): Dropout rate.
    checkpoint_interval (int, optional): Interval for saving checkpoints.
    intialisation (str, optional): Initialization method for positional encoding ('random' or 'zero').
    diagonal_loss (bool, optional): If True, use diagonal loss function.
    lambda_penalty (float, optional): Penalty for diagonal attention.
    parallel_kfolds (bool, optional): If True, train kfolds in parallel.
    num_layers (int, optional): Number of transformer layers.
    random_seed (int, optional): Random seed for reproducibility.
    single_fold (int, optional): If specified, train only the specified fold.

    Returns:
    None
    """
    # access the logger object
    logger.add(save_path + "trainer.log", level="DEBUG")

    # Log the parameters being used to train the model
    logger.info(f"Training parameters: attention={attention}, n_splits={n_splits}, batch_size={batch_size}, epochs={epochs}, lr={lr}, step_size={step_size}, gamma={gamma}, min_lr_ratio={min_lr_ratio}, save_path={save_path}, num_heads={num_heads}, hidden_dim={hidden_dim}, device={device}, dropout={dropout}, checkpoint_interval={checkpoint_interval}, intialisation={intialisation}, diagonal_loss={diagonal_loss}, lambda_penalty={lambda_penalty}, parallel_kfolds={parallel_kfolds}, num_layers={num_layers}, output_dim={output_dim}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold = 1
    logger.info(
        "Training with K-Fold crossvalidation with " + str(n_splits) + " folds..."
    )

    if single_fold is not None:
        if single_fold < 1 or single_fold > n_splits:
            logger.error(f"Invalid single_fold value: {single_fold}. It must be between 1 and {n_splits}.")
            raise ValueError(f"Invalid single_fold value: {single_fold}. It must be between 1 and {n_splits}.")
        logger.info(f"Training only fold {single_fold}")
        for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
            if fold == single_fold:
                fold_logger = logger.bind(fold=fold)  # Ensure fold_logger is defined
                logger.info(f"Training fold {fold} on device {device}")
                train_fold(fold, train_index, val_index, device, dataset, attention, batch_size, epochs, lr, step_size, gamma, min_lr_ratio, save_path, num_heads, hidden_dim, dropout, checkpoint_interval, intialisation, diagonal_loss, lambda_penalty, num_layers, output_dim)
                break
    else:
        if parallel_kfolds:
            # Set start method for multiprocessing
            mp.set_start_method('spawn', force=True)

            # Create a process for each fold
            processes = []
            num_gpus = torch.cuda.device_count()
            for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
                device_id = (fold - 1) % num_gpus
                logger.info(f"Training fold {fold} on device {device_id}")
                p = mp.Process(target=train_fold, args=(fold, train_index, val_index, device_id, dataset, attention, batch_size, epochs, lr, step_size, gamma, min_lr_ratio, save_path, num_heads, hidden_dim, dropout, checkpoint_interval, intialisation, diagonal_loss, lambda_penalty, num_layers, output_dim))
                p.start()
                processes.append(p)

            # Wait for all processes to finish
            for p in processes:
                p.join()

            # Check if any process is still alive
            for p in processes:
                if p.is_alive():
                    logger.error(f"Process {p.pid} is still alive. Terminating...")
                    p.terminate()
                    p.join()

            # Clear cache after all processes finish
            gc.collect()
            torch.cuda.empty_cache()
        else:
            num_gpus = torch.cuda.device_count()
            for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
                device_id = (fold - 1) % num_gpus
                fold_logger = logger.bind(fold=fold)  # Ensure fold_logger is defined
                logger.info(f"Training fold {fold} on device {device_id}")
                train_fold(fold, train_index, val_index, device_id, dataset, attention, batch_size, epochs, lr, step_size, gamma, min_lr_ratio, save_path, num_heads, hidden_dim, dropout, checkpoint_interval, intialisation, diagonal_loss, lambda_penalty, num_layers, output_dim)

            # Clear cache after all folds finish
            gc.collect()
            torch.cuda.empty_cache()

def evaluate(model, dataloader, phrog_integer, device, output_dir="metrics_output"):
    """
    Evaluate the model.

    Parameters:
    model (nn.Module): Model to evaluate.
    dataloader (DataLoader): DataLoader for evaluation data.
    phrog_integer (dict): Dictionary mapping integer labels to categories.
    device (str): Device to evaluate on ('cuda' or 'cpu').
    output_dir (str, optional): Directory to save evaluation metrics.

    Returns:
    None
    """
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

        roc_df = pd.DataFrame({"FPR": fpr, "TPR": thresholds})
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


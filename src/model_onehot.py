"""
Modules for building transformer model 
"""

#from build.lib.src.model_onehot import MaskedTokenFeatureDropout
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
from tqdm import tqdm  # Add import for tqdm

# Increase the limit of open files
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft * 2, hard), hard))

# Set the sharing strategy to file_system
torch.multiprocessing.set_sharing_strategy('file_system')

# Set random seed globally at the start of the script
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, categories, labels, num_classes = 9, mask_token=-1, mask_portion=0.15, shuffle_features=False, zero_idx=False, strand_gene_length=True, noise_std=0):
        """
        Initialize the dataset with embeddings, categories, and labels.

        Parameters:
        embeddings (list of torch.Tensor): List of embedding tensors.
        categories (list of torch.Tensor): List of category tensors.
        labels (list of str): List of labels associated with the data.
        mask_token (int): Token used for masking.
        mask_portion (float): Portion of tokens to mask during training.
        shuffle_features (bool): If True, shuffle the embedding features during training.
        noise_std (float): Standard deviation of the Gaussian noise to add to the embeddings.
        """
        try:
            self.embeddings = [embedding.float() for embedding in embeddings]
            self.categories = [category.long() for category in categories]
            self.labels = labels  # Add labels attribute
            self.mask_token = mask_token
            self.num_classes = num_classes # hard coded in for now
            self.mask_portion = mask_portion
            self.shuffle_features = shuffle_features  # Add shuffle_features attribute
            self.zero_idx = zero_idx  # Add zero_idx attribute
            self.noise_std = noise_std  # Add noise_std attribute
            self.strand_gene_length = strand_gene_length  # Add strand_gene_length attribute
            self.class_weights = self.calculate_class_weights()
            logger.info(f"Computed class weights: {self.class_weights}")
            logger.info(f'noise_std: {self.noise_std}')
            self.masked_category_counts = torch.zeros(num_classes)  # Add masked_category_counts attribute
        except Exception as e:
            logger.error(f"Error initializing EmbeddingDataset: {e}")
            raise

    def calculate_class_weights(self):

        # Calculate the frequency of each class 
        class_counts = torch.zeros(self.num_classes)
        for category in self.categories:
            valid_indices = category != self.mask_token # Get the indices of valid categories
            unique, counts = torch.unique(category[valid_indices], return_counts=True)
            class_counts[unique] += counts


        # Compute the inverse frequency weights
        class_weights = 1.0 / class_counts
        #logger.info(f"Class weights: {class_weights}")
        class_weights /= class_weights.sum()  # Normalize the weights
        #logger.info(f"Normalized class weights: {class_weights}")
        return class_weights

        
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

    def __getitem__(self, i):
        """
        Get a single viruses from the dataset at the specified index.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        tuple: Tuple containing the embedding, category, mask, and masked indices.
        """
        try:
            embedding = self.embeddings[i]
            category = self.categories[i]
            mask = (
                category != self.mask_token
            ).float()  # 1 for valid, 0 for missing - this indicates which rows to consider when training

            # select a random category to mask if training
            if self.training:
                masked_category, idx = self.category_mask(category, mask)
                self.update_masked_category_counts(category, idx)  # Update masked category counts
                if self.shuffle_features:
                    embedding = self.shuffle_masked_features(embedding, idx)  # Shuffle only masked features
                if self.noise_std > 0:
                    embedding = self.add_gaussian_noise(embedding, idx, std=self.noise_std)  # Add Gaussian noise to masked features
                if self.zero_idx:
                    embedding = self.set_zero_idx(embedding, idx)  # Set the embeddings of the masked tokens to zeros
                if not self.strand_gene_length:
                    embedding = self.ignore_strand_gene_length(embedding, idx)  # Ignore strand and gene length features
            elif self.validation:
                masked_category, idx = self.validation_mask(category, i)
            else:
                masked_category = category
                idx = [] 

            # use the masked category to generate a one-hot encoding
            one_hot = self.custom_one_hot_encode(masked_category)

            # append the one_hot encoding to the front of the embedding
            embedding_one_hot = torch.tensor(np.hstack((one_hot, embedding)))

            return embedding_one_hot, category, mask, idx
        except Exception as e:
            logger.error(f"Error getting item from dataset at index {i}: {e}")
            logger.error(f"Category at index {i}: {category}")
            logger.error(f"mask_token: {self.mask_token}")
            raise

    def set_training(self, training=True):
        """
        Set the training mode for the dataset.

        Parameters:
        training (bool): If True, the dataset is in training mode.
        """
        self.training = training
        self.validation = False

    def set_validation(self, validation_categories, validation=True):
        """
        Set the validation mode for the dataset.

        Parameters:
        validation (bool): If True, the dataset is in validation mode.
        validation_categories (list): Categories to validate against 
        """
        
        self.validation = validation
        self.training = False

        # store the original categories for prediction
        self.predict_categories = self.categories   

        # replace the categories with the validation categories
        self.categories = validation_categories 

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
            probability_distribution = mask.float() * self.mask_portion

            # Mutily the probability distribution by the class weights
            class_weights = self.class_weights
            expected_class_weights = torch.ones_like(class_weights) / len(class_weights)
            scale_class_weights = class_weights / expected_class_weights
            weighted_probability_distribution = probability_distribution * scale_class_weights[category]  # Fix method call
    
            # Check for invalid values in the weighted probability distribution
            if torch.isnan(weighted_probability_distribution).any() or torch.isinf(weighted_probability_distribution).any() or (weighted_probability_distribution < 0).any():
                logger.info(f"Weighted probability distribution: {weighted_probability_distribution}")
                logger.info(f"probability_distribution: {probability_distribution}")
                logger.info(f"Class weights: {class_weights}")
                logger.info(f"Category: {category}")
                logger.error(f"Invalid values found in weighted probability distribution: {weighted_probability_distribution}")
                raise ValueError("Weighted probability distribution contains invalid values")

            # Use a Bernoulli distribution to sample the indices to mask
            rand_vals = torch.rand_like(weighted_probability_distribution)
            idx = torch.nonzero(rand_vals < weighted_probability_distribution, as_tuple=True)[0]
            
            # generate masked versions
            masked_category = category.clone()
            masked_category[idx] = self.mask_token

            return masked_category, idx
        except Exception as e:
            logger.error(f"Error in category_mask: {e}")
            raise

    def validation_mask(self, category, ii):
        """
        Mask the validation categories to test the model's ability to predict the correct categories.
        
        Parameters:
        category (torch.Tensor): Category tensor.
        mask (torch.Tensor): Mask tensor.
        idx (torch.Tensor): Index tensor.
        """
        
        # the idx will be the idx that are different between the validation categories and the predicted categories
        idx = (category != self.predict_categories[ii]).nonzero(as_tuple=True)[0]
        #logger.info(f'validation mask: category: {category}')
        #logger.info(f'validation mask: predict_categories: {self.predict_categories[ii]}')
        #logger.info(f'validation mask: idx: {idx}')

        # generate masked versions
        masked_category = category.clone()
        masked_category[idx] = self.mask_token

        # check there are no unknowns in idx 
        if self.mask_token in category[idx]:
            raise ValueError("Mask token found in validation mask")

        return masked_category, idx 

    def shuffle_rows(self):
        """
        Shuffle rows within each instance in the dataset.
        This modifies the code in place.
        """
        random.seed(random_seed)  # Ensure reproducibility
        zipped_data = list(zip(self.embeddings, self.categories, self.labels))
        random.shuffle(zipped_data)
        self.embeddings, self.categories, self.labels = zip(*zipped_data)
        self.embeddings = list(self.embeddings)
        self.categories = list(self.categories)
        self.labels = list(self.labels)

   

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

    def process_all(self):
        """
        Process all the data in the dataset.
        """
        embeddings = []
        categories = []
        masks = []
        indices = []
        for idx in range(len(self)):
            embedding_one_hot, category, mask, idx = self.__getitem__(idx)
            embeddings.append(embedding_one_hot)
            categories.append(category)
            masks.append(mask)
            indices.append(idx)
        return embeddings, categories, masks, indices

    def shuffle_embedding_features(self, embedding):
        """
        Shuffle the embedding features.

        Parameters:
        embedding (torch.Tensor): Embedding tensor to shuffle.

        Returns:
        torch.Tensor: Shuffled embedding tensor.
        """
        indices = torch.randperm(embedding.size(0))
        return embedding[indices]

    def shuffle_masked_features(self, embedding, idx):
        """
        Shuffle only the masked features in the embedding.

        Parameters:
        embedding (torch.Tensor): Embedding tensor to shuffle.
        idx (torch.Tensor): Indices of masked features.

        Returns:
        torch.Tensor: Embedding tensor with shuffled masked features.
        """
        original_features = embedding.clone() # copy to check if the embedding has been shuffled
        masked_features = embedding[idx].clone()
        masked_features[:, 13:] = torch.stack([row[torch.randperm(row.size(0))] for row in masked_features[:,13:]])
        embedding[idx] = masked_features

        if torch.equal(embedding, original_features): 
            raise Exception("Shuffling did not change the embedding features.")

        return embedding

    def add_gaussian_noise(self, embedding, idx, mean=0.0, std=0.1):
        """
        Add Gaussian noise to the masked features in the embedding.

        Parameters:
        embedding (torch.Tensor): Embedding tensor to add noise to.
        idx (torch.Tensor): Indices of masked features.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

        Returns:
        torch.Tensor: Embedding tensor with added Gaussian noise.
        """
        #logger.info(f"Adding Gaussian noise with mean {mean} and std {std}")
        masked_features = embedding[idx].clone()
        #logger.info(f"Masked features: {masked_features}")
        noise = torch.normal(mean, std, size=masked_features[:, 3:].size())

        # log the effect of the noise 
        #logger.info(f'masked_features shape: {masked_features.size()}')
        #logger.info(f'Noise shape: {noise.size()}')
        #logger.info(f'masked_features max: {torch.max(masked_features[:, 3:], dim=1)}')
        #logger.info(f'masked_features min: {torch.min(masked_features[:, 3:], dim=1)}')
        #logger.info(f"Noise max: {torch.max(noise, dim=1)}")
        #logger.info(f"Noise min: {torch.min(noise, dim=1)}")

        #logger.info(f"Noise: {noise}")
        #logger.info(f"Noise shape: {noise.size()}")
        masked_features[:, 3:] += noise # is this the correct size if the embeddings haven't been added on yet 
        #logger.info(f"Noisy masked features: {masked_features}")
        embedding[idx] = masked_features

        return embedding
    
    def set_zero_idx(self, embedding, idx):
        """
        Set the embeddings of the masked tokens to zeros.

        Parameters:
        embedding (torch.Tensor): Embedding tensor to modify.
        idx (torch.Tensor): Indices of masked features.

        Returns:
        torch.Tensor: Embedding tensor with masked tokens set to zeros.
        """
        new_features = embedding[idx].clone()
        new_features[:,3:] = torch.zeros_like(new_features[:,3:])
        embedding[idx] = new_features
        #print(embedding[idx])

        return embedding
    
    def ignore_strand_gene_length(self, embedding, idx):
        """
        Ignore the strand and gene length features in the embedding.

        Parameters:
        embedding (torch.Tensor): Embedding tensor to modify.

        Returns:
        torch.Tensor: Embedding tensor with strand and gene length features ignored.
        """
        new_features = embedding[idx].clone()
        new_features[:, :3] = torch.zeros_like(new_features[:, :3])
        embedding[idx] = new_features

        return embedding

    def update_masked_category_counts(self, category, idx):
        """
        Update the counts of masked categories.

        Parameters:
        category (torch.Tensor): Original category tensor.
        idx (torch.Tensor): Indices of masked features.
        """
        for index in idx:
            cat = category[index].item()
            if cat != self.mask_token:
                self.masked_category_counts[cat] += 1

def fourier_positional_encoding(seq_len, d_model, device):
    """
    Generate Fourier positional encoding.

    Parameters:
    seq_len (int): Sequence length.
    d_model (int): Dimension of the model.
    device (torch.device): Device to create the tensor on.

    Returns:
    torch.Tensor: Fourier positional encoding tensor.
    """
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(np.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        lstm_hidden_dim=512,  # Add lstm_hidden_dim parameter
        dropout=0.1,
        intialisation='random',
        output_dim=None,  # Add output_dim parameter
        use_lstm=False,  # Add use_lstm parameter
        positional_encoding=fourier_positional_encoding,  # Use Fourier positional encoding
        use_positional_encoding=True,  # Add use_positional_encoding parameter
        protein_dropout_rate=0.0  # Add parameter for protein feature dropout
    ):
        """
        Initialize the Transformer Classifier.

        Parameters:
        input_dim (int): Input dimension size of the ESM2 embeddings
        num_classes (int): Number of output classes.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        hidden_dim (int): Hidden dimension size.
        dropout (float): Dropout rate.
        intialisation (str): Initialization method for positional encoding ('random' or 'zero').
        """
        super(TransformerClassifier, self).__init__()
        """
        Difference is between the position of the LSTM layer and the the type of positional encoding that is used
        """

        # check if cuda is available 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the number of classes
        self.num_classes = num_classes

        # Embedding layers
        self.func_embedding = nn.Embedding(num_classes, 16).to(device)
        self.strand_embedding = nn.Linear(2, 2).to(device)  # Change to linear layer
        self.length_embedding = nn.Linear(1, 8).to(device)
        self.embedding_layer = nn.Linear(input_dim, hidden_dim - 28).to(device)  # Use input_dim

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

        # Final Classification Layer
        self.output_dim = output_dim if output_dim else num_classes  # Set output_dim
        if use_lstm:
            self.lstm = nn.LSTM(
                hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional=True  # Use lstm_hidden_dim
            ).to(device)
            self.fc = nn.Linear(2 * lstm_hidden_dim, self.output_dim).to(device)  # Use lstm_hidden_dim
            d_model = lstm_hidden_dim * 2  # Use lstm_hidden_dim
        else:
            self.lstm = None  # Ensure lstm attribute exists
            self.fc = nn.Linear(hidden_dim, self.output_dim).to(device)  # Use hidden_dim directly
            d_model = hidden_dim  # Use hidden_dim directly

        # Transformer Encoder
        # Create transformer encoder only if num_layers > 0
        if num_layers > 0:
            encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True  # Use d_model
        ).to(device)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers).to(device)
        else:
            # Create an identity module when no transformer layers are needed
            self.transformer_encoder = nn.Identity().to(device)
            logger.info("Using Identity layer instead of transformer encoder (num_layers=0)")
            
        # Positional Encoding
        if use_positional_encoding:
            self.positional_encoding = positional_encoding(1000, hidden_dim, device).to(device)
        else:
            self.positional_encoding = None

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
        func_ids, strand_ids, gene_length, protein_embeds = x[:,:,:self.num_classes], x[:,:,self.num_classes:self.num_classes+2], x[:,:,self.num_classes+2:self.num_classes+3], x[:,:,self.num_classes+3:]
        func_embeds = self.func_embedding(func_ids.argmax(-1))
        strand_embeds = self.strand_embedding(strand_ids.float())  # Change to linear layer
        length_embeds = self.length_embedding(gene_length)
        protein_embeds = self.embedding_layer(protein_embeds)

        x = torch.cat([func_embeds, strand_embeds, length_embeds, protein_embeds], dim=-1)
        if self.positional_encoding is not None:
            x = x + self.positional_encoding[: x.size(1), :].to(x.device)

        x = self.dropout(x)
        if self.lstm:
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
    if model.positional_encoding is not None:
        logger.info(f"positional_encoding is on device: {model.positional_encoding.device}")
    else:
        logger.info("positional_encoding is not used")
    if model.lstm:
        logger.info(f"lstm is on device: {next(model.lstm.parameters()).device}")
    if model.num_layers > 0:
        logger.info(f"transformer_encoder is on device: {next(model.transformer_encoder.parameters()).device}")
        logger.info(f"transformer_encoder is on device: {next(model.transformer_encoder.parameters()).device}")
        logger.info(f"fc is on device: {model.fc.weight.device}")

class RelativePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1500, batch_first=True, intialisation = 'random'):
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

        # Check for NaN values in the input tensors
        if torch.isnan(query).any() or torch.isnan(key).any() or torch.isnan(value).any():
            logger.error("NaN values found in input tensors")

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
            d_model // self.num_heads
        )

        # Add relative position biases
        rel_positions = self.relative_position_k[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        scores = scores + torch.einsum("bhqd,bqd->bhq", q, rel_positions)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(3) == 0, float("-inf"))


        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Mask the resulting attention weights to ensure padded positions have zero attention
        if src_key_padding_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0) # swap any nan values remaining from the padding to zeros 

        attn_output = torch.matmul(attn_weights, v)

        # broadcasting 
        rel_positions_v = self.relative_position_v[:seq_len, :]
        attn_output = attn_output + torch.matmul(attn_weights, rel_positions_v)

        # Check for NaN values in the attention weights
        if torch.isnan(attn_weights).any():
            logger.error("NaN values found in attention weights")

        # Check for NaN values in the attention weights
        if torch.isnan(attn_weights).any():
            logger.error("NaN values found in attention weights")

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
        self, d_model, num_heads, dim_feedforward=512, dropout=0.1, max_len=1500, intialisation='random'
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


class TransformerClassifierRelativeAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        lstm_hidden_dim=512,  # Add lstm_hidden_dim parameter
        dropout=0.1,
        max_len=1500,
        intialisation='random',
        output_dim=None,  # Add output_dim parameter
        use_lstm=False,  # Add use_lstm parameter
        positional_encoding=fourier_positional_encoding,  # Use Fourier positional encoding
        use_positional_encoding=True,  # Add use_positional_encoding parameter
        protein_dropout_rate=0.0  # Add parameter for protein feature dropout
    ):
        """
        Initialize the Transformer Classifier with Relative Attention.

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
        super(TransformerClassifierRelativeAttention, self).__init__()

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the number of classes
        self.num_classes = num_classes

        # Embedding layers
        self.func_embedding = nn.Embedding(self.num_classes, 16).to(device)
        self.strand_embedding = nn.Linear(2, 2).to(device)  # Change to linear layer
        self.length_embedding = nn.Linear(1, 8).to(device)
        self.embedding_layer = nn.Linear(input_dim, hidden_dim - 28).to(device)  # Use input_dim

        self.dropout = nn.Dropout(dropout).to(device)
        self.positional_encoding = positional_encoding(max_len, hidden_dim, device).to(device)
        self.output_dim = output_dim if output_dim else num_classes  # Set output_dim
        if use_lstm:
            self.lstm = nn.LSTM(
                hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional=True  # Use lstm_hidden_dim
            ).to(device)
            self.fc = nn.Linear(2 * lstm_hidden_dim, self.output_dim).to(device)  # Use lstm_hidden_dim
            d_model = lstm_hidden_dim * 2  # Use lstm_hidden_dim
        else:
            self.lstm = None  # Ensure lstm attribute exists
            self.fc = nn.Linear(hidden_dim, self.output_dim).to(device)  # Use hidden_dim directly
            d_model = hidden_dim  # Use hidden_dim directly

        # Create transformer encoder only if num_layers > 0
        if num_layers > 0:
            encoder_layers = CustomTransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                max_len=max_len,
                initialisation=intialisation
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers).to(device)
        else:
            # Create an identity module when no transformer layers are needed
            self.transformer_encoder = nn.Identity().to(device)
            logger.info("Using Identity layer instead of transformer encoder (num_layers=0)")

        if use_positional_encoding:
            self.positional_encoding = positional_encoding(max_len, hidden_dim, device).to(device)
        else:
            self.positional_encoding = None


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
        func_ids, strand_ids, gene_length, protein_embeds = x[:,:,:self.num_classes], x[:,:,self.num_classes:self.num_classes+2], x[:,:,self.num_classes+2:self.num_classes+3], x[:,:,self.num_classes+3:]
        func_embeds = self.func_embedding(func_ids.argmax(-1))
        strand_embeds = self.strand_embedding(strand_ids.float())  # Change to linear layer
        length_embeds = self.length_embedding(gene_length)
        protein_embeds = self.embedding_layer(protein_embeds)

        x = torch.cat([func_embeds, strand_embeds, length_embeds, protein_embeds], dim=-1)
        if self.positional_encoding is not None:
            x = x + self.positional_encoding[: x.size(1), :].to(x.device)

        x = self.dropout(x)
        if self.lstm:
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
    def __init__(self, d_model, num_heads, max_len=1500, batch_first=True, intialisation = 'random'):
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
            .expand(batch_size, -1, -1, -1)
        )
        scores += torch.einsum("bhqd,bqkd->bhqk", q, rel_positions_k)

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
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0) # swap any nan values remaining from the padding to zeros 

        attn_output = torch.matmul(attn_weights, v)

        # Adjust dimensions for broadcasting
        rel_positions_v = (
            self.relative_position_v[circular_indices]
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
        )
        attn_output += torch.einsum("bhqk,bqkd->bhqd", attn_weights, rel_positions_v)

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
        self, d_model, num_heads, dim_feedforward=512, dropout=0.1, max_len=1500, initialisation='random', pre_norm=False
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
        pre_norm (bool): If True, use pre-normalization, otherwise use post-normalization.
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
        self.pre_norm = pre_norm

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
        if self.pre_norm:
            # Pre-normalization architecture
            if return_attn_weights:
                # Apply layer norm before attention
                src_norm = self.norm1(src)
                src2, attn_weights = self.self_attn(src_norm, src_norm, src_norm, 
                                                   attn_mask=src_mask, is_causal=is_causal, 
                                                   src_key_padding_mask=src_key_padding_mask, 
                                                   return_attn_weights=return_attn_weights)
                src = src + self.dropout1(src2)
                
                # Apply layer norm before FFN
                src_norm = self.norm2(src)
                src2 = self.linear2(self.dropout(F.relu(self.linear1(src_norm))))
                src = src + self.dropout2(src2)
                return src, attn_weights
            else:
                # Apply layer norm before attention
                src_norm = self.norm1(src)
                src2 = self.self_attn(src_norm, src_norm, src_norm, 
                                     attn_mask=src_mask, src_key_padding_mask=src_key_padding_mask, 
                                     is_causal=is_causal)
                src = src + self.dropout1(src2)
                
                # Apply layer norm before FFN
                src_norm = self.norm2(src)
                src2 = self.linear2(self.dropout(F.relu(self.linear1(src_norm))))
                src = src + self.dropout2(src2)
                return src
        else:
            # Post-normalization architecture (original)
            if return_attn_weights: 
                src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, 
                                                   is_causal=is_causal, 
                                                   src_key_padding_mask=src_key_padding_mask, 
                                                   return_attn_weights=return_attn_weights)
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
                src = src + self.dropout2(src2)
                src = self.norm2(src)
                return src, attn_weights
            else: 
                src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
                                     src_key_padding_mask=src_key_padding_mask, 
                                     is_causal=is_causal)
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
                src = src + self.dropout2(src2)
                src = self.norm2(src)
                return src
            

class TransformerClassifierCircularRelativeAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        hidden_dim=512,
        lstm_hidden_dim=512,
        dropout=0.1,
        max_len=1500,
        intialisation='random',
        output_dim=None,
        use_lstm=False,
        positional_encoding=fourier_positional_encoding,
        use_positional_encoding=True,
        protein_dropout_rate=0.0,  # Add parameter for protein feature dropout
        function_embedding_dim=16,
        strand_embedding_dim=2,
        length_embedding_dim=8,
        pre_norm=False,
        progressive_dropout=False,
        initial_dropout_rate=1.0,
        final_dropout_rate=0.4,
        progressive_epochs=25,
    ):
        super(TransformerClassifierCircularRelativeAttention, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the number of classes
        self.num_classes = num_classes

        # Embedding layers
        self.func_embedding = nn.Embedding(self.num_classes,function_embedding_dim).to(device)
        self.strand_embedding = nn.Linear(2, strand_embedding_dim).to(device)
        self.length_embedding = nn.Linear(1, length_embedding_dim).to(device)
        self.gene_feature_dim = function_embedding_dim + strand_embedding_dim + length_embedding_dim
        logger.info(f'gene feature dim: {self.gene_feature_dim}')
        self.embedding_layer = nn.Linear(input_dim, hidden_dim - self.gene_feature_dim).to(device)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout).to(device)
     
        
        # Add protein feature normalization
        #self.protein_norm = nn.LayerNorm(hidden_dim - self.gene_feature_dim).to(device)
        
        # Add protein feature dropout layer
        self.protein_feature_dropout = MaskedTokenFeatureDropout(dropout_rate=protein_dropout_rate, 
                                                                 progressive_dropout=progressive_dropout,  # Default value set to False
                                                                 initial_dropout_rate=initial_dropout_rate,
                                                                 final_dropout_rate=final_dropout_rate, 
                                                                 protein_idx=self.gene_feature_dim,  # just added this 
                                                                 total_epochs=progressive_epochs) 
        #self.protein_feature_dropout.num_classes = num_classes  # Pass num_classes
        self.positional_encoding = positional_encoding(max_len, hidden_dim, device).to(device) if use_positional_encoding else None
        
        # Add positional normalization
        self.pos_norm = nn.LayerNorm(hidden_dim).to(device)
        self.output_dim = output_dim if output_dim else num_classes

        # Create transformer encoder only if num_layers > 0
        if num_layers > 0:
            encoder_layers = CircularTransformerEncoderLayer(
                d_model=hidden_dim,  # Transformer's input dimension matches the embedding dim
                num_heads=num_heads,
                dropout=dropout,
                max_len=max_len,
                initialisation=intialisation,
                pre_norm=pre_norm
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers).to(device)
            
            # LSTM takes transformer output as input
            if use_lstm:
                self.lstm = nn.LSTM(
                    hidden_dim,  # Input size matches transformer's output dimension
                    lstm_hidden_dim, 
                    batch_first=True, 
                    bidirectional=True
                ).to(device)
                self.fc = nn.Linear(2 * lstm_hidden_dim, self.output_dim).to(device)
            else:
                self.lstm = None
                self.fc = nn.Linear(hidden_dim, self.output_dim).to(device)
        else:
            # Create an identity module when no transformer layers are needed
            self.transformer_encoder = nn.Identity().to(device)
            logger.info("Using Identity layer instead of transformer encoder (num_layers=0)")
            
            # If no transformer, LSTM takes embedding output directly
            if use_lstm:
                self.lstm = nn.LSTM(
                    hidden_dim,
                    lstm_hidden_dim, 
                    batch_first=True, 
                    bidirectional=True
                ).to(device)
                self.fc = nn.Linear(2 * lstm_hidden_dim, self.output_dim).to(device)
            else:
                self.lstm = None
                self.fc = nn.Linear(hidden_dim, self.output_dim).to(device)

    def forward(self, x, src_key_padding_mask=None, idx=None, return_attn_weights=False, save_lstm_output=False, save_transformer_output=False):
        x = x.float()

        # Extract the different components from the input tensor
        func_ids, strand_ids, gene_length, protein_embeds = x[:,:,:self.num_classes], x[:,:,self.num_classes:self.num_classes+2], x[:,:,self.num_classes+2:self.num_classes+3], x[:,:,self.num_classes+3:]
        func_embeds = self.func_embedding(func_ids.argmax(-1))
        strand_embeds = self.strand_embedding(strand_ids.float())
        length_embeds = self.length_embedding(gene_length)
        protein_embeds = self.embedding_layer(protein_embeds)

        # Concatenate the embeddings 
        x = torch.cat([func_embeds, strand_embeds, length_embeds, protein_embeds], dim=-1)

        # Apply protein feature dropout to the masked tokens 
        if idx is not None:
            x = self.protein_feature_dropout(x, idx)

        # Apply positional encoding
        if self.positional_encoding is not None:
            x = x + self.positional_encoding[: x.size(1), :].to(x.device)
            
        # Normalize combined features after positional encoding
        x = self.pos_norm(x)

        # Apply dropout
        x = self.dropout(x)

        # TRANSFORMER FIRST: Apply transformer encoder
        if return_attn_weights and isinstance(self.transformer_encoder, nn.TransformerEncoder) and hasattr(self.transformer_encoder, 'layers') and len(self.transformer_encoder.layers) > 0:
            x, attn_weights = self.transformer_encoder.layers[0](x, src_key_padding_mask=src_key_padding_mask, return_attn_weights=True)
            for layer in self.transformer_encoder.layers[1:]:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer_encoder(x)
        
        if save_transformer_output:
            self.saved_transformer_output = x.clone().detach().cpu().numpy()
        
        # THEN LSTM: Apply LSTM after transformer
        if self.lstm:
            x, _ = self.lstm(x)
            if save_lstm_output:
                self.saved_lstm_output = x.clone().detach().cpu().numpy()
        
        # Final classification
        x = self.fc(x)
        if self.output_dim != self.num_classes:
            x = F.softmax(x, dim=-1)
        
        # Return with attention weights if requested
        if return_attn_weights:
            if 'attn_weights' in locals():
                return x, attn_weights
            else:
                # Create dummy attention weights
                batch_size = x.size(0)
                seq_len = x.size(1)
                dummy_attn_weights = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=x.device)
                return x, dummy_attn_weights
        else:
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

def combine_loss_distillation(output, target, mask, attn_weights, idx, src_key_padding_mask, ignore_index=-1, alpha=0.5, lambda_penalty=0.1):

    # compute the combined loss
    penalised_loss, classification_loss = combined_loss(output, target, mask, attn_weights, idx, src_key_padding_mask, lambda_penalty=lambda_penalty) 

    # now computed the distillation loss
    #penalised_loss + kl_loss(output.view(-1, num_classes), logits from teachers )

    return penalised_loss, classification_loss  


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

    # Flatten the batch and sequence dimensions
    output_flat = output.view(-1, num_classes) #maybe this one 
    target_flat = target.view(-1)
    mask_flat = mask.view(-1)

    # Adjust indices for flattened batch and sequence
    idx_overall = [ii + val * seq_len for val, ii in enumerate(idx)]
    idx_flat = torch.cat([t.flatten() for t in idx_overall])

    # Apply the mask
    target_flat[mask_flat == 0] = ignore_index

    # Compute classification loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none").to(device)
    classification_loss = loss_fct(output_flat, target_flat) # output_flat these are the 10 class logits 
    classification_loss = classification_loss[idx_flat].sum() / len(idx_flat)

    # Check for NaN values in the classification loss
    if torch.isnan(classification_loss).any():
        logger.error("NaN values found in classification loss")

    # If lambda_penalty is 0 or no attention weights, skip diagonal penalty
    if lambda_penalty == 0 or attn_weights is None:
        return classification_loss, classification_loss

    # Compute diagonal penalty
    diagonal_penalty = diagonal_attention_penalty(attn_weights, src_key_padding_mask, idx)

    # Check for NaN values in the diagonal penalty
    if torch.isnan(diagonal_penalty).any():
        logger.error("NaN values found in diagonal penalty")

    # Combined loss
    penalised_loss = classification_loss + lambda_penalty * diagonal_penalty

    # Check for NaN values in the combined loss
    if torch.isnan(penalised_loss).any():
        logger.error("NaN values found in combined loss")

    return penalised_loss, classification_loss

def diagonal_attention_entire(attn_weights, src_key_padding_mask):
    """
    Calculate the diagonal attention penalty. This doesn't focus on the masked tokens but the entire sequence

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
    penalty = torch.nansum(diag_values, dim=(-2,-1)) / valid_lengths.sum(dim=-1)

    return penalty.mean()

def diagonal_attention_penalty(attn_weights, src_key_padding_mask, idx):
    """
    Enhanced diagonal attention penalty mechanism. Compares the ratio of self attention to other attention 
    """
     # If there are no attention weights (num_layers=0), return zero penalty
    if attn_weights is None:
        return torch.tensor(0.0, device=src_key_padding_mask.device)
    
    batch_size, num_heads, seq_len, _ = attn_weights.size()
    device = attn_weights.device
    
    # Create a mask that only considers positions in idx (the masked tokens)
    # Initialize with zeros
    idx_mask = torch.zeros(batch_size, seq_len, device=device)
    
    # For each batch item, mark the positions in idx as 1
    for b in range(batch_size):
        if b < len(idx) and idx[b].numel() > 0:
            idx_mask[b, idx[b]] = 1.0
    
    # Expand for attention heads
    idx_mask = idx_mask.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, seq_len)
    
    # Create diagonal mask
    diag_mask = torch.eye(seq_len, device=device).unsqueeze(0).unsqueeze(0)
    
    # Apply masks to keep only diagonal attention for masked tokens
    valid_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
    masked_diag_attn = attn_weights * diag_mask * idx_mask * valid_mask
    
    # Get total attention from masked tokens
    masked_total_attn = attn_weights * idx_mask * valid_mask
    
    # Calculate ratio of diagonal attention to total for masked tokens
    diag_sum = masked_diag_attn.sum(dim=(-2, -1))
    total_sum = masked_total_attn.sum(dim=(-2, -1)) + 1e-8
    
    return (diag_sum / total_sum).mean()
    

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
    # Ensure reproducibility in scheduler
    np.random.seed(random_seed)

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

def calculate_accuracy(output, target, mask, idx):
    """
    Calculate accuracy using only the indices of the masked data.

    Parameters:
    output (torch.Tensor): Output tensor from the model.
    target (torch.Tensor): Target tensor.
    mask (torch.Tensor): Mask tensor.
    idx (list of torch.Tensor): Indices of masked tokens.

    Returns:
    float: Calculated accuracy.
    """
    with torch.no_grad():
        _, preds = torch.max(output, dim=2)
        correct = (preds == target) * mask

        # Adjust indices for flattened batch and sequence
        batch_size, seq_len = target.shape
        idx_overall = [ii + val * seq_len for val, ii in enumerate(idx)]
        idx_flat = torch.cat([t.flatten() for t in idx_overall])

        # Only consider the elements at the specific `idx` positions
        correct_flat = correct.view(-1)[idx_flat]
        mask_flat = mask.view(-1)[idx_flat]

        accuracy = correct_flat.sum().float() / mask_flat.sum().float()
    return accuracy.item()

def train(
    model,
    train_dataloader,
    test_dataloader,
    epochs=10,
    lr=1e-5,
    min_lr_ratio=0.1,
    save_path="model",
    device="cuda",
    checkpoint_interval=1, 
    lambda_penalty=0.1,
    num_classes=9,  # Add num_classes parameter
    zero_idx=False,  # Add zero_idx parameter
    strand_gene_length=True  # Add strand_gene_length parameter
):
    """
    Train the model.

    Parameters:
    model (nn.Module): Model to train.
    train_dataloader (DataLoader): DataLoader for training data.
    train_dataloader (DataLoader): DataLoader for training data.
    epochs (int, optional): Number of training epochs.
    lr (float, optional): Learning rate.
    min_lr_ratio (float, optional): Minimum learning rate ratio.
    save_path (str, optional): Path to save the model.
    device (str, optional): Device to train on ('cuda' or 'cpu').
    checkpoint_interval (int, optional): Interval for saving checkpoints.
    lambda_penalty (float, optional): Penalty for diagonal attention.
    num_classes (int, optional): Number of classes.
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
    
    # lists to store metrics 
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_classification_losses = []
    final_validation_weights = []
    final_validation_attention = [] 
    final_validation_masks = []
    diagonal_penalities = []

    # Track best validation loss and corresponding model
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_path, "best_model.pth")
    best_epoch = 0

    # Initialize DataFrame to store metrics
    metrics_df = pd.DataFrame(
        columns=[
            "epoch",
            "training losses",
            "validation losses",
            "diagonal_penalities",
            "classification_losses",
            "training accuracies",
            "validation accuracies",
        ]
    )

    logger.info("Beginning training loop")
    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode
        total_loss = 0
        total_correct = 0
        total_samples = 0
        masked_category_counts = torch.zeros(num_classes, device=device)  # Initialize masked category counts

        # At the beginning of each epoch in train()
        if hasattr(model, 'protein_feature_dropout') and hasattr(model.protein_feature_dropout, 'update_dropout_rate'):
            logger.info(f'Updating dropout rate for epoch {epoch + 1}')
            model.protein_feature_dropout.update_dropout_rate(epoch)
        else: 
            logger.info('No dropout rate update function found in model')

        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        for embeddings, categories, masks, idx in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            embeddings, categories, masks = (
                embeddings.to(device).float(),
                categories.to(device).long(),
                masks.to(device).float(),
            )

            optimizer.zero_grad()
            
            # Mask the padding from the transformer 
            src_key_padding_mask = (masks != -2).bool().to(device) 
        
            # Forward pass
            with torch.amp.autocast('cuda'):
                outputs, attn_weights = model(embeddings, src_key_padding_mask=src_key_padding_mask, idx=idx, return_attn_weights=True)
                loss, _ = combined_loss(outputs, categories, masks, attn_weights, idx, src_key_padding_mask, lambda_penalty=lambda_penalty) 
                accuracy = calculate_accuracy(outputs, categories, masks, idx)
                       
            # Backpropagation and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update total loss and accuracy
            total_loss += loss.item()
            total_correct += accuracy * len(idx)
            total_samples += len(idx)

            # Update masked category counts
            masked_category_counts += train_dataloader.dataset.masked_category_counts.to(device)
            #logger.info(f"Masked Category Counts - prior: {masked_category_counts.cpu().numpy()}")
            train_dataloader.dataset.masked_category_counts.zero_()  # Reset counts after each batch

        # Calculate average training loss and accuracy for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_accuracy = total_correct / total_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Log training loss, accuracy, and masked category counts
        logger.info(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Training Accuracy: {avg_train_accuracy}")
        logger.info(f"Epoch {epoch + 1}/{epochs}, Masked Category Counts: {masked_category_counts.cpu().numpy()}")

        # Clear cache after training epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Validation loss and accuracy
        model.eval()  # Ensure model is in evaluation mode
        total_val_loss = 0
        total_val_classification_loss = 0
        total_diagonal_penalty = 0
        total_val_correct = 0
        total_val_samples = 0
        final_validation_categories = []  # List to store categories for each validation instance

        # Initialize counters for validation
        val_masked_category_counts = torch.zeros(num_classes, device=device)
        val_predicted_counts = torch.zeros(num_classes, device=device)
        val_correct_counts = torch.zeros(num_classes, device=device)

        with torch.no_grad():  # No need to calculate gradients for validation
            for embeddings, categories, masks, idx in tqdm(test_dataloader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                embeddings, categories, masks = (
                    embeddings.to(device).float(),
                    categories.to(device).long(),
                    masks.to(device).float(),
                )
               
                # Mask the padding from the transformer
                src_key_padding_mask = (masks != -2).bool().to(device)  
                
                # Forward pass
                with torch.amp.autocast('cuda'):
                    outputs, attn_weights = model(embeddings, src_key_padding_mask=src_key_padding_mask, idx=idx, return_attn_weights=True)
                    val_loss, val_classification_loss = combined_loss(outputs, categories, masks, attn_weights, idx, src_key_padding_mask, lambda_penalty=lambda_penalty)
                    accuracy = calculate_accuracy(outputs, categories, masks, idx)
                    
                    # Calculate diagonal penalty so it can be stored in the metrics
                    diagonal_penalty = diagonal_attention_penalty(attn_weights, src_key_padding_mask, idx)

                # Update total validation loss and accuracy
                total_val_loss += val_loss.item()
                total_val_classification_loss += val_classification_loss.item()
                total_diagonal_penalty += diagonal_penalty.item()
                total_val_correct += accuracy * len(idx)
                total_val_samples += len(idx)

                # Update validation counters
                for batch_idx, indices in enumerate(idx):
                    for i in indices:
                        true_label = categories[batch_idx, i].item()
                        pred_label = outputs.argmax(dim=-1)[batch_idx, i].item()
                        if true_label != -1:  # Ignore padding or invalid labels
                            val_masked_category_counts[true_label] += 1
                            val_predicted_counts[pred_label] += 1
                            if true_label == pred_label:
                                val_correct_counts[true_label] += 1

                # Store the final validation weights and attention weights
                if epoch == epochs - 1:
                    final_validation_weights.append(outputs.cpu().detach().numpy())
                    final_validation_attention.append(attn_weights.cpu().detach().numpy())
                    final_validation_categories.append(categories.cpu().detach().numpy())  # Store categories
                    final_validation_masks.append(masks.cpu().detach().numpy())  # Store masks

        # Calculate average validation loss and accuracy
        avg_val_loss = total_val_loss / len(test_dataloader)
        avg_val_classification_loss = total_val_classification_loss / len(test_dataloader)
        avg_diagonal_penalty = total_diagonal_penalty / len(test_dataloader)
        avg_val_accuracy = total_val_correct / total_val_samples

        # Check if this is the best model so far based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Epoch {epoch + 1}/{epochs}: New best validation loss: {best_val_loss:.4f}. Model saved to {best_model_path}")

        # Log validation metrics
        logger.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")
        logger.info("Validation masked category counts:")
        for i in range(num_classes):
            logger.info(f"Category {i}: {val_masked_category_counts[i].item()} masked")
        logger.info("Validation predicted counts:")
        for i in range(num_classes):
            logger.info(f"Category {i}: {val_predicted_counts[i].item()} predicted")
        logger.info("Validation accuracy per category:")
        per_category_accuracies = {}
        for i in range(num_classes):
            if val_masked_category_counts[i] > 0:
                accuracy = val_correct_counts[i].item() / val_masked_category_counts[i].item()
                logger.info(f"Category {i}: {accuracy:.2f}")
                per_category_accuracies[i] = accuracy
            else:
                logger.info(f"Category {i}: No masked samples")
                per_category_accuracies[i] = None

        # Save per-category metrics to a file
        per_category_metrics = {
            "epoch": epoch + 1,
            "masked_counts": val_masked_category_counts.cpu().tolist(),
            "predicted_counts": val_predicted_counts.cpu().tolist(),
            "accuracies": per_category_accuracies,
        }
        metrics_file = os.path.join(save_path, f"per_category_metrics_epoch_{epoch + 1}.json")
        with open(metrics_file, "w") as f:
            import json
            json.dump(per_category_metrics, f, indent=4)
        logger.info(f"Per-category metrics saved to {metrics_file}")

        # Append metrics to lists
        val_losses.append(avg_val_loss)
        val_classification_losses.append(avg_val_classification_loss)
        diagonal_penalities.append(avg_diagonal_penalty)
        val_accuracies.append(avg_val_accuracy)

        # Log validation loss and accuracy
        #logger.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")

        # Save the final validation categories and masks
        with open(os.path.join(save_path, "final_validation_categories.pkl"), "wb") as f:
            pickle.dump(final_validation_categories, f)
        with open(os.path.join(save_path, "final_validation_masks.pkl"), "wb") as f:  
            pickle.dump(final_validation_masks, f)

        # Clear cache after validation epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Update scheduler
        scheduler.step()
        logger.info(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {scheduler.get_last_lr()}")

        # Save checkpoint every 'checkpoint_interval' epochs 
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Create a DataFrame for the current epoch's metrics
        epoch_metrics = pd.DataFrame(
            {
                "epoch": [epoch],
                "training losses": [avg_train_loss],
                "validation losses": [avg_val_loss],
                "diagonal_penalities": [avg_diagonal_penalty],
                "classification_losses": [avg_val_classification_loss],
                "training accuracies": [avg_train_accuracy],
                "validation accuracies": [avg_val_accuracy],
            }
        )

        # Concatenate the current epoch's metrics with the overall metrics DataFrame
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        # Save the metrics DataFrame to CSV after each epoch
        metrics_df.to_csv(os.path.join(save_path, "metrics.csv"), index=False)

    # Save the final model's state dictionary
    model_save_path = os.path.join(save_path, "transformer_state_dict.pth")
    torch.save(model.state_dict(), model_save_path)  # Ensure only state_dict is saved
    logger.info(f"Model state dictionary saved at {model_save_path}")
    logger.info(f"Best model was from epoch {best_epoch} with validation loss {best_val_loss:.4f}")

    # Reload the best model for evaluation instead of the last model
    logger.info("Reloading the best model for validation testing...")
    reloaded_model = type(model)(  # Reinitialize the model with the same parameters
        input_dim=model.embedding_layer.in_features, #+ 28,  # Adjust input_dim
        num_classes=model.num_classes,
        num_heads=model.transformer_encoder.layers[0].self_attn.num_heads,
        num_layers=len(model.transformer_encoder.layers),
        hidden_dim=model.embedding_layer.out_features + 28,
        lstm_hidden_dim=model.lstm.hidden_size if model.lstm else None,
        dropout=model.dropout.p,
        intialisation='random',  # Adjust as needed
        output_dim=model.output_dim,
        use_lstm=model.lstm is not None,
        positional_encoding=sinusoidal_positional_encoding,
        use_positional_encoding=model.positional_encoding is not None,
        protein_dropout_rate=model.protein_feature_dropout.dropout_rate if hasattr(model, 'protein_feature_dropout') else 0.0
    ).to(device)

    # Log the parameters used for reloading the model
    logger.info(f'Model type: {type(model)}')
    logger.info(f"Reloaded model parameters: input_dim={model.embedding_layer.in_features}, "
                f"num_classes={model.num_classes}, num_heads={model.transformer_encoder.layers[0].self_attn.num_heads}, "
                f"num_layers={len(model.transformer_encoder.layers)}, hidden_dim={model.embedding_layer.out_features + 28}, "
                f"lstm_hidden_dim={model.lstm.hidden_size if model.lstm else None}, dropout={model.dropout.p}, "
                f"output_dim={model.output_dim}, use_lstm={model.lstm is not None}, "
                f"use_positional_encoding={model.positional_encoding is not None}, "
                f"protein_dropout_rate={model.protein_feature_dropout.dropout_rate if hasattr(model, 'protein_feature_dropout') else 0.0}")

    reloaded_model.load_state_dict(torch.load(best_model_path, map_location=device, strict=False))
    reloaded_model.eval()

    # Test the reloaded model on validation data
    with torch.no_grad():
        total_val_correct = 0
        total_val_samples = 0
        val_masked_category_counts = torch.zeros(num_classes, device=device)
        val_predicted_counts = torch.zeros(num_classes, device=device)
        val_correct_counts = torch.zeros(num_classes, device=device)

        for embeddings, categories, masks, idx in test_dataloader:
            embeddings, categories, masks = (
                embeddings.to(device).float(),
                categories.to(device).long(),
                masks.to(device).float(),
            )
            src_key_padding_mask = (masks != -2).bool().to(device)
            outputs = reloaded_model(embeddings, src_key_padding_mask=src_key_padding_mask, idx=idx)

            # Evaluate only rows with idx
            for batch_idx, indices in enumerate(idx):
                for i in indices:
                    pred = outputs.argmax(dim=-1)[batch_idx, i].item()
                    true_label = categories[batch_idx, i].item()
                    if true_label != -1:  # Ignore padding or invalid predictions
                        val_masked_category_counts[true_label] += 1
                        val_predicted_counts[pred] += 1
                        if pred == true_label:
                            val_correct_counts[true_label] += 1
                        

            accuracy = calculate_accuracy(outputs, categories, masks, idx)
            total_val_correct += accuracy * len(idx)
            total_val_samples += len(idx)

        avg_val_accuracy = total_val_correct / total_val_samples
        logger.info(f"Validation Accuracy of Reloaded Model: {avg_val_accuracy:.2f}")

        # Log validation metrics
        logger.info("Validation masked category counts:")
        for i in range(num_classes):
            logger.info(f"Category {i}: {val_masked_category_counts[i].item()} masked")
        logger.info("Validation predicted counts:")
        for i in range(num_classes):
            logger.info(f"Category {i}: {val_predicted_counts[i].item()} predicted")
        logger.info("Validation accuracy per category:")
        for i in range(num_classes):
            if val_masked_category_counts[i] > 0:
                accuracy = val_correct_counts[i].item() / val_masked_category_counts[i].item()
                logger.info(f"Category {i}: {accuracy:.2f}")
            else:
                logger.info(f"Category {i}: No masked samples")

    # Save the training and validation loss to CSV
    logger.info("Saving metrics to CSV")
    loss_df = pd.DataFrame(
        {
            "epoch": [i for i in range(epochs)],
            "training losses": train_losses,
            "validation losses": val_losses,  # this here is the 'penalised loss'
            "diagonal_penalities": diagonal_penalities,
            "classification_losses": val_classification_losses,
            "training accuracies": train_accuracies,
            "validation accuracies": val_accuracies,
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

def train_fold(fold, train_index, val_index, device_id, dataset, attention, batch_size, epochs, lr, min_lr_ratio, save_path, num_heads, hidden_dim, lstm_hidden_dim, dropout, checkpoint_interval, intialisation, lambda_penalty, num_layers, output_dim, input_size, use_lstm, positional_encoding=fourier_positional_encoding, use_positional_encoding=True, zero_idx=False, strand_gene_length=True, protein_dropout_rate=0.0, pre_norm=False, progressive_dropout=False, initial_dropout_rate=1.0, final_dropout_rate=0.4):
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

        # Log the indices of training and validation data to ensure no leakage
        #fold_logger.info(f"Training indices: {train_index}")
        #fold_logger.info(f"Validation indices: {val_index}")

        # Check for data leakage
        train_set = set(train_index)
        val_set = set(val_index)
        intersection = train_set.intersection(val_set)
        if (intersection):
            fold_logger.error(f"Data leakage detected! Overlapping indices: {intersection}")
        else:
            fold_logger.info("No data leakage detected between training and validation sets.")

        # save the validation data object as well as the labels used in validation 
        pickle.dump(val_kfold_loader, open(output_dir + "/val_kfold_loader.pkl", "wb"))
        with open(output_dir + "/val_kfold_labels.txt", "w") as file: 
            for idx in val_index: 
                label = dataset.labels[idx]  # Get the label from the dataset object
                file.write(f"{label}\n")

        # Initialize model
        try:
            fold_logger.info("Initializing model")
            input_dim = dataset[0][0].shape[1]
            num_classes = 9 # TODO FIX THIS hardcoded value
            fold_logger.info(f"Model parameters: input_dim={input_dim}, num_classes={num_classes}, output_dim={output_dim}")  # Log input_dim, num_classes, and output_dim
            if (attention == "circular"):
                kfold_transformer_model = TransformerClassifierCircularRelativeAttention(
                    input_dim=input_size,  # Use input_size
                    num_classes=num_classes,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    lstm_hidden_dim=lstm_hidden_dim,  # Pass lstm_hidden_dim
                    dropout=dropout,
                    intialisation=intialisation,
                    num_layers=num_layers,
                    output_dim=output_dim,  # Pass output_dim
                    use_lstm=use_lstm,  # Pass use_lstm
                    positional_encoding=positional_encoding,  # Use Fourier positional encoding
                    use_positional_encoding=use_positional_encoding,  # Pass use_positional_encoding
                    protein_dropout_rate=protein_dropout_rate,  # Pass the parameter here
                    pre_norm=pre_norm,  # Pass pre_norm
                    progressive_dropout=progressive_dropout,  # Pass progressive_dropout
                    initial_dropout_rate=initial_dropout_rate,  # Pass initial_dropout_rate
                    final_dropout_rate=final_dropout_rate  # Pass final_dropout_rate
                ).to(device)
            elif attention == "relative":
                kfold_transformer_model = TransformerClassifierRelativeAttention(
                    input_dim=input_size,  # Use input_size
                    num_classes=num_classes,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    lstm_hidden_dim=lstm_hidden_dim,  # Pass lstm_hidden_dim
                    dropout=dropout,
                    intialisation=intialisation,
                    num_layers=num_layers,
                    output_dim=output_dim,  # Pass output_dim
                    use_lstm=use_lstm,  # Pass use_lstm
                    positional_encoding=positional_encoding,  # Use Fourier positional encoding
                    use_positional_encoding=use_positional_encoding,  # Pass use_positional_encoding
                    protein_dropout_rate=protein_dropout_rate  # Pass the parameter here
                ).to(device)
            elif attention == "absolute":
                kfold_transformer_model = TransformerClassifier(
                    input_dim=input_size,  # Use input_size
                    num_classes=num_classes,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    lstm_hidden_dim=lstm_hidden_dim,  # Pass lstm_hidden_dim
                    dropout=dropout,
                    intialisation=intialisation,
                    num_layers=num_layers,
                    output_dim=output_dim,  # Pass output_dim
                    use_lstm=use_lstm,  # Pass use_lstm
                    positional_encoding=positional_encoding,  # Use Fourier positional encoding
                    use_positional_encoding=use_positional_encoding,  # Pass use_positional_encoding
                    protein_dropout_rate=protein_dropout_rate  # Pass the parameter here
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
            min_lr_ratio=min_lr_ratio,
            save_path=output_dir,
            device=device,
            checkpoint_interval=checkpoint_interval,
            lambda_penalty=lambda_penalty,
            num_classes=num_classes,  # Pass num_classes
            zero_idx=zero_idx,  # Pass zero_idx
            strand_gene_length=strand_gene_length  # Pass strand_gene_length
        )

        fold_logger.info(f"FOLD: {fold} - Training completed")

        # Clear cache after training each fold
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        if fold_logger: 
            fold_logger.error(f"Error in fold {fold} at line {traceback.extract_tb(e.__traceback__)[-1][1]}: {e}")
        else:
            fold_logger.error(f"Error in fold {fold} at line {traceback.extract_tb(e.__traceback__)[-1][1]}: {e}")

def train_crossValidation(
    dataset,
    attention,
    n_splits=10,
    batch_size=16,
    epochs=10,
    lr=1e-5,
    min_lr_ratio=0.1,
    save_path="out",
    num_heads=4,
    hidden_dim=512,
    lstm_hidden_dim=512,
    device="cuda",
    dropout=0.1,
    checkpoint_interval=10,
    intialisation='random',
    lambda_penalty=0.1,
    parallel_kfolds=False,
    num_layers=2,
    random_seed=42,
    single_fold=None,
    output_dim=None,
    input_size=None,
    use_lstm=False,
    positional_encoding=fourier_positional_encoding,
    use_positional_encoding=True,
    noise_std=0.0,
    zero_idx=False,
    strand_gene_length=True,
    protein_dropout_rate=0.0,
    pre_norm=False,
    progressive_dropout=False,
    initial_dropout_rate=1.0,
    final_dropout_rate=0.4,
):
    """
    Train the model using K-Fold cross-validation.

    Parameters:
    # ...existing docstring...
    """
    logger.add(save_path + "/trainer.log", level="DEBUG")

    # Log the parameters being used to train the model
    logger.info(f"Training parameters: attention={attention}, n_splits={n_splits}, batch_size={batch_size}, epochs={epochs}, lr={lr}, min_lr_ratio={min_lr_ratio}, save_path={save_path}, num_heads={num_heads}, hidden_dim={hidden_dim}, device={device}, dropout={dropout}, checkpoint_interval={checkpoint_interval}, intialisation={intialisation}, lambda_penalty={lambda_penalty}, parallel_kfolds={parallel_kfolds}, num_layers={num_layers}, output_dim={output_dim}, positional_encoding={positional_encoding}, use_positional_encoding={use_positional_encoding}, noise_std={noise_std}, zero_idx={zero_idx}, strand_gene_length={strand_gene_length}, protein_dropout_rate={protein_dropout_rate}, pre_norm={pre_norm}, progressive_dropout={progressive_dropout}, initial_dropout_rate={initial_dropout_rate}, final_dropout_rate={final_dropout_rate}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    logger.info(f"Training with K-Fold cross-validation with {n_splits} folds...")

    if single_fold is not None:
        if single_fold < 1 or single_fold > n_splits:
            logger.error(f"Invalid single_fold value: {single_fold}. It must be between 1 and {n_splits}.")
            raise ValueError(f"Invalid single_fold value: {single_fold}. It must be between 1 and {n_splits}.")
        logger.info(f"Training only fold {single_fold}")
        for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
            if fold == single_fold:
                train_fold(
                    fold,
                    train_index,
                    val_index,
                    device,
                    dataset,
                    attention,
                    batch_size,
                    epochs,
                    lr,
                    min_lr_ratio,
                    save_path,
                    num_heads,
                    hidden_dim,
                    lstm_hidden_dim,
                    dropout,
                    checkpoint_interval,
                    intialisation,
                    lambda_penalty,
                    num_layers,
                    output_dim,
                    input_size,
                    use_lstm,
                    positional_encoding,
                    use_positional_encoding,
                    zero_idx,
                    strand_gene_length,
                    protein_dropout_rate,
                    pre_norm,
                    progressive_dropout,
                    initial_dropout_rate,
                    final_dropout_rate,
                )
                break
    else:
        if parallel_kfolds:
            mp.set_start_method('spawn', force=True)
            processes = []
            num_gpus = torch.cuda.device_count()
            for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
                device_id = (fold - 1) % num_gpus
                logger.info(f"Training fold {fold} on device {device_id}")
                p = mp.Process(
                    target=train_fold,
                    args=(
                        fold,
                        train_index,
                        val_index,
                        device_id,
                        dataset,
                        attention,
                        batch_size,
                        epochs,
                        lr,
                        min_lr_ratio,
                        save_path,
                        num_heads,
                        hidden_dim,
                        lstm_hidden_dim,
                        dropout,
                        checkpoint_interval,
                        intialisation,
                        lambda_penalty,
                        num_layers,
                        output_dim,
                        input_size,
                        use_lstm,
                        positional_encoding,
                        use_positional_encoding,
                        zero_idx,
                        strand_gene_length,
                        protein_dropout_rate,
                        pre_norm,
                        progressive_dropout,
                        initial_dropout_rate,
                        final_dropout_rate,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            for p in processes:
                if p.is_alive():
                    logger.error(f"Process {p.pid} is still alive. Terminating...")
                    p.terminate()
                    p.join()

            gc.collect()
            torch.cuda.empty_cache()
        else:
            num_gpus = torch.cuda.device_count()
            for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
                device_id = (fold - 1) % num_gpus
                logger.info(f"Training fold {fold} on device {device_id}")
                train_fold(
                    fold,
                    train_index,
                    val_index,
                    device_id,
                    dataset,
                    attention,
                    batch_size,
                    epochs,
                    lr,
                    min_lr_ratio,
                    save_path,
                    num_heads,
                    hidden_dim,
                    lstm_hidden_dim,
                    dropout,
                    checkpoint_interval,
                    intialisation,
                    lambda_penalty,
                    num_layers,
                    output_dim,
                    input_size,
                    use_lstm,
                    positional_encoding,
                    use_positional_encoding,
                    zero_idx,
                    strand_gene_length,
                    protein_dropout_rate,
                    pre_norm,
                    progressive_dropout,
                    initial_dropout_rate,
                    final_dropout_rate,
                )

            gc.collect()
            torch.cuda.empty_cache()

def save_logits(model, dataloader, device, output_dir="logits_output"):
    """
    Evaluate the model and save the logits.

    Parameters:
    model (nn.Module): Model to evaluate.
    dataloader (DataLoader): DataLoader for evaluation data.
    device (str): Device to evaluate on ('cuda' or 'cpu').
    output_dir (str, optional): Directory to save the logits.

    Returns:
    None
    """
    model.to(device)
    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings, categories, masks, idx = batch
            embeddings = embeddings.to(device).float()
            masks = masks.to(device).float()
            src_key_padding_mask = (masks != -2).bool().to(device)

            outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask, return_attn_weights=False)
            if model.output_dim == 9:
                logits = outputs
                logger.info("Using raw outputs as logits")
            else:
                logits = F.softmax(outputs, dim=-1)
                logger.info("Using softmax outputs as logits")
            all_logits.append(logits.cpu().numpy())

    # Flatten the list of logits
    all_logits = np.concatenate(all_logits, axis=0)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

     # Save the logits to a file
    np.save(os.path.join(output_dir, "logits.npy"), all_logits)
    logger.info("Logits saved successfully")



class ProteinFeatureDropout(nn.Module):
    """
    Custom dropout layer that specifically targets protein features in embeddings.
    
    During training, applies standard dropout.
    During inference, applies a consistent dropout mask for stability.
    """

    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        # Register buffer to store fixed mask for inference
        self.register_buffer('fixed_mask', None)
        
    def forward(self, x, protein_idx=None, is_training=None):
        """
        Apply dropout to protein features starting from protein_idx.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, feature_dim]
            protein_idx (int): Index where protein features start in the feature dimension
            is_training (bool, optional): Override the module's training state
            
        Returns:
            torch.Tensor: The input tensor with dropout applied only to protein features
        """
        # Default protein index if not specified (assuming format from EmbeddingDataset)
        if (protein_idx is None):
            # Default: after one-hot class encoding, strand info, and gene length
            # num_classes + 2 (strand) + 1 (length)
            if hasattr(self, 'num_classes'):
                protein_idx = self.num_classes + 3
            else:
                protein_idx = 12  # Default assuming 9 classes + 3 features
                
        # Use module's training state if not specified
        is_training = self.training if is_training is None else is_training
        
        # Create a copy of the input to modify
        result = x.clone()
        
        # Extract protein features
        protein_features = x[:, :, protein_idx:]
        
        if is_training:
            # Apply standard dropout during training
            dropped_features = F.dropout(protein_features, p=self.dropout_rate, training=True)
        else:
            # Use consistent mask during inference
            if (self.fixed_mask is None or self.fixed_mask.shape != protein_features.shape):
                # Generate a new fixed mask if needed
                self.fixed_mask = torch.bernoulli(
                    torch.ones_like(protein_features) * (1 - self.dropout_rate)
                ).to(protein_features.device)
            
            # Apply the fixed mask
            dropped_features = protein_features * self.fixed_mask
        
        # Replace protein features in the result tensor
        result[:, :, protein_idx:] = dropped_features
        
        return result

class MaskedTokenFeatureDropout(nn.Module): 
    """
    Custom Dropout layer that applies dropout to embeddings of masked tokens.
    During training, applies standard dropout to masked tokens.
    Can progressively decrease dropout rate during training to encourage the model
    to first learn gene order and gradually incorporate protein features.
    """

    def __init__(self, dropout_rate=0.2, protein_idx=28, progressive_dropout=False, 
                 initial_dropout_rate=1.0, final_dropout_rate=None, total_epochs=15):
        super().__init__()

        self.protein_idx = protein_idx 
        self.progressive_dropout = progressive_dropout

        # Set up progesssive dropout parameters 
        if progressive_dropout:
            self.initial_dropout_rate = initial_dropout_rate
            self.final_dropout_rate = final_dropout_rate if final_dropout_rate is not None else dropout_rate
            self.dropout_rate = initial_dropout_rate # Start with initial dropout rate
        else: 
            self.dropout_rate = dropout_rate # Fixed dropout rate

        self.total_epochs = total_epochs
        self.current_epoch = 0

        # dont register the buffer here, as it will be created in the forward pass
        self.register_buffer('fixed_mask', None)

    def update_dropout_rate(self, current_epoch): 
        """
        Update the dropout rate based on the current epoch.

        Args:
            current_epoch (int): Current epoch number.
        """
        if self.progressive_dropout:
            self.current_epoch = current_epoch

            # Calculate progress as a fraction (0 to 1)
            progress = min(1.0, self.current_epoch / self.total_epochs)
            # Linearly interpolate between initial and final dropout rates
            self.dropout_rate = self.initial_dropout_rate - (self.initial_dropout_rate - self.final_dropout_rate) * progress
            logger.info(f"Epoch {self.current_epoch}: Updated masked protein dropout rate: {self.dropout_rate:.4f}")
        else:
            logger.info("Progressive dropout is not enabled. Using fixed dropout rate.")

    def forward(self, x, idx, is_training=None):
        """
        Apply dropout to the features of masked tokens.

        Args: 
            x (torch.Tensor): Input tensor [batch_size, seq_len, feature_dim]
            idx (torch.Tensor): Indices of the masked tokens
            protein_idx (int): Index where protein features start in the feature dimension
            is_training (bool, optional): Override the module's training state

        Returns:
            torch.Tensor: The input tensor with dropout applied only to protein features
        """
        # Use module's training state if not specified
        is_training = self.training if is_training is None else is_training

        # Create a copy of the input to modify
        result = x.clone() 

        # Only process if we have indices for masked tokens
        if not idx:
            return result

        # Process each batch item individually 
        batch_size = x.size(0)
        for b in range(batch_size): 

            # check this batch goes up to the maximum size  (last batch is sometimes smaller) and that it isn't empty 
            if (b < len(idx) and idx[b].numel() > 0): 

                # get the masked features for the masked tokens in this batch 
                masked_features = x[b, idx[b], self.protein_idx:]  # Extract features of masked tokens

                if is_training:
                    # Apply standard dropout during training
                    dropped_features = F.dropout(masked_features, p=self.dropout_rate, training=True)

                else: 
                    if (self.fixed_mask is None or self.fixed_mask.shape != masked_features.shape):
                        # Generate a new fixed mask if needed
                        self.fixed_mask = torch.bernoulli(
                            torch.ones_like(masked_features) * (1 - self.dropout_rate)
                        ).to(masked_features.device)

                    # Apply the fixed mask
                    dropped_features = masked_features * self.fixed_mask

                # Upate only the maksed token features 
                result[b, idx[b], self.protein_idx:] = dropped_features

        return result































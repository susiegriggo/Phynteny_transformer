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
import pandas as pd
import random
import os
import gc 

class VariableSeq2SeqEmbeddingDataset(Dataset):
    def __init__(self, embeddings, categories, mask_token=-1, mask_portion=0.15):
        self.embeddings = [embedding.float() for embedding in embeddings]
        self.categories = [category.long() for category in categories]
        self.mask_token = mask_token
        self.num_classes = 10 # hard coded in for now 
        self.mask_portion = mask_portion
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        category = self.categories[idx]
        mask = (category != self.mask_token).float()  # 1 for valid, 0 for missing - this indicates which rows to consider when training 
        
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
        
    def category_mask_old(self, category): 
        
        # select a category to randomly mask 
        idx = random.randint(0, len(category) - 1)
        while category[idx] == -1: 
            idx = random.randint(0, len(category) - 1)
        
        # generate masked versions 
        masked_category = category.clone()
        masked_category[idx] = self.mask_token
        
        return masked_category, idx
    
    def category_mask(self, category, mask): 

        # Define a probability distribution over maskable tokens
        probability_distribution = mask.float() / mask.sum()

        # Calculate the number of tokens to mask based on input length
        num_maskable_tokens = mask.sum().item()  # Count of maskable tokens
        num_tokens_to_mask = max(1, int(self.mask_portion * num_maskable_tokens))  # Ensure at least 1 token is masked

        # Sample tokens to mask
        idx = torch.multinomial(probability_distribution, num_samples=num_tokens_to_mask, replacement=False)

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
        
class VariableSeq2SeqTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=512, lstm_hidden_dim=512, dropout=0.1):
        super(VariableSeq2SeqTransformerClassifier, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, hidden_dim).cuda()
        self.dropout = nn.Dropout(dropout).cuda()

        # LSTM layer 
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True).cuda()  
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim).cuda()

        # Positional Encoding # this poisiitonal encoding might be better 
        self.positional_encoding = self.create_positional_encoding(hidden_dim, max_len=1000).cuda()
        
        # Learnable scaling factor for positional encodings 
        self.positional_scale = nn.Parameter(torch.ones(1))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True).cuda()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).cuda()
        
        self.fc = nn.Linear(hidden_dim, num_classes).cuda()
    
    def forward(self, x, src_key_padding_mask=None):
        x = x.float()  # Ensure input is of type float
        x = self.embedding_layer(x)  # x: [batch_size, seq_len, input_dim]
        x = self.layer_norm(x)  # Apply Layer Normalisation

        # Adjust positional encoding to match the sequence length of the input
        pos_enc = self.positional_encoding[:, :x.size(1), :]
        x = x + self.positional_scale * pos_enc  # Add positional encoding
        
        # lstm layer doesn't seem to get used here 

        x = self.dropout(x)  # Apply dropout after embedding layer 
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # x: [batch_size, seq_len, hidden_dim]
        x = self.fc(x)  # x: [batch_size, seq_len, num_classes]

        return x
    
    def create_positional_encoding(self, dim, max_len=1000): 
        pos_enc = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000) / dim))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)
    
    
class ImprovedSeq2SeqTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=512, lstm_hidden_dim=512, dropout=0.1, max_len=1000):
        super(ImprovedSeq2SeqTransformerClassifier, self).__init__()
        """
        Difference is between the position of the LSTM layer and the the type of positional encoding that is used
        """

        # Embedding layer
        self.embedding_layer = nn.Linear(input_dim, hidden_dim).cuda()

        # could add a normalisation layer here 
        self.dropout = nn.Dropout(dropout).cuda()

        # Positional Encoding (now entirely learnable) 
        #self.positional_encoding = nn.Parameter(torch.zeros(1000, hidden_dim)).cuda()

        # Relative Position Embeddings
        self.relative_position_bias = nn.Parameter(torch.zeros((2* max_len - 1, num_heads)))

        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional = True).cuda()

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True).cuda()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).cuda()
    
        # Final Classification Layer
        self.fc = nn.Linear(hidden_dim, num_classes).cuda()

    def forward(self, x, src_key_padding_mask=None):
        x = x.float() # Ensure input is of type float 
        x = self.embedding_layer(x) 
        # could apply a normalisation layer here 

        # entirely learnable positional encoding 
        #x = x + self.positional_encoding[:x.size(1), :]

        x = self.dropout(x) # apply dropout after the embedding layer 
        x, _ = self.lstm(x)  # LSTM layer
        
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Apply transformer encoder with relative positional encodings
        x = self.relative_attention(x, src_key_padding_mask)

        self.fc(x)
        return x
    
    def relative_attention(self, x, src_key_padding_mask=None):
        batch_size, seq_len, dim = x.size()
        # Create a relative position matrix
        relative_position_matrix = self.create_relative_position_matrix(seq_len)
        
        # Compute the attention weights with relative position encodings
        attention_scores = self.compute_attention_scores(x, relative_position_matrix)
        
        # Apply mask if provided
        if src_key_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(src_key_padding_mask[:, None, None, :], float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        context_layer = torch.matmul(attention_probs, x)
        return context_layer

    def create_relative_position_matrix(self, seq_len):
        range_vec = torch.arange(seq_len)
        relative_positions = range_vec[:, None] - range_vec[None, :]
        relative_positions += seq_len - 1  # Shift to ensure all indices are positive
        return relative_positions

    def compute_attention_scores(self, x, relative_position_matrix):
        batch_size, seq_len, dim = x.size()
        num_heads = self.relative_position_bias.size(1)

        # Compute the standard query-key dot product attention
        queries = x.view(batch_size, seq_len, num_heads, dim // num_heads)
        keys = x.view(batch_size, seq_len, num_heads, dim // num_heads).transpose(1, 2)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias[relative_position_matrix]
        attention_scores += relative_position_bias.permute(2, 0, 1)  # Adjust for broadcasting

        return attention_scores
    
def masked_loss(output, target, mask, idx, ignore_index=-1):
    # Loss function focused on predicting the specific category
    target = target.clone()
    target[mask == 0] = ignore_index
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none').cuda()
    loss = loss_fct(output.view(-1, output.size(-1)), target.view(-1))
    return loss[idx].sum()/len(idx)

def collate_fn(batch):
    embeddings, categories, masks, idx = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    categories_padded = pad_sequence(categories, batch_first=True, padding_value=-1)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return embeddings_padded, categories_padded, masks_padded, idx 
 
def loss(output, target, mask, ignore_index=-1):
    # this loss function should be looking at the predicting gene not everything 
    target = target.clone()
    target[mask == 0] = ignore_index
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none').cuda()
    loss = loss_fct(output.view(-1, output.size(-1)), target.view(-1))
    loss = loss * mask.view(-1)
    return loss.sum() / mask.sum()

def train(model, train_dataloader, test_dataloader, epochs=5, lr=1e-5, save_path='model', device='cuda', mask_unknowns = True):
    print('Training on ' + str(device), flush=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler()
    train_losses = []
    val_losses = []
    
    print('Beginning training loop', flush=True)
    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode
        total_loss = 0
        for embeddings, categories, masks, idx in train_dataloader:
            embeddings, categories, masks = embeddings.to(device).float(), categories.to(device).long(), masks.to(device).float()
            optimizer.zero_grad()
            if mask_unknowns: 
                src_key_padding_mask = (masks == 0).bool()  # Mask for transformer
            else: 
                src_key_padding_mask = None 
            with autocast():
                # for original model 
                outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                loss = masked_loss(outputs, categories, masks, idx) # need to specify which index to reference in the loss function
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss}")

        # Clear cache after training epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Validation loss
        model.eval()  # Ensure model is in evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for embeddings, categories, masks, idx in test_dataloader:
                embeddings, categories, masks = embeddings.to(device).float(), categories.to(device).long(), masks.to(device).float()
                src_key_padding_mask = (masks == 0).bool()  # Mask for transformer
                with autocast():
                    outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                    val_loss = masked_loss(outputs, categories, masks, idx) # need to include idx in the loss function 
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Clear cache after validation epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Update scheduler
        scheduler.step()
        
    # Save the model
    torch.save(model.state_dict(), save_path + 'transformer.model')

    # Save the training and validation loss to CSV
    loss_df = pd.DataFrame({'epoch': [i for i in range(epochs)], 
                            'training losses': train_losses, 
                            'validation losses': val_losses}) 
    output_dir =  f'{save_path}'
    loss_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
def train_crossValidation(dataset, phrog_integer, n_splits=10, batch_size=16, epochs=10, lr=1e-5, save_path='out', num_heads=4, hidden_dim=512, device='cuda', dropout=0.1, mask_unknowns=True): 
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1

    for train_index, val_index in kf.split(dataset):
        print(f'FOLD: {fold}', flush=True)
        
        # Create output directory for the current fold
        print('Creating output directory', flush = True)
        output_dir = f'{save_path}/fold_{fold}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory created: {output_dir}")
        else:
            print(f"Warning: Directory {output_dir} already exists.")

        # Use SubsetRandomSampler for efficient subsetting
        print('generating subsamples', flush = True)
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)
        
        # Create data loaders with samplers
        print('Saving datasets', flush = True)
        train_kfold_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
        val_kfold_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn, pin_memory=True)
        pickle.dump(val_kfold_loader, open(output_dir + '/val_kfold_loader.pkl', 'wb'))

        # Initialize model
        input_dim = dataset[0][0].shape[1] 
        kfold_transformer_model = ImprovedSeq2SeqTransformerClassifier(input_dim=input_dim, num_classes=9, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
        kfold_transformer_model.to(device)

        # Train model
        print('Training the model', flush = True )
        train(kfold_transformer_model, train_kfold_loader, val_kfold_loader, epochs=epochs, lr=lr, save_path=output_dir, device=device, mask_unknowns=mask_unknowns)
        
        # Clear cache after training each fold
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate performance for this kfold
        evaluate(kfold_transformer_model, val_kfold_loader, phrog_integer, device, output_dir=output_dir)
        
        fold += 1

def evaluate(model, dataloader, phrog_integer, device, output_dir='metrics_output'):
    ## This here needs work. Could be done outside of this function 
    model.to(device)
    model.eval()
    labels = list(phrog_integer.keys())
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, categories, masks, idx in dataloader:
            embeddings, categories, masks = embeddings.to(device).float(), categories.to(device).long(), masks.to(device).float()
            src_key_padding_mask = (masks == 0)
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
        print('AUC for category ' + phrog_integer[label] + ': ' + str(roc_auc))

        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        roc_df.to_csv(os.path.join(output_dir, f'roc_curve_category_{phrog_integer[label]}.csv'), index=False)

    # Compute F1 score, precision, recall for each category
    all_preds = np.argmax(all_probs, axis=1)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=1)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=1)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=1)
        
    print('f1')
    print(f1)
    print('precision')
    print(precision)
    print('recall')
    print(recall)
    print('labels')
    print(labels)

    # Save F1, precision, recall to CSV
    #metrics_df = pd.DataFrame({
    #    'Category': [phrog_integer.get(i) for i in range(0,10)], 
    #    'F1': f1, 
    #    'Precision': precision, 
    #    'Recall': recall
    #})
    #metrics_df.to_csv(os.path.join(output_dir, 'metrics_with_optimal_thresholds.csv'), index=False)

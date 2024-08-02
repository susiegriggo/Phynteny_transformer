"""
Modules for building transformer model 
""" 

import torch.nn as nn 
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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
    def __init__(self, embeddings, categories, mask_token=-1):
        self.embeddings = [embedding.float() for embedding in embeddings]
        self.categories = [category.long() for category in categories]
        self.mask_token = mask_token
        self.num_classes = 10 # hard coded in for now 
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        category = self.categories[idx]
        mask = (category != self.mask_token).float()  # 1 for valid, 0 for missing - this indicates which rows to consider when training 
        
        # select a random cateogory to mask if training 
        if self.training: 
            masked_category = self.category_mask(category)
        else: 
            masked_category = category
            
        # use the masked category to generate a one-hot encoding 
        one_hot = self.custom_one_hot_encode(masked_category)
        
        # append the one_hot encoding to the front of the embedding 
        embedding_one_hot = torch.tensor(np.hstack((one_hot, embedding)))
     
        self.embedding = embedding_one_hot

        return embedding_one_hot, category, mask 
    
    def set_training(self, training=True):
        self.training = training
        
    def category_mask(self, category): 
        
        # select a category to randomly mask 
        idx = random.randint(0, len(category) - 1)
        while category[idx] == -1: 
            idx = random.randint(0, len(category) - 1)
        
        # generate masked versions 
        masked_category = category.clone()
        masked_category[idx] = self.mask_token
        
        return masked_category

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
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=512, dropout=0.1):
        super(VariableSeq2SeqTransformerClassifier, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, hidden_dim).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True).cuda()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).cuda()
        
        self.fc = nn.Linear(hidden_dim, num_classes).cuda()
    
    def forward(self, x, src_key_padding_mask=None):
        x = x.float()  # Ensure input is of type float
        x = self.embedding_layer(x)  # x: [batch_size, seq_len, input_dim]
        x = self.dropout(x)  # Apply dropout after embedding layer 
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # x: [batch_size, seq_len, hidden_dim]
        x = self.fc(x)  # x: [batch_size, seq_len, num_classes]

        return x
       
def collate_fn(batch):
    embeddings, categories, masks = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    categories_padded = pad_sequence(categories, batch_first=True, padding_value=-1)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return embeddings_padded, categories_padded, masks_padded
 
def masked_loss(output, target, mask, ignore_index=-1):
    target = target.clone()
    target[mask == 0] = ignore_index
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none').cuda()
    loss = loss_fct(output.view(-1, output.size(-1)), target.view(-1))
    loss = loss * mask.view(-1)
    return loss.sum() / mask.sum()

def train(model, train_dataloader, test_dataloader, epochs=5, lr=1e-5, save_path='model', device='cuda'):
    print('Training on ' + str(device), flush = True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = [] 
    val_losses = [] 
    model.train()
    total_loss = 0 
    
    print('Beginning training loop', flush = True)
    for epoch in range(epochs):
        total_loss = 0
        for embeddings, categories, masks in train_dataloader:
            embeddings, categories, masks = embeddings.to(device).float(), categories.to(device).long(), masks.to(device).float()
            optimizer.zero_grad()
            src_key_padding_mask = (masks == 0)  # Mask for transformer - referes to the masks tensor which is composed of 1s and 0s 
            outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
            loss = masked_loss(outputs, categories, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss}")

        # Clear cache after training epoch
        gc.collect()
        torch.cuda.empty_cache()

        # Validation loss 
        model.eval() 
        total_val_loss = 0 
        with torch.no_grad():
            for embeddings, categories, masks in test_dataloader:
                embeddings, categories, masks = embeddings.to(device).float(), categories.to(device).long(), masks.to(device).float()
                src_key_padding_mask = (masks == 0)  # Mask for transformer
                outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                val_loss = masked_loss(outputs, categories, masks)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Clear cache after validation epoch
        gc.collect()
        torch.cuda.empty_cache()
        
    # Save the model
    torch.save(model.state_dict(), save_path + 'transformer.model')

    # Save the training and validation loss to CSV
    loss_df = pd.DataFrame({'epoch': [i for i in range(epochs)], 
                            'training losses': train_losses, 
                            'validation losses': val_losses}) 
    output_dir =  f'{save_path}'
    loss_df.to_csv(os.path.join(output_dir, 'metrics.csv')
                   , index=False)

def train_crossValidation(dataset, phrog_integer, n_splits=10, batch_size=16, epochs=10, lr=1e-5, save_path='out', num_heads=4, hidden_dim=512, device='cuda', dropout=0.1): 
    
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
        print('Intialising the model', flush = True)
        input_dim = dataset[0][0].shape[1] 
        print(dataset[0][0].shape, flush = True) 
        print(input_dim, flush = True)
        kfold_transformer_model = VariableSeq2SeqTransformerClassifier(input_dim=input_dim, num_classes=9, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
        kfold_transformer_model.to(device)

        # Train model
        print('Training the model', flush = True )
        train(kfold_transformer_model, train_kfold_loader, val_kfold_loader, epochs=epochs, lr=lr, save_path=output_dir, device=device)
        
        # Clear cache after training each fold
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate performance for this kfold
        evaluate_with_optimal_thresholds(kfold_transformer_model, val_kfold_loader, phrog_integer, device, output_dir=output_dir)
        
        fold += 1

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    counter = 0 
    with torch.no_grad():
        for embeddings, categories, masks in dataloader:
            
            src_key_padding_mask = (masks == 0)  # Mask for transformer
            outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
            _, predicted = torch.max(outputs, 2) # 2 referes to the vlaue 
            
            
            counter += 1 
            
            valid = masks.byte() # specify which predictions are not masked 
            correct += (predicted == categories).masked_select(valid).sum().item()
            #print((predicted == categories).masked_select(valid).sum().item())
            total += valid.sum().item()
            
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    print("Total predictions: " + str(total))

def evaluate_with_threshold(model, dataloader, threshold=0.5):
    model.eval()
    correct = 0
    total_confident_predictions = 0
    
    with torch.no_grad():
        for embeddings, categories, masks in dataloader:
            embeddings = embeddings.float()
            src_key_padding_mask = (masks == 0)  # Mask for transformer
            outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
            
            # Apply softmax to get probabilitilsls
            es
            probs = F.softmax(outputs, dim=2)
            max_probs, predicted = torch.max(probs, dim=2)
            
            # Apply thresholding
            confident_predictions = max_probs >= threshold
            
            # Mask for valid and confident predictions
            valid = masks.byte()
            confident_and_valid = confident_predictions & valid
            
            correct += (predicted == categories).masked_select(confident_and_valid).sum().item()
            total_confident_predictions += confident_and_valid.sum().item()
    
    accuracy = correct / total_confident_predictions if total_confident_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confident Predictions: {total_confident_predictions}")

def evaluate_with_metrics_and_save(model, dataloader, threshold=0.5, output_dir='metrics_output'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, categories, masks in dataloader:
            embeddings = embeddings.float()
            src_key_padding_mask = (masks == 0)  # Mask for transformer
            outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
            
            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=2)
            max_probs, predicted = torch.max(probs, dim=2)
            
            valid = masks.byte()
            confident_predictions = max_probs >= threshold
            
            # Flatten arrays and filter out padding
            valid = valid.view(-1)
            probs = probs.view(-1, probs.size(-1)).cpu().numpy()
            predicted = predicted.view(-1).cpu().numpy()
            categories = categories.view(-1).cpu().numpy()
            
            confident_indices = confident_predictions.view(-1).cpu().numpy().astype(bool)
            valid_indices = valid.cpu().numpy().astype(bool)
            mask = confident_indices & valid_indices
            
            all_labels.extend(categories[mask])
            all_preds.extend(predicted[mask])
            all_probs.extend(probs[mask])
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Compute F1 score, precision, recall for each category
    f1 = f1_score(all_labels, all_preds, average=None)
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    
    # Save F1, precision, recall to CSV
    metrics_df = pd.DataFrame({'Category': np.arange(len(f1)), 'F1': f1, 'Precision': precision, 'Recall': recall})
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Save ROC curve data to CSV
    num_classes = all_probs.shape[1]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_df.to_csv(os.path.join(output_dir, f'roc_curve_category_{i}.csv'), index=False)
        print(f"Category {i} ROC AUC: {roc_auc:.4f}")

    print(f"F1 Scores per category: {f1}")
    print(f"Precision per category: {precision}")
    print(f"Recall per category: {recall}")

    # Modified evaluation function to include ROC curve and metrics calculation

def evaluate_with_optimal_thresholds(model, dataloader, phrog_integer, device, output_dir='metrics_output'):
    """
    Threshold is selected that optimizes Youden's J index 
    """
    model.to(device)
    model.eval()
    all_labels = []
    all_probs = []
    

    # TODO - this evaluate is incorrect - work through this 
    with torch.no_grad():
        for embeddings, categories, masks in dataloader:
            embeddings, categories, masks = embeddings.to(device).float(), categories.to(device).long(), masks.to(device).float()
            src_key_padding_mask = (masks == 0)  # Mask for transformer
            outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
            
            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=2)
            max_probs, predicted = torch.max(probs, dim=2)
            
            valid = masks.byte()
            
            # Flatten arrays and filter out padding
            valid = valid.view(-1)
            probs = probs.view(-1, probs.size(-1)).cpu().numpy()
            predicted = predicted.view(-1).cpu().numpy()
            categories = categories.view(-1).cpu().numpy()
            
            valid_indices = valid.cpu().numpy().astype(bool)
            
            all_labels.extend(categories[valid_indices])
            all_probs.extend(probs[valid_indices])
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Ensure there are predictions to evaluate
    if len(all_labels) == 0 or len(all_probs) == 0:
        print("No predictions to evaluate.")
        return
    
    labels = list(set(all_labels))
    optimal_thresholds = np.zeros(len(labels))
    retained_proportion = np.zeros(len(labels))

    for idx, label in enumerate(labels):
        fpr, tpr, thresholds = roc_curve(all_labels == label, all_probs[:, idx])
        roc_auc = auc(fpr, tpr)

        # Find the optimal threshold using Youden's J statistic (maximizing (sensitivity + specificity - 1))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thresholds[idx] = thresholds[optimal_idx]

        # Calculate the retained proportion of predictions above the optimal threshold
        total_predictions = len(all_probs[:, idx])
        retained_predictions = np.sum(all_probs[:, idx] >= optimal_thresholds[idx])
        retained_proportion[idx] = retained_predictions / total_predictions

        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        roc_df.to_csv(os.path.join(output_dir, f'roc_curve_category_{phrog_integer.get(label)}.csv'), index=False)
        print(f"Category: {phrog_integer.get(label)} \n\tROC AUC: {roc_auc:.4f}, Optimal Threshold: {optimal_thresholds[idx]:.4f}")

    # Use optimal thresholds to make final predictions
    final_preds = (all_probs >= optimal_thresholds).astype(int)
    
    # Compute F1 score, precision, recall for each category using optimal thresholds
    f1 = f1_score(all_labels, final_preds.argmax(axis=1), average=None, zero_division=1)
    precision = precision_score(all_labels, final_preds.argmax(axis=1), average=None, zero_division=1)
    recall = recall_score(all_labels, final_preds.argmax(axis=1), average=None, zero_division=1)
    
    # Save F1, precision, recall to CSV
    metrics_df = pd.DataFrame({
        'Category': [phrog_integer.get(label) for label in labels], 
        'F1': f1, 
        'Precision': precision, 
        'Recall': recall, 
        'Optimal Threshold': optimal_thresholds,
        'Proportion Retained': retained_proportion
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_with_optimal_thresholds.csv'), index=False)

    print(f"F1 Scores per category: {f1}")
    print(f"Precision per category: {precision}")
    print(f"Recall per category: {recall}")
    print(f"Proportion of predictions retained: {retained_proportion}")

"""
Modules for building transformer model 
""" 

import torch.nn as nn 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import os

class VariableSeq2SeqEmbeddingDataset(Dataset):
    def __init__(self, embeddings, categories, mask_token=-1):
        self.embeddings = [embedding.float() for embedding in embeddings]
        self.categories = [category.long() for category in categories]
        self.mask_token = mask_token
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        category = self.categories[idx]
        mask = (category != self.mask_token).float()  # 1 for valid, 0 for missing
        return embedding, category, mask 

class VariableSeq2SeqTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=512):
        super(VariableSeq2SeqTransformerClassifier, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, src_key_padding_mask=None):
        x = x.float()  # Ensure input is of type float
        x = self.embedding_layer(x)  # x: [batch_size, seq_len, input_dim]
        x = x.permute(1, 0, 2)  # x: [seq_len, batch_size, hidden_dim]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # x: [seq_len, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)  # x: [batch_size, seq_len, hidden_dim]
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
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    loss = loss_fct(output.view(-1, output.size(-1)), target.view(-1))
    loss = loss * mask.view(-1)
    return loss.sum() / mask.sum()

def train(model, train_dataloader, test_dataloader, epochs=5, lr=1e-5, save_path='model.pth'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = [] 
    val_losses = [] 
    model.train()
    total_loss = 0 
    
    for epoch in range(epochs):
        total_loss = 0
        for embeddings, categories, masks in train_dataloader:
            embeddings, categories, masks = embeddings.float(), categories.long(), masks.float()
            optimizer.zero_grad()
            src_key_padding_mask = (masks == 0)  # Mask for transformer
            outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
            loss = masked_loss(outputs, categories, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss}")

        # Validation loss 
        model.eval() 
        total_val_loss = 0 
        with torch.no_grad():
            for embeddings, categories, masks in test_dataloader:
                embeddings, categories, masks = embeddings.float(), categories.long(), masks.float()
                src_key_padding_mask = (masks == 0)  # Mask for transformer
                outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                val_loss = masked_loss(outputs, categories, masks)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

    # Save the model
    torch.save(model.state_dict(), save_path)

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
            
            # Apply softmax to get probabilities
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
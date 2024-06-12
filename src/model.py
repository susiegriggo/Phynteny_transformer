"""
Modules for building transformer model 
""" 

import torch.nn as nn 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim

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
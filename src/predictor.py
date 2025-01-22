""" 
Module for handling predictions with torch 
""" 

from src import model_onehot
import torch
import torch.nn.functional as F
import os
import numpy as np 
from loguru import logger

class Predictor: 

    def __init__(self, device = 'cuda'): 
        self.device = torch.device(device)
        self.models = []

    def predict(self, X, src_key_padding):
        """
        Predict the scores for the given input data.

        :param X: Dictionary of input data
        :param src_key_padding: Padding mask for the input data
        :return: Dictionary of predicted scores
        """
        self.models = [model.to(self.device) for model in self.models]
        for model in self.models:
            model.eval()

        all_scores = {}

        # Convert input data to tensors
        X_tensor = torch.stack([X[key] for key in X.keys()]).to(self.device)
        src_key_padding_tensor = torch.tensor(src_key_padding).to(self.device)

        with torch.no_grad():
            for model in self.models:
                outputs = model(X_tensor, src_key_padding_mask=src_key_padding_tensor)
                outputs = F.softmax(outputs, dim=-1)
                for i, key in enumerate(X.keys()):
                    if key not in all_scores:
                        all_scores[key] = outputs[i].cpu().numpy()
                    else:
                        all_scores[key] += outputs[i].cpu().numpy()

        return all_scores

    def read_model(self, model_path): 

        # Read in the model 
        model = torch.load(model_path, map_location=torch.device(self.device))

        # build the model into a predictor object 
        #input_dim = model.get('func_embedding.weight').shape[1] + model.get('strand_embedding.weight').shape[1] + model.get('length_embedding.weight')[1] + model.get('embedding_layer.weight').shape[1]
        input_dim = 1293 # this is hardcoded for now - I can't figure out any other way around it 
        num_classes = model.get('fc.weight').shape[0]
        hidden_dim = model.get('lstm.weight_hh_l0').shape[1]
        d_model = hidden_dim*2 # because the lstm layer is bidirectional
        num_heads = d_model // model.get('transformer_encoder.layers.0.self_attn.relative_position_k').shape[1] 
        dropout = 0.1 # set this to an arbitrary value - doesn't matter if the model isn't in training mode 
        print(f"Model parameters: input_dim: {input_dim}, num_classes: {num_classes}, hidden_dim: {hidden_dim}, num_heads: {num_heads}, dropout: {dropout}")
        
        # Check if dropout layer exists
        #if 'transformer.encoder.layers.0.self_attn.dropout' in model:
        #    dropout = model.get('transformer.encoder.layers.0.self_attn.dropout').item()
        #else:
        #    raise ValueError("Dropout layer not found in the model")

        # Create the predictor option 
        predictor = model_onehot.Seq2SeqTransformerClassifierCircularRelativeAttention( # Could change to read in different types of models as well 
            input_dim=input_dim, 
            num_classes=num_classes, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            dropout=dropout, 
        )
    
        predictor.load_state_dict(model)

        return predictor

    def read_models_from_directory(self, directory_path):

        print('Reading models from directory: ', directory_path)

        for filename in os.listdir(directory_path):
            if filename.endswith(".model"):  # Assuming model files have .pt extension
                model_path = os.path.join(directory_path, filename)
                model = self.read_model(model_path)
                self.models.append(model)








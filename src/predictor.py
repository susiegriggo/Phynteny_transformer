""" 
Module for handling predictions with torch 
""" 

from src import model_onehot
import torch
import torch.nn.functional as F
import os
import numpy as np 

class Predictor: 

    def __init__(self, device = 'cpu'): 
        self.device = device
        self.models = []

    def predict(self): 
        # list to store predictions
        scores = []

        # Loop through the data and make predictions using each model
        for data_point, src_key_padding in zip(self.data, self.src_key_padding):
            model_probs = []
            for model in self.models: 
                model.eval()
                model.to(self.device)
                with torch.no_grad(): 
                    prediction = model(data_point, src_key_padding)
                    probs = F.softmax(prediction, dim=2)
                    model_probs.append(probs)
            np.array(model_probs).sum(axis=0)
        
        return scores
    


    def read_model(self, model_path): 

        # Read in the model 
        model = torch.load(model_path, map_location=torch.device(self.device))

        # build the model into a predictor object 
        input_dim = model.get('lstm.weight_ih_l0').shape[1]
        num_classes = model.get('fc.weight').shape[0]
        hidden_dim = model.get('lstm.weight_hh_l0').shape[1]
        d_model = hidden_dim*2 # because the lstm layer is bidirectional
        num_heads = d_model // model.get('transformer.encoder.layers.0.self_attn.relative_position_k').shape[1] 
        dropout = 0.1 # set this to an arbitrary value - doesn't matter if the model isn't in training mode 
        
        # Check if dropout layer exists
        if 'transformer.encoder.layers.0.self_attn.dropout' in model:
            dropout = model.get('transformer.encoder.layers.0.self_attn.dropout').item()
        else:
            raise ValueError("Dropout layer not found in the model")

        # Create the predictor option 
        predictor = model_onehot.Seq2SeqTransformerClassifierCircularRelativeAttention( # Could change to read in different types of models as well 
            input_dim=input_dim, 
            num_classes=num_classes, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            dropout=dropout, 
        )
        predictor.load_state_dict(model)

    def read_models_from_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith(".pt"):  # Assuming model files have .pt extension
                model_path = os.path.join(directory_path, filename)
                model = self.read_model(model_path)
                self.models.append(model)








""" 
Module for handling predictions with torch 
""" 

from src import model_onehot
from src import format_data
import torch
import torch.nn.functional as F
import os
import numpy as np 
from loguru import logger
from sklearn.neighbors import KernelDensity

class Predictor: 

    def __init__(self, device = 'cuda'): 
        self.device = torch.device(device)
        self.models = []

    def predict_batch(self, embeddings, src_key_padding_mask):

        self.models = [model.to(self.device) for model in self.models]
        for model in self.models:
            model.eval()

        all_scores = {}

        with torch.no_grad():
            for model in self.models:
                outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                outputs = F.softmax(outputs, dim=-1)
                
                if len(all_scores) == 0:
                    all_scores = outputs.cpu().numpy()
                else:
                    all_scores += outputs.cpu().numpy()
        return all_scores   


    def predict(self, X, y): # not sure if this is right 
        """
        Predict the scores for the given input data.

        :param X: List of input data values
        :param y: List of target data values
        :return: Dictionary of predicted scores
        """
        self.models = [model.to(self.device) for model in self.models]
        for model in self.models:
            model.eval()

        all_scores = {}

        # Find the maximum length of the input sequences
        max_length = max([v.shape[0] for v in X])

        # Pad the input sequences to the same length
        X_padded = [F.pad(x, (0, 0, 0, max_length - x.shape[0])) for x in X]

        # Convert input data to tensors
        X_tensor = torch.stack(X_padded).to(self.device)

        # Compute src_key_padding using format_data.pad_sequence
        src_key_padding = format_data.pad_sequence(y)
        src_key_padding_tensor = torch.tensor(src_key_padding).to(self.device)

        with torch.no_grad():
            for model in self.models:
                outputs = model(X_tensor, src_key_padding_mask=src_key_padding_tensor)
                outputs = F.softmax(outputs, dim=-1)
                for i, key in enumerate(y.keys()):
                    if key not in all_scores:
                        all_scores[key] = outputs[i].cpu().numpy()
                    else:
                        all_scores[key] += outputs[i].cpu().numpy()

        return all_scores

    def read_model(self, model_path): 
        # Read in the model 
        model = torch.load(model_path, map_location=torch.device(self.device))

        # Dynamically determine input_dim from the model
        input_dim = model.get('embedding_layer.weight').shape[1]
        num_classes = model.get('fc.weight').shape[0]
        lstm_hidden_dim = model.get('lstm.weight_hh_l0').shape[1]  # Read lstm_hidden_dim
        d_model = lstm_hidden_dim * 2  # because the lstm layer is bidirectional
        num_heads = d_model // model.get('transformer_encoder.layers.0.self_attn.relative_position_k').shape[1]
        dropout = 0.1  # set this to an arbitrary value - doesn't matter if the model isn't in training mode
        max_len = model.get('transformer_encoder.layers.1.self_attn.relative_position_k').shape[0]

        print(f"Model parameters: input_dim: {input_dim}, num_classes: {num_classes}, lstm_hidden_dim: {lstm_hidden_dim}, num_heads: {num_heads}, dropout: {dropout}, max_len: {max_len}")

        # Create the predictor option 
        predictor = model_onehot.TransformerClassifierCircularRelativeAttention(
            input_dim=input_dim, 
            num_classes=num_classes, 
            num_heads=num_heads, 
            hidden_dim=lstm_hidden_dim,  # Use lstm_hidden_dim
            lstm_hidden_dim=lstm_hidden_dim,  # Pass lstm_hidden_dim
            dropout=dropout, 
            max_len=max_len  # Specify max_len
        )
    
        # Resize model parameters to match the checkpoint
        state_dict = model
        for name, param in predictor.state_dict().items():
            if name in state_dict:
                if state_dict[name].shape != param.shape:
                    state_dict[name] = param
        predictor.load_state_dict(state_dict)

        return predictor

    def read_models_from_directory(self, directory_path):

        print('Reading models from directory: ', directory_path)

        for filename in os.listdir(directory_path):
            if filename.endswith(".model"):  # Assuming model files have .pt extension
                model_path = os.path.join(directory_path, filename)
                model = self.read_model(model_path)
                self.models.append(model)



########
# Confidence scoring functions
########
def count_critical_points(arr):
    return np.sum(np.diff(np.sign(np.diff(arr))) != 0)
 

def build_confidence_dict(label, prediction, scores, bandwidth, categories):

    # range over values to compute kernel density over
    vals = np.arange(1.5, 10, 0.001)

    # save a dictionary which contains all the information required to compute confidence scores
    confidence_dict = dict()

    # loop through the categories
    print("Computing kernel denisty for each category...")
    for cat in range(0, 9):
        print("processing " + str(cat))

        # fetch the true labels of the predictions of this category
        this_labels = label[prediction == cat]

        # fetch the scores associated with these predictions
        this_scores = scores[prediction == cat]

        # separate false positives and true positives
        TP_scores = this_scores[this_labels == cat]
        FP_scores = this_scores[this_labels != cat]

        # loop through potential bandwidths
        for b in bandwidth:
            # compute the kernel density
            kde_TP = KernelDensity(kernel="gaussian", bandwidth=b)
            kde_TP.fit(TP_scores[:, cat].reshape(-1, 1))
            e_TP = np.exp(kde_TP.score_samples(vals.reshape(-1, 1)))

            kde_FP = KernelDensity(kernel="gaussian", bandwidth=b)
            kde_FP.fit(FP_scores[:, cat].reshape(-1, 1))
            e_FP = np.exp(kde_FP.score_samples(vals.reshape(-1, 1)))

            conf_kde = (e_TP * len(TP_scores)) / (
                e_TP * len(TP_scores) + e_FP * len(FP_scores)
            )

            if count_critical_points(conf_kde) <= 1:
                break

        # save the best estimators
        confidence_dict[categories.get(cat)] = {
            "kde_TP": kde_TP,
            "kde_FP": kde_FP,
            "num_TP": len(TP_scores),
            "num_FP": len(FP_scores),
            "bandwidth": b,
        }

    return confidence_dict

def compute_confidence(scores, confidence_dict, categories):
    """
    Function which computes the confidence of a Phynteny prediction
    Input is a vector of Phynteny scores
    """

    # get the prediction for each score
    score_predictions = np.array([np.argmax(score) for idx, score in enumerate(scores)])

    # make an array to store the confidence of each prediction
    confidence_out = np.zeros(len(scores))
    predictions_out = np.zeros(len(scores))

    # loop through each of potential categories
    for i in range(0, 9):
        # get the scores relevant to the current category
        cat_scores = np.array(scores)[score_predictions == i]

        if len(cat_scores) > 0:
            # compute the kernel density estimates
            e_TP = np.exp(
                confidence_dict.get(categories.get(i))
                .get("kde_TP")
                .score_samples(cat_scores[:, i].reshape(-1, 1))
            )
            e_FP = np.exp(
                confidence_dict.get(categories.get(i))
                .get("kde_FP")
                .score_samples(cat_scores[:, i].reshape(-1, 1))
            )

            # fetch the number of TP and FP
            num_TP = confidence_dict.get(categories.get(i)).get("num_TP")
            num_FP = confidence_dict.get(categories.get(i)).get("num_FP")

            # compute the confidence scores
            conf_kde = (e_TP * num_TP) / (e_TP * num_TP + e_FP * num_FP)

            # save the scores to the output vector
            confidence_out[score_predictions == i] = conf_kde
            predictions_out[score_predictions == i] = [i for k in range(len(conf_kde))]

    return predictions_out, confidence_out




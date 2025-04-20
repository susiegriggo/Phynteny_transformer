""" 
Module for handling predictions with torch 
""" 

from src import model_onehot
from src import format_data
import torch
import torch.nn.functional as F
import os
import numpy  as np 
from sklearn.neighbors import KernelDensity
from Bio import SeqIO  # Add this import
from loguru import logger  # Add this import
from torch.utils.data import DataLoader  # Add this import at the top of the file

class Predictor: 

    def __init__(self, device = 'cuda'): 
        self.device = torch.device(device)
        self.models = []

    def predict_batch(self, embeddings, src_key_padding_mask):
        """
        Predict the scores for a batch of embeddings.

        :param embeddings: Tensor of input embeddings
        :param src_key_padding_mask: Mask tensor for padding in the input sequences
        :return: Dictionary of predicted scores
        """
        self.models = [model.to(self.device) for model in self.models]
        embeddings = embeddings.to(self.device)  # Move embeddings to device
        src_key_padding_mask = src_key_padding_mask.to(self.device)  # Move mask to device

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

        self.scores = all_scores
        return all_scores  # Remove the TODO comment if the return statement is necessary
    
    def predict_inference(self, X, y, batch_size=128):
        """
        Predict using batch processing for inference.
        
        :param X: Dictionary of embeddings with keys as identifiers
        :param y: Dictionary of categories with keys as identifiers
        :param batch_size: Size of batches for processing
        :return: Dictionary of predicted scores with keys matching input dictionaries
        """
        logger.info(f"Starting batch inference with batch size {batch_size}")
        
        # Prepare for batch processing
        self.models = [model.to(self.device) for model in self.models]
        for model in self.models:
            model.eval()
        
        # Create dataset
        keys_list = list(X.keys())
        dataset = model_onehot.EmbeddingDataset(
            list(X.values()), 
            list(y.values()), 
            keys_list,
            mask_portion=0  # No masking for inference
        )
        
        # Set inference mode
        dataset.training = False
        dataset.validation = False
        
        # Create DataLoader
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=model_onehot.collate_fn
        )
        
        logger.info(f"Processing {len(keys_list)} samples in batches of {batch_size}")
        
        # Process batches
        all_scores = {}
        batches_processed = 0
        
        for embeddings_batch, categories_batch, masks_batch, idx_batch in data_loader:
            try:
                # Move tensors to device
                embeddings_batch = embeddings_batch.to(self.device)
                masks_batch = masks_batch.to(self.device)
                src_key_padding_mask = (masks_batch != -2).to(self.device)
                
                # Get predictions for this batch
                batch_scores = self.predict_batch(embeddings_batch, src_key_padding_mask)
                
                # Store scores by their original keys
                for i in range(len(idx_batch)):
                    if i < len(keys_list):
                        key = keys_list[i]
                        
                        if isinstance(batch_scores, torch.Tensor):
                            if i < batch_scores.shape[0]:
                                all_scores[key] = batch_scores[i].cpu().numpy()
                        else:
                            # It's already a numpy array
                            if i < len(batch_scores):
                                all_scores[key] = batch_scores[i]
                
                batches_processed += 1
                if batches_processed % 10 == 0:
                    logger.info(f"Processed {batches_processed} batches...")
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"Total batches processed: {batches_processed}")
        logger.info(f"Collected scores for {len(all_scores)} samples")
        
        # Store scores for compatibility with other methods
        self.scores = all_scores
        return all_scores

    def write_genbank(self,  gb_dict, out):
        """
        Write the predicted scores to a Genbank file.

        :param gb_dict: Dictionary of Genbank records
        :param out: Output directory
        """
        for key in self.scores.keys():
            record = gb_dict[key]
            record.annotations["phynteny score"] = self.scores[key]
            SeqIO.write(record, os.path.join(out, key + ".gb"), "genbank")

    def read_model(self, model_path, input_dim, num_classes, num_heads, hidden_dim, lstm_hidden_dim, dropout, use_lstm, max_len, protein_dropout_rate=0.0, attention='circular', positional_encoding_type='fourier', pre_norm=False, progressive_dropout=False, initial_dropout_rate=1.0, final_dropout_rate=0.4, output_dim=None, num_layers=2): 
        """
        Read and load a model from the given path.

        :param model_path: Path to the model file
        :param input_dim: Input dimension for the model
        :param num_classes: Number of classes for the model
        :param num_heads: Number of attention heads for the model
        :param hidden_dim: Hidden dimension for the model
        :param lstm_hidden_dim: LSTM hidden dimension for the model
        :param dropout: Dropout rate for the model
        :param use_lstm: Whether to use LSTM in the model
        :param max_len: Maximum length for the model
        :param protein_dropout_rate: Dropout rate for protein features
        :param attention: Type of attention mechanism
        :param positional_encoding_type: Type of positional encoding
        :param pre_norm: Whether to use pre-normalization
        :param progressive_dropout: Whether to use progressive dropout
        :param initial_dropout_rate: Initial dropout rate for progressive dropout
        :param final_dropout_rate: Final dropout rate for progressive dropout
        :param output_dim: Output dimension for the model
        :param num_layers: Number of transformer layers
        :return: Loaded model
        """
        # Default output_dim to num_classes if not specified
        if output_dim is None:
            output_dim = num_classes
        
        # Read in the model state dict
        state_dict = torch.load(model_path, map_location=torch.device(self.device))

        logger.info(f'Reading model with input_dim={input_dim}, num_classes={num_classes}, output_dim={output_dim}, num_heads={num_heads}, hidden_dim={hidden_dim}, lstm_hidden_dim={lstm_hidden_dim}, dropout={dropout}, use_lstm={use_lstm}, max_len={max_len}, protein_dropout_rate={protein_dropout_rate}, attention={attention}, positional_encoding_type={positional_encoding_type}, pre_norm={pre_norm}, num_layers={num_layers}')

        # Select the positional encoding function based on the type
        positional_encoding_func = model_onehot.fourier_positional_encoding if positional_encoding_type == 'fourier' else model_onehot.sinusoidal_positional_encoding
        
        # Create the appropriate model based on attention type
        if attention == 'circular':
            model = model_onehot.TransformerClassifierCircularRelativeAttention(
                input_dim=input_dim, 
                num_classes=num_classes, 
                num_heads=num_heads, 
                hidden_dim=hidden_dim, 
                lstm_hidden_dim=lstm_hidden_dim if use_lstm else None,
                dropout=dropout, 
                max_len=max_len,
                use_lstm=use_lstm, 
                positional_encoding=positional_encoding_func,
                protein_dropout_rate=protein_dropout_rate,
                pre_norm=pre_norm,
                progressive_dropout=progressive_dropout,
                initial_dropout_rate=initial_dropout_rate,
                final_dropout_rate=final_dropout_rate,
                output_dim=output_dim
            )
        elif attention == 'relative':
            model = model_onehot.TransformerClassifierRelativeAttention(
                input_dim=input_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim if use_lstm else None,
                dropout=dropout,
                max_len=max_len,
                use_lstm=use_lstm,
                positional_encoding=positional_encoding_func,
                protein_dropout_rate=protein_dropout_rate,
                output_dim=output_dim
            )
        elif attention == 'absolute':
            model = model_onehot.TransformerClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim if use_lstm else None,
                dropout=dropout,
                num_layers=num_layers,  # Use the passed parameter instead of hardcoded value
                use_lstm=use_lstm,
                positional_encoding=positional_encoding_func,
                protein_dropout_rate=protein_dropout_rate,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Invalid attention type: {attention}")
        
        # Initialize protein feature dropout with dummy forward pass
        if hasattr(model, 'protein_feature_dropout'):
            logger.info("Initializing protein_feature_dropout buffers with dummy forward pass")
            batch_size, seq_len = 2, 10  # Minimal batch for initialization
            feature_dim = model.gene_feature_dim + (hidden_dim - model.gene_feature_dim)
            dummy_input = torch.zeros((batch_size, seq_len, feature_dim), device=self.device)
            dummy_idx = [torch.tensor([0, 1]) for _ in range(batch_size)]
            model.protein_feature_dropout(dummy_input, dummy_idx)
        
        # Load the state dictionary with strict=False to handle missing/unexpected keys
        model.load_state_dict(state_dict, strict=False)
        
        return model

    def read_models_from_directory(self, directory_path, input_dim=1280, num_classes=9, num_heads=8, hidden_dim=512, lstm_hidden_dim=512, dropout=0.1, use_lstm=True, max_len=1500, protein_dropout_rate=0.6, attention='circular', positional_encoding_type='sinusoidal', pre_norm=False, progressive_dropout=True, initial_dropout_rate=1.0, final_dropout_rate=0.4, output_dim=None, num_layers=2):
        """
        Read and load all models from the given directory.

        :param directory_path: Path to the directory containing model files
        :param input_dim: Input dimension for the model
        :param num_classes: Number of classes for the model
        :param num_heads: Number of attention heads for the model
        :param hidden_dim: Hidden dimension for the model
        :param lstm_hidden_dim: LSTM hidden dimension for the model
        :param dropout: Dropout rate for the model
        :param use_lstm: Whether to use LSTM in the model
        :param max_len: Maximum length for the model
        :param protein_dropout_rate: Dropout rate for protein features
        :param attention: Type of attention mechanism
        :param positional_encoding_type: Type of positional encoding
        :param pre_norm: Whether to use pre-normalization
        :param progressive_dropout: Whether to use progressive dropout
        :param initial_dropout_rate: Initial dropout rate for progressive dropout
        :param final_dropout_rate: Final dropout rate for progressive dropout
        :param output_dim: Output dimension for the model
        :param num_layers: Number of transformer layers
        """
        print('Reading models from directory: ', directory_path)
        
        # Set default output_dim to num_classes if not specified
        if output_dim is None:
            output_dim = num_classes
        
        # Log parameters for debugging
        logger.info(f"Loading models with parameters: input_dim={input_dim}, num_classes={num_classes}, output_dim={output_dim}, " 
                    f"num_heads={num_heads}, hidden_dim={hidden_dim}, lstm_hidden_dim={lstm_hidden_dim}, "
                    f"dropout={dropout}, use_lstm={use_lstm}, max_len={max_len}, "
                    f"protein_dropout_rate={protein_dropout_rate}, attention={attention}, "
                    f"positional_encoding_type={positional_encoding_type}, pre_norm={pre_norm}")

        for filename in os.listdir(directory_path):
            if filename.endswith(".model"):  # Assuming model files have .model extension
                model_path = os.path.join(directory_path, filename)
                model = self.read_model(
                    model_path, input_dim, num_classes, num_heads, hidden_dim, 
                    lstm_hidden_dim, dropout, use_lstm, max_len, protein_dropout_rate,
                    attention, positional_encoding_type, pre_norm, progressive_dropout,
                    initial_dropout_rate, final_dropout_rate, output_dim, num_layers
                )
                self.models.append(model)


    def compute_confidence(self, scores, confidence_dict, categories):
        """
        Compute the confidence of a Phynteny prediction.

        :param scores: List of Phynteny scores
        :param confidence_dict: Dictionary containing confidence information
        :param categories: Dictionary of categories
        :return: Tuple of predictions and confidence scores
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


########
# Confidence scoring functions
########
def count_critical_points(arr):
    """
    Count the number of critical points in an array.

    :param arr: Input array
    :return: Number of critical points
    """
    return np.sum(np.diff(np.sign(np.diff(arr))) != 0)
 

def build_confidence_dict(label, prediction, scores, bandwidth, categories):
    """
    Build a dictionary containing confidence information for each category.

    :param label: Array of true labels
    :param prediction: Array of predicted labels
    :param scores: Array of scores
    :param bandwidth: List of bandwidth values for kernel density estimation
    :param categories: Dictionary of categories
    :return: Dictionary containing confidence information
    """
    # range over values to compute kernel density over
    vals = np.arange(1.5, 10, 0.001)

    # save a dictionary which contains all the information required to compute confidence scores
    confidence_dict = dict()

    # loop through the categories
    print("Computing kernel density for each category...")
    for cat in range(0, 9):
        print(f"Processing category {cat}")

        # fetch the true labels of the predictions of this category
        this_labels = label[prediction == cat]

        # fetch the scores associated with these predictions
        this_scores = scores[prediction == cat]

        # separate false positives and true positives
        TP_scores = this_scores[this_labels == cat]
        FP_scores = this_scores[this_labels != cat]

        print(f"Category {cat}: TP_scores shape: {TP_scores.shape}, FP_scores shape: {FP_scores.shape}")

        if TP_scores.shape[0] == 0 or FP_scores.shape[0] == 0:
            print(f"Skipping category {cat} due to insufficient data.")
            continue

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





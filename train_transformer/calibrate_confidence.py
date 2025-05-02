import click
import torch
from torch.utils.data import DataLoader
from src import predictor, model_onehot
from loguru import logger
from joblib import load, dump
import numpy as np
import os
import pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

# Parameters
PARAMETERS = {
    'model_directory': 'Directory containing the models.',
    'embeddings_path': 'Path to the embeddings file.',
    'categories_path': 'Path to the categories file.',
    'validation_categories_path': 'Path to the validation categories file.',
    'integer_category_path': 'Path to the integer category file.',
    'output_path': 'Path to save the output calibration models.',
    'force': 'Overwrite output directory if it exists.',
    'batch_size': 'Batch size for processing the data.'
}

logger.info("Parameters used in the script:")
for param, desc in PARAMETERS.items():
    logger.info(f"{param}: {desc}")

@click.command()
@click.option('--models', 'model_directory', type=click.Path(exists=True), help=PARAMETERS['model_directory'])
@click.option('--X', 'embeddings_path', type=click.Path(exists=True), help=PARAMETERS['embeddings_path'])
@click.option('--y', 'categories_path', type=click.Path(exists=True), help=PARAMETERS['categories_path'])
@click.option('--y-validation', 'validation_categories_path', type=click.Path(exists=True), help=PARAMETERS['validation_categories_path'])
@click.option('--integer-categories', 'integer_category_path', type=click.Path(exists=True), help=PARAMETERS['integer_category_path'])
@click.option('-o', '--output', 'output_path', type=click.Path(), help=PARAMETERS['output_path'])
@click.option('-f', '--force', is_flag=True, help=PARAMETERS['force'])
@click.option('--batch-size', default=128, help=PARAMETERS['batch_size'])
@click.option('--input-dim', default=1280, help='Input dimension for the model.')
@click.option('--num-classes', default=9, help='Number of classes for the model.')
@click.option('--num-heads', default=4, help='Number of attention heads for the model.')
@click.option('--hidden-dim', default=256, help='Hidden dimension for the model.')
@click.option('--lstm-hidden-dim', default=512, help='LSTM hidden dimension for the model.')
@click.option('--no-lstm', is_flag=True, help='Specify if LSTM should not be used in the model.')
@click.option('--max-len', default=1500, help='Maximum length for the model.')
@click.option('--attention', type=click.Choice(['absolute', 'relative', 'circular']), default='circular', help='Type of attention mechanism to use.')
@click.option('--positional-encoding-type', type=click.Choice(['fourier', 'sinusoidal']), default='fourier', help='Type of positional encoding to use.')
@click.option('--pre-norm', is_flag=True, default=False, help='Use pre-normalization in transformer layers instead of post-normalization.')
@click.option('--output-dim', default=None, type=int, help='Output dimension for the model. Defaults to num_classes if not specified.')
@click.option('--num-layers', default=2, type=int, help='Number of transformer layers in the model.')
@click.option('--dropout', default=0.0, help='Dropout rate for the model. Should be 0 for inference.', type=float)
@click.option('--protein-dropout-rate', default=0.0, help='Dropout rate for protein features. Should be 0 for inference.', type=float)
@click.option('--progressive-dropout', is_flag=True, default=False, help='Enable progressive dropout for protein features.')
@click.option('--initial-dropout-rate', default=1.0, help='Initial dropout rate when using progressive dropout.', type=float)
@click.option('--final-dropout-rate', default=0.4, help='Final dropout rate when using progressive dropout.', type=float)

def main(model_directory, embeddings_path, categories_path, validation_categories_path, integer_category_path, output_path, force, batch_size, input_dim, num_classes, num_heads, hidden_dim, lstm_hidden_dim, no_lstm, max_len, attention, positional_encoding_type, pre_norm, output_dim, num_layers, dropout, protein_dropout_rate, progressive_dropout, initial_dropout_rate, final_dropout_rate):
    """
    Calibrate confidence scores for model predictions using Isotonic Regression
    """
    logger.info("Starting confidence calibration with isotonic regression")
    
    # Check if output directory exists
    if os.path.exists(output_path) and not force:
        logger.error(f"Output directory {output_path} already exists. Use --force to overwrite.")
        return
    elif os.path.exists(output_path) and force:
        logger.info(f"Output directory {output_path} exists, overwriting...")
    else:
        os.makedirs(output_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    embeddings, categories, validation_categories, phrog_integer = load_data(
        embeddings_path, categories_path, validation_categories_path, integer_category_path
    )
    
    # Create predictor
    p = create_predictor(
        model_directory, device, input_dim, num_classes, num_heads, 
        hidden_dim, lstm_hidden_dim, dropout, not no_lstm, max_len, 
        protein_dropout_rate, attention, positional_encoding_type, pre_norm,
        progressive_dropout, initial_dropout_rate, final_dropout_rate, output_dim, num_layers
    )
    
    # Create dataset for calibration
    conf_dataset_loader = create_dataloader(embeddings, categories, validation_categories, batch_size)
    
    # Process data in batches to get predictions
    all_probs, all_labels = process_batches(p, conf_dataset_loader, device)
    
    # Calibrate predictions with isotonic regression
    calibration_models, calibration_stats = calibrate_probabilities(
        all_probs, all_labels, phrog_integer, num_classes
    )
    
    # Save calibration models and statistics
    save_calibration_models(calibration_models, calibration_stats, output_path, phrog_integer)
    
    logger.info("Calibration completed successfully")

def load_data(embeddings_path, categories_path, validation_categories_path, integer_category_path):
    """
    Load data from specified paths
    """
    logger.info("Loading data...")
    
    # Load embeddings
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    logger.info(f"Loaded embeddings: {len(embeddings)} items")
    
    # Load categories
    with open(categories_path, 'rb') as f:
        categories = pickle.load(f)
    logger.info(f"Loaded categories: {len(categories)} items")
    
    # Load validation categories
    with open(validation_categories_path, 'rb') as f:
        validation_categories = pickle.load(f)
    logger.info(f"Loaded validation categories: {len(validation_categories)} items")
    
    # Load integer categories mapping
    with open(integer_category_path, 'rb') as f:
        phrog_integer = pickle.load(f)
    logger.info(f"Loaded integer categories mapping: {len(phrog_integer)} items")
    
    return embeddings, categories, validation_categories, phrog_integer

def create_predictor(model_directory, device, input_dim, num_classes, num_heads, hidden_dim, lstm_hidden_dim, dropout, use_lstm, max_len, protein_dropout_rate=0.0, 
                    attention='circular', positional_encoding_type='fourier', pre_norm=False, progressive_dropout=False, initial_dropout_rate=1.0, final_dropout_rate=0.4, output_dim=None, num_layers=2):
    """
    Create and initialize a predictor with models from the specified directory
    """
    p = predictor.Predictor(device=device)
    p.read_models_from_directory(
        model_directory, 
        input_dim=input_dim, 
        num_classes=num_classes, 
        num_heads=num_heads, 
        hidden_dim=hidden_dim, 
        lstm_hidden_dim=lstm_hidden_dim, 
        dropout=dropout, 
        use_lstm=use_lstm, 
        max_len=max_len, 
        protein_dropout_rate=protein_dropout_rate, 
        attention=attention, 
        positional_encoding_type=positional_encoding_type, 
        pre_norm=pre_norm, 
        progressive_dropout=progressive_dropout, 
        initial_dropout_rate=initial_dropout_rate, 
        final_dropout_rate=final_dropout_rate, 
        output_dim=output_dim,
        num_layers=num_layers
    )
    return p

def create_dataloader(embeddings, categories, validation_categories, batch_size=64):
    """
    Create a DataLoader for the validation data
    """
    dataset = model_onehot.EmbeddingDataset(
        list(embeddings.values()),
        list(categories.values()),
        list(embeddings.keys()),
        mask_portion=0  # No masking for calibration
    )
    
    # Set validation mode
    dataset.set_validation(list(validation_categories.values()), validation=True)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=model_onehot.collate_fn
    )
    
    return dataloader

def process_batches(p, conf_dataset_loader, device):
    """
    Process batches and collect predictions and true labels
    """
    logger.info("Processing batches to collect calibration data...")
    
    all_probs = []
    all_labels = []
    
    # Put models in evaluation mode
    for model in p.models:
        model.eval()
    
    with torch.no_grad():
        for embeddings_batch, categories_batch, masks_batch, idx_batch in conf_dataset_loader:
            # Move data to device
            embeddings_batch = embeddings_batch.to(device)
            categories_batch = categories_batch.to(device)
            masks_batch = masks_batch.to(device)
            
            # Get padding mask
            src_key_padding_mask = (masks_batch != -2).to(device)
            
            # Get predictions
            batch_scores = p.predict_batch(embeddings_batch, src_key_padding_mask)
            
            # Process results for each item in batch
            for i, indices in enumerate(idx_batch):
                if len(indices) > 0:  # If there are masked tokens
                    for idx in indices:
                        # Get the probability and true label for this token
                        prob = batch_scores[i][idx].cpu().numpy()
                        label = categories_batch[i][idx].item()
                        if label != -1:  # Ignore padding tokens
                            all_probs.append(prob)
                            all_labels.append(label)
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    logger.info(f"Collected {len(all_labels)} samples for calibration")
    return all_probs, all_labels

def calibrate_probabilities(all_probs, all_labels, phrog_integer, num_classes):
    """
    Calibrate probabilities using isotonic regression for each class
    """
    logger.info("Calibrating probabilities with isotonic regression...")
    
    calibration_models = {}
    calibration_stats = {}
    
    # Create the reverse mapping from integer to category name
    categories_map = dict(zip(range(num_classes), phrog_integer.keys()))
    
    # Create isotonic regression model for each class
    for class_idx in range(num_classes):
        class_name = categories_map.get(class_idx, f"Class_{class_idx}")
        logger.info(f"Calibrating class {class_idx}: {class_name}")
        
        # Get binary indicator for this class
        y_true_binary = (all_labels == class_idx).astype(int)
        
        # Get predicted probabilities for this class
        y_pred_proba = all_probs[:, class_idx]
        
        # Make sure we have samples of this class
        if sum(y_true_binary) > 0:
            # Fit isotonic regression
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(y_pred_proba, y_true_binary)
            
            # Compute calibration metrics
            y_calibrated = ir.transform(y_pred_proba)
            brier = brier_score_loss(y_true_binary, y_pred_proba)
            brier_cal = brier_score_loss(y_true_binary, y_calibrated)
            logloss = log_loss(y_true_binary, y_pred_proba, eps=1e-15)
            logloss_cal = log_loss(y_true_binary, y_calibrated, eps=1e-15)
            
            # Store calibration model and statistics
            calibration_models[class_name] = {
                'calibrator': ir,
                'num_samples': len(y_true_binary),
                'positive_samples': sum(y_true_binary)
            }
            
            calibration_stats[class_name] = {
                'brier_score_raw': brier,
                'brier_score_calibrated': brier_cal,
                'log_loss_raw': logloss,
                'log_loss_calibrated': logloss_cal,
                'brier_improvement': (brier - brier_cal) / brier if brier > 0 else 0,
                'logloss_improvement': (logloss - logloss_cal) / logloss if logloss > 0 else 0
            }
            
            logger.info(f"  - Samples: {len(y_true_binary)}, Positive: {sum(y_true_binary)}")
            logger.info(f"  - Brier score: {brier:.4f} -> {brier_cal:.4f} ({calibration_stats[class_name]['brier_improvement']:.2%} improvement)")
            logger.info(f"  - Log loss: {logloss:.4f} -> {logloss_cal:.4f} ({calibration_stats[class_name]['logloss_improvement']:.2%} improvement)")
        else:
            logger.warning(f"No positive samples for class {class_idx}: {class_name}. Skipping calibration.")
    
    return calibration_models, calibration_stats

def save_calibration_models(calibration_models, calibration_stats, output_path, phrog_integer):
    """
    Save calibration models and statistics to disk
    """
    logger.info(f"Saving calibration models to {output_path}")
    
    # Save the models
    with open(os.path.join(output_path, 'calibration_models.pkl'), 'wb') as f:
        pickle.dump(calibration_models, f)
    
    # Save calibration statistics
    with open(os.path.join(output_path, 'calibration_stats.pkl'), 'wb') as f:
        pickle.dump(calibration_stats, f)
    
    # Also save a mapping from integer to category name for reference
    with open(os.path.join(output_path, 'category_mapping.pkl'), 'wb') as f:
        pickle.dump(phrog_integer, f)
    
    logger.info("Calibration models and statistics saved successfully")

if __name__ == "__main__":
    main()
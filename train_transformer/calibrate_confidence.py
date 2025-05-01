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
    all_probs, all_labels, all_categories = process_batches(p, conf_dataset_loader, device)
    
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
    logger.info(f"Loaded integer categories: {len(phrog_integer)} categories")
    
    return embeddings, categories, validation_categories, phrog_integer

def create_predictor(model_directory, device, input_dim, num_classes, num_heads, hidden_dim, lstm_hidden_dim, dropout, use_lstm, max_len, protein_dropout_rate=0.0, 
                    attention='circular', positional_encoding_type='fourier', pre_norm=False, progressive_dropout=False, initial_dropout_rate=1.0, final_dropout_rate=0.4, output_dim=None, num_layers=2):
    """
    Create a predictor model
    """
    logger.info(f"Creating predictor with model directory: {model_directory}")
    p = predictor.Predictor(device=device)
    p.read_models_from_directory(
        directory_path=model_directory,
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
    
    # Log the number of loaded models
    logger.info(f"Loaded {len(p.models)} models from {model_directory}")
    return p

def create_dataloader(embeddings, categories, validation_categories, batch_size=64):
    """
    Create a dataloader for the dataset
    """
    logger.info(f"Creating dataloader with batch size: {batch_size}")
    
    # Get keys that exist in all dictionaries
    common_keys = [k for k in embeddings.keys() if k in categories and k in validation_categories]
    logger.info(f"Found {len(common_keys)} common keys in datasets")
    
    # Filter dictionaries to only include common keys
    embeddings_filtered = {k: embeddings[k] for k in common_keys}
    categories_filtered = {k: categories[k] for k in common_keys}
    validation_filtered = {k: validation_categories[k] for k in common_keys}
    
    # Create dataset
    dataset = model_onehot.EmbeddingDataset(
        list(embeddings_filtered.values()),
        list(categories_filtered.values()),
        list(common_keys),
        mask_portion=0.0  # No masking for calibration
    )
    
    # Set validation mode
    dataset.set_validation(list(validation_filtered.values()), True)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=model_onehot.collate_fn
    )
    
    return dataloader

def process_batches(p, conf_dataset_loader, device):
    """
    Process data in batches and make predictions
    """
    logger.info("Processing batches...")
    all_probs = []
    all_labels = []
    all_categories = []
    
    # Process batches
    for i, batch in enumerate(conf_dataset_loader):
        embeddings_batch, categories_batch, masks_batch, idx_batch = batch
        
        # Move data to device
        embeddings_batch = embeddings_batch.to(device)
        categories_batch = categories_batch.to(device)
        masks_batch = masks_batch.to(device)
        
        # Create padding mask
        src_key_padding_mask = (masks_batch != -2).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = p.models[0](embeddings_batch, src_key_padding_mask=src_key_padding_mask)
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
            
            # For ensemble models, average predictions
            for model in p.models[1:]:
                model_output = model(embeddings_batch, src_key_padding_mask=src_key_padding_mask)
                model_output = torch.nn.functional.softmax(model_output, dim=-1)
                outputs += model_output
            
            outputs /= len(p.models)
        
        # Collect predictions and labels
        for batch_idx in range(len(idx_batch)):
            if len(idx_batch[batch_idx]) > 0:
                # Get indices of masked tokens
                indices = idx_batch[batch_idx]
                
                # Get predictions and labels for masked tokens
                probs = outputs[batch_idx, indices].cpu().numpy()
                labels = categories_batch[batch_idx, indices].cpu().numpy()
                
                # Add to lists
                all_probs.append(probs)
                all_labels.append(labels)
                all_categories.append(categories_batch[batch_idx].cpu().numpy())
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(conf_dataset_loader)} batches")
    
    # Concatenate results
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    logger.info(f"Processed {len(all_probs)} samples with {all_probs.shape[1]} classes")
    return all_probs, all_labels, all_categories

def calibrate_probabilities(all_probs, all_labels, phrog_integer, num_classes):
    """
    Calibrate probabilities using isotonic regression for each class
    """
    logger.info("Calibrating probabilities with isotonic regression...")
    
    # Initialize isotonic regression models and statistics
    calibration_models = {}
    calibration_stats = {}
    
    # Map integer labels to class names
    integer_to_category = {v: k for k, v in phrog_integer.items()}
    
    # For each class
    for class_idx in range(num_classes):
        logger.info(f"Calibrating class {class_idx}")
        
        # Get binary labels (1 for current class, 0 for others)
        binary_labels = (all_labels == class_idx).astype(int)
        
        # Get uncalibrated probabilities for current class
        uncalibrated_probs = all_probs[:, class_idx]
        
        # Skip classes with no positive examples
        if np.sum(binary_labels) == 0:
            logger.warning(f"Class {class_idx} has no positive examples, skipping calibration")
            continue
        
        # Skip classes with all positive examples
        if np.sum(binary_labels) == len(binary_labels):
            logger.warning(f"Class {class_idx} has all positive examples, skipping calibration")
            continue
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        try:
            iso_reg.fit(uncalibrated_probs, binary_labels)
            
            # Calculate calibrated probabilities
            calibrated_probs = iso_reg.predict(uncalibrated_probs)
            
            # Calculate metrics before and after calibration
            uncalibrated_brier = brier_score_loss(binary_labels, uncalibrated_probs)
            calibrated_brier = brier_score_loss(binary_labels, calibrated_probs)
            
            try:
                uncalibrated_log_loss = log_loss(binary_labels, uncalibrated_probs)
                calibrated_log_loss = log_loss(binary_labels, calibrated_probs)
            except (ValueError, np.core._exceptions._UFuncOutputCastingError):
                logger.warning(f"Could not calculate log loss for class {class_idx}, using NaN")
                uncalibrated_log_loss = float('nan')
                calibrated_log_loss = float('nan')
            
            # Get class name
            class_name = integer_to_category.get(class_idx, f"Class_{class_idx}")
            
            # Store model and statistics
            calibration_models[class_name] = iso_reg
            calibration_stats[class_name] = {
                'uncalibrated_brier': uncalibrated_brier,
                'calibrated_brier': calibrated_brier,
                'uncalibrated_log_loss': uncalibrated_log_loss,
                'calibrated_log_loss': calibrated_log_loss,
                'positive_examples': np.sum(binary_labels),
                'total_examples': len(binary_labels),
                'min_uncalibrated': np.min(uncalibrated_probs),
                'max_uncalibrated': np.max(uncalibrated_probs),
                'min_calibrated': np.min(calibrated_probs),
                'max_calibrated': np.max(calibrated_probs),
            }
            
            logger.info(f"Class {class_idx} ({class_name}): Brier score improved from {uncalibrated_brier:.4f} to {calibrated_brier:.4f}")
            logger.info(f"Class {class_idx} ({class_name}): Calibrated probs range: {np.min(calibrated_probs):.4f} to {np.max(calibrated_probs):.4f}")
            
        except Exception as e:
            logger.error(f"Error calibrating class {class_idx}: {e}")
    
    logger.info(f"Calibrated {len(calibration_models)} classes")
    return calibration_models, calibration_stats

def save_calibration_models(calibration_models, calibration_stats, output_path, phrog_integer):
    """
    Save calibration models and statistics to the output path
    """
    logger.info(f"Saving calibration models to {output_path}")
    
    # Save calibration models
    models_path = os.path.join(output_path, "calibration_models.pkl")
    with open(models_path, 'wb') as f:
        pickle.dump(calibration_models, f)
    logger.info(f"Saved calibration models to {models_path}")
    
    # Save calibration statistics
    stats_path = os.path.join(output_path, "calibration_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(calibration_stats, f)
    logger.info(f"Saved calibration statistics to {stats_path}")
    
    # Save integer to category mapping
    mapping_path = os.path.join(output_path, "integer_to_category.pkl")
    with open(mapping_path, 'wb') as f:
        pickle.dump(phrog_integer, f)
    logger.info(f"Saved integer to category mapping to {mapping_path}")

if __name__ == "__main__":
    main()
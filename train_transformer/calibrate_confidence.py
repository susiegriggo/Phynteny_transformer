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

    # adjust phrog integer to exlucde unknown
    phrog_integer = dict(zip(range(9), [phrog_integer.get(i) for i in range(1,10)]))
    
    # Create predictor
    p = create_predictor(
        model_directory, device, input_dim, num_classes, num_heads, 
        hidden_dim, lstm_hidden_dim, dropout, not no_lstm, max_len, 
        protein_dropout_rate, attention, positional_encoding_type, pre_norm,
        progressive_dropout, initial_dropout_rate, final_dropout_rate, output_dim, num_layers
    )
    
    # Create dataset for calibration
    conf_dataset_loader = create_dataloader(embeddings, categories, validation_categories, batch_size)
    
    # Process data in batches to get predictions from each model separately
    all_model_probs, all_labels = process_batches_per_model(p, conf_dataset_loader, device)
    
    # Calibrate predictions with isotonic regression, for each model separately
    calibration_models, calibration_stats, raw_scores = calibrate_probabilities_per_model(
        all_model_probs, all_labels, phrog_integer, num_classes
    )
    
    # Save calibration models and statistics
    save_calibration_models(calibration_models, calibration_stats, output_path, phrog_integer)
    
    # Save raw scores and detailed data for analysis
    save_detailed_prediction_data(raw_scores, output_path)
    
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
    logger.info("Creating the dataset and dataloader.")
    
    # only include data that occurs in both sets 
    logger.info("Removing categories not in validation set.")
    include_keys = list(set(categories.keys()).intersection(set(validation_categories.keys())))
    validation_categories = dict(zip(include_keys, [validation_categories.get(k) for k in include_keys]))
    categories = dict(zip(include_keys, [categories.get(k) for k in include_keys]))
    embeddings = dict(zip(include_keys, [embeddings.get(k) for k in include_keys]))
    
    # create the dataset
    conf_dataset = model_onehot.EmbeddingDataset(
        list(embeddings.values()), list(categories.values()), list(categories.keys()), mask_portion=0)
    
    # set the validation set
    logger.info("Setting the validation set.")
    conf_dataset.set_validation(list(validation_categories.values()))
    conf_dataset_loader = DataLoader(conf_dataset, batch_size=batch_size, collate_fn=model_onehot.collate_fn)
    
    return conf_dataset_loader

def process_batches_per_model(p, conf_dataset_loader, device):
    """
    Process batches and collect predictions from each model separately
    """
    logger.info("Processing batches to collect calibration data per model...")
    
    num_models = len(p.models)
    logger.info(f"Found {num_models} models to process")
    
    # Create separate lists for each model's predictions
    all_models_probs = [[] for _ in range(num_models)]
    all_labels = []
    
    # Put models in evaluation mode
    for model in p.models:
        model.eval()
    
    with torch.no_grad():
        for batch_idx, (embeddings, categories, masks, _) in enumerate(conf_dataset_loader):
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx}/{len(conf_dataset_loader)}")
            
            # Move data to device
            embeddings = embeddings.to(device)
            categories = categories.to(device)
            masks = masks.to(device)
            src_key_padding_mask = (masks != -2).to(device)
            
            # Process each model separately
            for i, model in enumerate(p.models):
                outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                
                # Make sure we gather all valid predictions
                valid_mask = masks.cpu().numpy() == 1
                for j in range(probs.shape[0]):  # For each sample in the batch
                    for k in range(probs.shape[1]):  # For each position in the sequence
                        if valid_mask[j, k]:
                            all_models_probs[i].append(probs[j, k])
            
            # Only collect labels once from the first model iteration
            valid_mask = masks.cpu().numpy() == 1
            for j in range(categories.shape[0]):  # For each sample in the batch
                for k in range(categories.shape[1]):  # For each position in the sequence
                    if valid_mask[j, k]:
                        all_labels.append(categories[j, k].cpu().numpy())
    
    # Convert to numpy arrays
    for i in range(num_models):
        all_models_probs[i] = np.array(all_models_probs[i])
    all_labels = np.array(all_labels)
    
    logger.info(f"Collected {len(all_labels)} samples for calibration across {num_models} models")
    return all_models_probs, all_labels

def calibrate_probabilities_per_model(all_models_probs, all_labels, phrog_integer, num_classes):
    """
    Calibrate probabilities using isotonic regression for each class and each model separately
    """
    logger.info("Calibrating probabilities with isotonic regression for each model...")
    
    num_models = len(all_models_probs)
    logger.info(f"Processing calibration for {num_models} models")
    
    all_calibration_models = []
    all_calibration_stats = []
    
    # Store original probabilities and true labels
    raw_scores = {
        "model_probabilities": all_models_probs,
        "true_labels": all_labels,
        "calibrated_model_probabilities": [],
        "model_predictions": [],
        "calibrated_model_predictions": [],
        "model_confidences": []
    }
    
    # For averaging results across all models
    avg_probs = np.zeros_like(all_models_probs[0])
    avg_calibrated_probs = np.zeros_like(all_models_probs[0])
    
    # Process each model separately
    for model_idx in range(num_models):
        logger.info(f"Calibrating model {model_idx+1}/{num_models}")
        
        model_probs = all_models_probs[model_idx]
        avg_probs += model_probs / num_models  # For ensemble average
        
        calibration_models = {}
        calibration_stats = {}
        
        # Create arrays to store calibrated probabilities for this model
        calibrated_probs = np.zeros_like(model_probs)
        
        # Create isotonic regression model for each class
        for class_idx in phrog_integer.keys():
            class_name = phrog_integer[class_idx]
            logger.info(f"Calibrating class {class_idx} ({class_name})")
            
            # Extract probabilities for this class
            class_probs = model_probs[:, class_idx]
            
            # One-vs-rest approach: binary labels for this class
            binary_labels = (all_labels == class_idx).astype(int)
            
            # Fit isotonic regression for this class
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(class_probs, binary_labels)
            
            # Predict calibrated probabilities
            calibrated_class_probs = iso_reg.predict(class_probs)
            calibrated_probs[:, class_idx] = calibrated_class_probs
            
            # Calculate calibration metrics
            brier = brier_score_loss(binary_labels, class_probs)
            calibrated_brier = brier_score_loss(binary_labels, calibrated_class_probs)
            
            # If any class_probs are 0 or 1, they cause warnings in log_loss, so clip them
            clipped_probs = np.clip(class_probs, 1e-15, 1 - 1e-15)
            clipped_calibrated = np.clip(calibrated_class_probs, 1e-15, 1 - 1e-15)
            
            logloss = log_loss(binary_labels, clipped_probs)
            calibrated_logloss = log_loss(binary_labels, clipped_calibrated)
            
            # Store the calibration model and stats for this class
            calibration_models[class_name] = iso_reg
            calibration_stats[class_name] = {
                "brier_score_raw": brier,
                "brier_score_calibrated": calibrated_brier,
                "log_loss_raw": logloss,
                "log_loss_calibrated": calibrated_logloss,
                "brier_improvement": brier - calibrated_brier,
                "logloss_improvement": logloss - calibrated_logloss
            }
            
        # Normalize calibrated probabilities to sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        calibrated_probs = np.divide(calibrated_probs, row_sums, out=np.zeros_like(calibrated_probs), where=row_sums!=0)
        
        # Add to ensemble average
        avg_calibrated_probs += calibrated_probs / num_models
        
        # Store the calibration results for this model
        all_calibration_models.append(calibration_models)
        all_calibration_stats.append(calibration_stats)
        
        # Store predictions and calibrated probabilities for this model
        raw_scores["calibrated_model_probabilities"].append(calibrated_probs)
        raw_scores["model_predictions"].append(np.argmax(model_probs, axis=1))
        raw_scores["calibrated_model_predictions"].append(np.argmax(calibrated_probs, axis=1))
        raw_scores["model_confidences"].append(np.max(calibrated_probs, axis=1))
    
    # Store the ensemble average results
    raw_scores["ensemble_probabilities"] = avg_probs
    raw_scores["ensemble_calibrated_probabilities"] = avg_calibrated_probs
    raw_scores["ensemble_predictions"] = np.argmax(avg_probs, axis=1)
    raw_scores["ensemble_calibrated_predictions"] = np.argmax(avg_calibrated_probs, axis=1)
    raw_scores["ensemble_confidence"] = np.max(avg_calibrated_probs, axis=1)
    
    logger.info("Calibration completed for all models")
    return all_calibration_models, all_calibration_stats, raw_scores

def save_calibration_models(calibration_models, calibration_stats, output_path, phrog_integer):
    """
    Save calibration models and statistics to files
    """
    logger.info("Saving calibration models and statistics")
    
    # Create output directories if needed
    os.makedirs(output_path, exist_ok=True)
    models_dir = os.path.join(output_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save individual calibration models for each model
    num_models = len(calibration_models)
    
    for model_idx, model_calibrations in enumerate(calibration_models):
        model_dir = os.path.join(models_dir, f'model_{model_idx}')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save each class calibration separately
        for class_name, model in model_calibrations.items():
            class_path = os.path.join(model_dir, f'calibration_{class_name}.pkl')
            with open(class_path, 'wb') as f:
                dump(model, f)
    
    # Save all calibration models as a single file - use the first model as default
    # This maintains compatibility with existing code
    all_models_path = os.path.join(output_path, 'calibration_models.pkl')
    with open(all_models_path, 'wb') as f:
        dump(calibration_models[0], f)  # Use first model as default
    logger.info(f"Saved default calibration models to {all_models_path}")
    
    # Also save the full collection of all model calibrations
    all_models_collection_path = os.path.join(output_path, 'all_calibration_models.pkl')
    with open(all_models_collection_path, 'wb') as f:
        dump(calibration_models, f)
    logger.info(f"Saved full collection of calibration models to {all_models_collection_path}")
    
    # Save calibration statistics
    stats_path = os.path.join(output_path, 'calibration_statistics.pkl')
    with open(stats_path, 'wb') as f:
        dump(calibration_stats, f)
    logger.info(f"Saved calibration statistics to {stats_path}")
    
    # Save calibration stats as CSV for easy reading - one file per model
    for model_idx, model_stats in enumerate(calibration_stats):
        stats_csv = []
        stats_csv.append("class_name,brier_score_raw,brier_score_calibrated,log_loss_raw,log_loss_calibrated,brier_improvement,logloss_improvement")
        
        for class_name, stats in model_stats.items():
            stats_csv.append(f"{class_name},{stats['brier_score_raw']},{stats['brier_score_calibrated']},{stats['log_loss_raw']},{stats['log_loss_calibrated']},{stats['brier_improvement']},{stats['logloss_improvement']}")
        
        # Write CSV manually since we're not importing pandas
        csv_path = os.path.join(output_path, f'calibration_statistics_model_{model_idx}.csv')
        with open(csv_path, 'w') as f:
            f.write('\n'.join(stats_csv))
        logger.info(f"Saved calibration statistics as CSV for model {model_idx} to {csv_path}")
    
    # Also save a mapping from integer to category name for reference
    with open(os.path.join(output_path, 'category_mapping.pkl'), 'wb') as f:
        dump(phrog_integer, f)
    
    logger.info("Calibration models and statistics saved successfully")

def save_detailed_prediction_data(raw_scores, output_path):
    """
    Save detailed prediction data in a format compatible with compute_confidence.py
    """
    detailed_dict = {
        'scores': raw_scores["ensemble_probabilities"],
        'true_labels': raw_scores["true_labels"],
        'predictions': raw_scores["ensemble_predictions"],
        'calibrated_scores': raw_scores["ensemble_calibrated_probabilities"],
        'calibrated_predictions': raw_scores["ensemble_calibrated_predictions"],
        'confidence': raw_scores["ensemble_confidence"],
        'per_model_scores': raw_scores["model_probabilities"],
        'per_model_calibrated_scores': raw_scores["calibrated_model_probabilities"],
        'per_model_predictions': raw_scores["model_predictions"],
        'per_model_calibrated_predictions': raw_scores["calibrated_model_predictions"],
        'per_model_confidences': raw_scores["model_confidences"]
    }
    
    detailed_output_path = os.path.join(output_path, "calibration_detailed.pkl")
    logger.info(f"Saving detailed prediction data to {detailed_output_path}")
    
    with open(detailed_output_path, 'wb') as f:
        dump(detailed_dict, f)
    
    # Also save in a format exactly matching compute_confidence output
    compute_confidence_format = os.path.join(output_path, "confidence_detailed.pkl")
    logger.info(f"Saving in compute_confidence compatible format to {compute_confidence_format}")
    
    with open(compute_confidence_format, 'wb') as f:
        dump(detailed_dict, f)
    
    logger.info("Detailed prediction data saved successfully")

if __name__ == "__main__":
    main()

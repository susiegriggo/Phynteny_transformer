import click
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src import predictor, model_onehot
from loguru import logger
import os
from joblib import load
import mmap

# Parameters
PARAMETERS = {
    'model_directory': 'Directory containing the models.',
    'embeddings_path': 'Path to the embeddings file.',
    'categories_path': 'Path to the categories file.',
    'validation_categories_path': 'Path to the validation categories file.',
    'integer_category_path': 'Path to the integer category file.',
    'output_path': 'Path to save the output confidence dictionary.',
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
    Main function to compute confidence scores.

    :param model_directory: Directory containing the models
    :param embeddings_path: Path to the embeddings file
    :param categories_path: Path to the categories file
    :param validation_categories_path: Path to the validation categories file
    :param integer_category_path: Path to the integer category file
    :param output_path: Path to save the output confidence dictionary
    :param force: Boolean indicating whether to overwrite the output directory if it exists
    :param batch_size: Batch size for processing the data
    :param input_dim: Input dimension for the model
    :param num_classes: Number of classes for the model
    :param num_heads: Number of attention heads for the model
    :param hidden_dim: Hidden dimension for the model
    :param lstm_hidden_dim: LSTM hidden dimension for the model
    :param no_lstm: Specify if LSTM should not be used in the model
    :param max_len: Maximum length for the model
    :param attention: Type of attention mechanism (absolute, relative, or circular)
    :param positional_encoding_type: Type of positional encoding (fourier or sinusoidal)
    :param pre_norm: Use pre-normalization in transformer layers
    :param output_dim: Output dimension for the model
    :param num_layers: Number of transformer layers in the model
    :param dropout: Dropout rate for the model
    :param protein_dropout_rate: Dropout rate for protein features
    :param progressive_dropout: Enable progressive dropout for protein features
    :param initial_dropout_rate: Initial dropout rate when using progressive dropout
    :param final_dropout_rate: Final dropout rate when using progressive dropout
    """
    use_lstm = not no_lstm
    # Use num_classes as output_dim if not specified
    if output_dim is None:
        output_dim = num_classes

    if os.path.exists(output_path) and not force:
        logger.error(f"Output directory already exists: {output_path}. Use --force to overwrite.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Starting the computation of confidence scores.")
    if device == 'cuda':
        logger.info("Using GPU for computation.")
    else:
        logger.info("Using CPU for computation.")
    
    embeddings, categories, validation_categories, phrog_integer = load_data(embeddings_path, categories_path, validation_categories_path, integer_category_path)
    
    p = create_predictor(model_directory, device, input_dim, num_classes, num_heads, hidden_dim, lstm_hidden_dim, dropout, use_lstm, max_len, protein_dropout_rate, 
                         attention, positional_encoding_type, pre_norm, progressive_dropout, initial_dropout_rate, final_dropout_rate, output_dim, num_layers)
    logger.info(f'Predictor models: {p.models}')
    logger.info("Creating the dataset and dataloader.")
    conf_dataset_loader = create_dataloader(embeddings, categories, validation_categories, batch_size)
    logger.info("Processing batches.")
    all_probs, all_categories, all_labels, all_protein_ids = process_batches(p, conf_dataset_loader, device)
    c_dict, detailed_dict = compute_confidence_dict(all_categories, all_labels, all_probs, phrog_integer)
    
    # Add protein IDs to the detailed dictionary
    detailed_dict['protein_ids'] = all_protein_ids
    
    save_confidence_dict(c_dict, detailed_dict, output_path)
    logger.info("Finished the computation of confidence scores.")


def load_data(embeddings_path, categories_path, validation_categories_path, integer_category_path):
    """
    Load embeddings and categories.

    :param embeddings_path: Path to the embeddings file
    :param categories_path: Path to the categories file
    :param validation_categories_path: Path to the validation categories file
    :param integer_category_path: Path to the integer category file
    :return: Tuple of embeddings, categories, validation categories, and phrog integer dictionary
    """
    logger.info("Loading embeddings and categories.")
    
    # Use memory mapping to load embeddings
    with open(embeddings_path, 'rb') as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        embeddings = pickle.load(mmapped_file)
    
    categories = load(categories_path)
    validation_categories = load(validation_categories_path)
    phrog_integer = load(integer_category_path)

    phrog_integer = dict(
        zip(
            [i - 1 for i in list(phrog_integer.keys())],
            [i for i in list(phrog_integer.values())],
        )
    )
    return embeddings, categories, validation_categories, phrog_integer


def create_predictor(model_directory, device, input_dim, num_classes, num_heads, hidden_dim, lstm_hidden_dim, dropout, use_lstm, max_len, protein_dropout_rate=0.0, 
                    attention='circular', positional_encoding_type='fourier', pre_norm=False, progressive_dropout=False, initial_dropout_rate=1.0, final_dropout_rate=0.4, output_dim=None, num_layers=2):
    """
    Create the predictor object.

    :param model_directory: Directory containing the models
    :param device: Device to use for computation (cpu or cuda)
    :param input_dim: Input dimension for the model
    :param num_classes: Number of classes for the model
    :param num_heads: Number of attention heads for the model
    :param hidden_dim: Hidden dimension for the model
    :param lstm_hidden_dim: LSTM hidden dimension for the model
    :param dropout: Dropout rate for the model
    :param use_lstm: Whether to use LSTM in the model
    :param max_len: Maximum length for the model
    :param protein_dropout_rate: Dropout rate for protein features
    :param attention: Type of attention mechanism (absolute, relative, or circular)
    :param positional_encoding_type: Type of positional encoding (fourier or sinusoidal)
    :param pre_norm: Use pre-normalization in transformer layers
    :param progressive_dropout: Enable progressive dropout for protein features
    :param initial_dropout_rate: Initial dropout rate when using progressive dropout
    :param final_dropout_rate: Final dropout rate when using progressive dropout
    :param output_dim: Output dimension for the model
    :param num_layers: Number of transformer layers
    :return: Predictor object
    """
    logger.info("Creating the predictor object.")
    p = predictor.Predictor(device=device)
    
    # Set output_dim to num_classes if not specified
    if output_dim is None:
        output_dim = num_classes
    
    # Log the model parameters being used
    logger.info(f"Model parameters: input_dim={input_dim}, num_classes={num_classes}, num_heads={num_heads}, "
                f"hidden_dim={hidden_dim}, lstm_hidden_dim={lstm_hidden_dim}, dropout={dropout}, "
                f"use_lstm={use_lstm}, max_len={max_len}, protein_dropout_rate={protein_dropout_rate}, "
                f"attention={attention}, positional_encoding_type={positional_encoding_type}, pre_norm={pre_norm}, "
                f"output_dim={output_dim}, num_layers={num_layers}, "
                f"progressive_dropout={progressive_dropout}, initial_dropout_rate={initial_dropout_rate}, final_dropout_rate={final_dropout_rate}")
    
    # Select positional encoding function
    positional_encoding_func = model_onehot.fourier_positional_encoding if positional_encoding_type == 'fourier' else model_onehot.sinusoidal_positional_encoding
    
    # Read models with updated parameters - using consistent approach with generate_roc_data.py
    models = []
    
    for k in range(10):  # Assuming 10 folds like in generate_roc_data.py
        try:
            # Create model instance directly 
            m = model_onehot.TransformerClassifierCircularRelativeAttention(
                input_dim=input_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim,
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
            
            # Initialize buffers with dummy forward pass if needed
            if hasattr(m, 'protein_feature_dropout'):
                logger.info(f"Initializing protein_feature_dropout buffers for model {k+1} with dummy forward pass")
                batch_size, seq_len = 2, 10  # Minimal batch for initialization
                feature_dim = m.gene_feature_dim + (hidden_dim - m.gene_feature_dim)
                dummy_input = torch.zeros((batch_size, seq_len, feature_dim), device=device)
                dummy_idx = [torch.tensor([0, 1]) for _ in range(batch_size)]
                # Run a forward pass with the dummy data to initialize any dynamic buffers
                m.protein_feature_dropout(dummy_input, dummy_idx)
            
            # Try both file path patterns for models
            try:
                # First try the pattern used in generate_roc_data.py
                model_path = f'{model_directory}/fold_{k+1}transformer.model'
                state_dict = torch.load(model_path, map_location=device)
                logger.info(f"Loading model from {model_path}")
            except FileNotFoundError:
                # Fall back to the pattern with slash
                model_path = f'{model_directory}/fold_{k+1}/transformer_state_dict.pth'
                state_dict = torch.load(model_path, map_location=device)
                logger.info(f"Loading model from {model_path}")
                
            # Load state dict with strict=False to handle parameter differences
            m.load_state_dict(state_dict, strict=False)
            m.to(device)
            m.eval()
            models.append(m)
            logger.info(f"Successfully loaded model {k+1}")
        except Exception as e:
            logger.warning(f"Could not load model for fold {k+1}: {str(e)}")
    
    p.models = models
    logger.info(f"Loaded models: {len(p.models)} models loaded.")
    return p


def create_dataloader(embeddings, categories, validation_categories, batch_size=64):
    """
    Create the dataset and dataloader.

    :param embeddings: Dictionary of embeddings
    :param categories: Dictionary of categories
    :param validation_categories: Dictionary of validation categories
    :param batch_size: Batch size for processing the data
    :return: DataLoader object
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
    conf_dataset_loader = DataLoader(conf_dataset, batch_size=batch_size, collate_fn=model_onehot.collate_fn)  # Added num_workers for parallel data loading
    return conf_dataset_loader


def process_batches(p, conf_dataset_loader, device):
    """
    Process batches to compute probabilities and categories.

    :param p: Predictor object
    :param conf_dataset_loader: DataLoader object for the dataset
    :param device: Device to use for computation (cpu or cuda)
    :return: Tuple of all probabilities, all categories, all labels and protein IDs
    """
    logger.info("Processing batches.")
    all_probs = []
    all_categories = []
    all_protein_ids = []  # Track protein IDs
    batches = 0

    total_batches = len(conf_dataset_loader)
    logger.info(f"Total number of batches to process: {total_batches}")

    # Initialize dictionaries to count the number of each category that gets masked and predicted
    masked_category_counts = {i: 0 for i in range(p.models[0].num_classes)}
    predicted_category_counts = {i: 0 for i in range(p.models[0].num_classes)}

    for embeddings, categories, masks, idx in conf_dataset_loader:
        embeddings = embeddings.to(device)
        masks = masks.to(device)
        src_key_padding_mask = (masks != -2).to(device)

        phynteny_scores = p.predict_batch(embeddings, src_key_padding_mask)

        batch_size = len(idx)
        

        for m in range(batch_size):
            try:
                scores_at_idx = torch.tensor(phynteny_scores[m][idx[m]]).to(device)
                if len(scores_at_idx.shape) == 1:
                    scores_at_idx = scores_at_idx.reshape(1, -1)
                all_probs.append(scores_at_idx.cpu().numpy())
                all_categories.append(categories[m][idx[m]].cpu())
                all_protein_ids.append(conf_dataset_loader.dataset.ids[m])  # Save protein IDs

                # Update masked category counts
                for cat in categories[m][idx[m]].tolist():
                    masked_category_counts[cat] += 1

                # Update predicted category counts
                predicted_category = np.argmax(scores_at_idx.cpu().numpy(), axis=1)
                for cat in predicted_category:
                    predicted_category_counts[cat] += 1

            except KeyError as e:
                logger.error(f"KeyError: {e} - phynteny_scores[{m}] or idx[{m}] might be invalid.")
                logger.error(f"phynteny_scores[{m}]: {phynteny_scores[m]}")
                logger.error(f"idx[{m}]: {idx[m]}")
                raise e
        batches += 1
        if batches % 100 == 0:
            logger.info(f"Processed {batches} batches...")
            # Log the number of each category that gets masked and predicted
            logger.info(f"Masked category counts: {masked_category_counts}")
            logger.info(f"Predicted category counts: {predicted_category_counts}")

    all_categories = [t for t in all_categories if t.numel() > 0]
    all_categories = torch.cat(all_categories).tolist()
    all_probs = [arr for arr in all_probs if arr.size > 0]
    all_probs = [row for row in np.vstack(all_probs)]
    all_labels = [np.argmax(p) for p in all_probs]

    all_categories = np.array(all_categories)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Log the number of each category that gets masked and predicted
    logger.info(f"Masked category counts: {masked_category_counts}")
    logger.info(f"Predicted category counts: {predicted_category_counts}")

    return all_probs, all_categories, all_labels, all_protein_ids


def compute_confidence_dict(all_categories, all_labels, all_probs, phrog_integer):
    """
    Compute the confidence dictionary.

    :param all_categories: List of all categories
    :param all_labels: List of all labels
    :param all_probs: List of all probabilities
    :param phrog_integer: Dictionary mapping phrog annotations to integers
    :return: Tuple of confidence dictionary and detailed prediction dictionary
    """
    logger.info("Computing the confidence dictionary.")
    logger.info(f"all_categories shape: {all_categories.shape}")
    logger.info(f"all_labels shape: {all_labels.shape}")
    logger.info(f"all_probs shape: {all_probs.shape}")
    bandwidth = np.arange(0, 5, 0.01)[1:] # have updated to 0.01 to include a greater range of values 
    c_dict = predictor.build_confidence_dict(all_categories, all_labels, all_probs, bandwidth, phrog_integer)
    
    # Create a detailed prediction dictionary for plotting
    detailed_dict = {
        'scores': all_probs,
        'true_labels': all_categories,
        'predictions': all_labels,
    }
    
    # Calculate confidence for each prediction
    confidence_values = []
    for i, (prediction, prob) in enumerate(zip(all_labels, all_probs)):
        max_prob = np.max(prob)
        confidence = predictor.calculate_confidence(max_prob, prediction, c_dict)
        confidence_values.append(confidence)
    
    detailed_dict['confidence'] = np.array(confidence_values)
    
    return c_dict, detailed_dict


def save_confidence_dict(c_dict, detailed_dict, output_path):
    """
    Save the confidence dictionary and detailed prediction dictionary.

    :param c_dict: Confidence dictionary
    :param detailed_dict: Detailed prediction dictionary for plotting
    :param output_path: Path to save the confidence dictionary
    """
    logger.info("Saving the confidence dictionary.")
    with open(output_path, 'wb') as f:
        pickle.dump(c_dict, f)
    
    # Save detailed prediction dictionary for plotting
    detailed_output_path = output_path.replace('.pkl', '_detailed.pkl')
    if output_path == detailed_output_path:
        detailed_output_path = output_path + '_detailed.pkl'
    
    logger.info(f"Saving detailed prediction data to {detailed_output_path}")
    with open(detailed_output_path, 'wb') as f:
        pickle.dump(detailed_dict, f)


if __name__ == '__main__':
    main()

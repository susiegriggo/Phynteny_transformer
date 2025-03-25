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

def main(model_directory, embeddings_path, categories_path, validation_categories_path, integer_category_path, output_path, force, batch_size):
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
    """
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
    p = create_predictor(model_directory, device)
    logger.info(f'Predictor models: {p.models}')
    logger.info("Creating the dataset and dataloader.")
    conf_dataset_loader = create_dataloader(embeddings, categories, validation_categories, batch_size)
    logger.info("Processing batches.")
    all_probs, all_categories, all_labels = process_batches(p, conf_dataset_loader, device)
    c_dict = compute_confidence_dict(all_categories, all_labels, all_probs, phrog_integer)
    save_confidence_dict(c_dict, output_path)
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


def create_predictor(model_directory, device):
    """
    Create the predictor object.

    :param model_directory: Directory containing the models
    :param device: Device to use for computation (cpu or cuda)
    :return: Predictor object
    """
    logger.info("Creating the predictor object.")
    p = predictor.Predictor(device=device)
    p.read_models_from_directory(model_directory)
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
    conf_dataset_loader = DataLoader(conf_dataset, batch_size=batch_size, collate_fn=model_onehot.collate_fn, num_workers=4)  # Added num_workers for parallel data loading
    return conf_dataset_loader


def process_batches(p, conf_dataset_loader, device):
    """
    Process batches to compute probabilities and categories.

    :param p: Predictor object
    :param conf_dataset_loader: DataLoader object for the dataset
    :param device: Device to use for computation (cpu or cuda)
    :return: Tuple of all probabilities, all categories, and all labels
    """
    logger.info("Processing batches.")
    all_probs = []
    all_categories = []
    batches = 0

    total_batches = len(conf_dataset_loader)
    logger.info(f"Total number of batches to process: {total_batches}")

    for embeddings, categories, masks, idx in conf_dataset_loader:
        embeddings = embeddings.to(device)
        masks = masks.to(device)
        src_key_padding_mask = (masks != -2).to(device)
        
        logger.info(f"Embeddings device: {embeddings.device}, Masks device: {masks.device}, src_key_padding_mask device: {src_key_padding_mask.device}")

        phynteny_scores = p.predict_batch(embeddings, src_key_padding_mask)

        batch_size = len(idx)
        #logger.info(f"Processing batch {batches + 1}/{total_batches}, batch size: {batch_size}")

        for m in range(batch_size):
            try:
                scores_at_idx = torch.tensor(phynteny_scores[m][idx[m]]).to(device)
                if len(scores_at_idx.shape) == 1:
                    scores_at_idx = scores_at_idx.reshape(1, -1)
                all_probs.append(scores_at_idx.cpu().numpy())
                all_categories.append(categories[m][idx[m]].cpu())
            except KeyError as e:
                logger.error(f"KeyError: {e} - phynteny_scores[{m}] or idx[{m}] might be invalid.")
                logger.error(f"phynteny_scores[{m}]: {phynteny_scores[m]}")
                logger.error(f"idx[{m}]: {idx[m]}")
                raise e

        batches += 1
        if batches % 10 == 0:
            logger.info(f"Processed {batches} batches...")

    all_categories = [t for t in all_categories if t.numel() > 0]
    all_categories = torch.cat(all_categories).tolist()
    all_probs = [arr for arr in all_probs if arr.size > 0]
    all_probs = [row for row in np.vstack(all_probs)]
    all_labels = [np.argmax(p) for p in all_probs]

    all_categories = np.array(all_categories)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    return all_probs, all_categories, all_labels


def compute_confidence_dict(all_categories, all_labels, all_probs, phrog_integer):
    """
    Compute the confidence dictionary.

    :param all_categories: List of all categories
    :param all_labels: List of all labels
    :param all_probs: List of all probabilities
    :param phrog_integer: Dictionary mapping phrog annotations to integers
    :return: Confidence dictionary
    """
    logger.info("Computing the confidence dictionary.")
    logger.info(f"all_categories shape: {all_categories.shape}")
    logger.info(f"all_labels shape: {all_labels.shape}")
    logger.info(f"all_probs shape: {all_probs.shape}")
    bandwidth = np.arange(0, 5, 0.05)[1:]
    c_dict = predictor.build_confidence_dict(all_categories, all_labels, all_probs, bandwidth, phrog_integer)
    return c_dict


def save_confidence_dict(c_dict, output_path):
    """
    Save the confidence dictionary.

    :param c_dict: Confidence dictionary
    :param output_path: Path to save the confidence dictionary
    """
    logger.info("Saving the confidence dictionary.")
    with open(output_path, 'wb') as f:
        pickle.dump(c_dict, f)


if __name__ == '__main__':
    main()
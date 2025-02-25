import click
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src import predictor, model_onehot
from loguru import logger
import os


@click.command()
@click.option('--models', 'model_directory', type=click.Path(exists=True), help='Directory containing the models.')
@click.option('--X', 'embeddings_path', type=click.Path(exists=True), help='Path to the embeddings file.')
@click.option('--y', 'categories_path', type=click.Path(exists=True), help='Path to the categories file.')
@click.option('--y-validation', 'validation_categories_path', type=click.Path(exists=True), help='Path to the validation categories file.')
@click.option('--integer-categories', 'integer_category_path', type=click.Path(exists=True), help='Path to the integer category file.')
@click.option('-o', '--output', 'output_path', type=click.Path(), help='Path to save the output confidence dictionary.')
@click.option('--device', default='cpu', help='Device to use for computation (default: cpu)')
@click.option('-f', '--force', is_flag=True, help='Overwrite output directory if it exists.')

def main(model_directory, embeddings_path, categories_path, validation_categories_path, integer_category_path, output_path, device, force):
    if os.path.exists(output_path) and not force:
        logger.error(f"Output directory already exists: {output_path}. Use --force to overwrite.")
        return

    logger.info("Starting the computation of confidence scores.")
    embeddings, categories, validation_categories, phrog_integer = load_data(embeddings_path, categories_path, validation_categories_path, integer_category_path)
    p = create_predictor(model_directory, device)
    conf_dataset_loader = create_dataloader(embeddings, categories, validation_categories)
    all_probs, all_categories, all_labels = process_batches(p, conf_dataset_loader)
    c_dict = compute_confidence_dict(all_categories, all_labels, all_probs, phrog_integer)
    save_confidence_dict(c_dict, output_path)
    logger.info("Finished the computation of confidence scores.")


def load_data(embeddings_path, categories_path, validation_categories_path, integer_category_path):
    logger.info("Loading embeddings and categories.")
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    with open(categories_path, 'rb') as f:
        categories = pickle.load(f)
    with open(validation_categories_path, 'rb') as f:
        validation_categories = pickle.load(f)
    with open(integer_category_path, 'rb') as f:
        phrog_integer = pickle.load(f)

    phrog_integer = dict(
        zip(
            [i - 1 for i in list(phrog_integer.keys())],
            [i for i in list(phrog_integer.values())],
        )
    )
    return embeddings, categories, validation_categories, phrog_integer


def create_predictor(model_directory, device):
    logger.info("Creating the predictor object.")
    p = predictor.Predictor(device=device)
    p.read_models_from_directory(model_directory)
    return p


def create_dataloader(embeddings, categories, validation_categories):
    logger.info("Creating the dataset and dataloader.")
    conf_dataset = model_onehot.EmbeddingDataset(
        list(embeddings.values()), list(categories.values()), mask_portion=0)
    conf_dataset.set_validation(list(validation_categories.values()))
    conf_dataset_loader = DataLoader(conf_dataset, batch_size=16, collate_fn=model_onehot.collate_fn)
    return conf_dataset_loader


def process_batches(p, conf_dataset_loader):
    logger.info("Processing batches.")
    all_probs = []
    all_categories = []
    batches = 0

    total_batches = len(conf_dataset_loader)
    logger.info(f"Total number of batches to process: {total_batches}")

    for embeddings, categories, masks, idx in conf_dataset_loader:
        src_key_padding_mask = (masks != -2)
        phynteny_scores = p.predict_batch(embeddings, src_key_padding_mask)

        batch_size = len(idx)

        for m in range(batch_size):
            scores_at_idx = phynteny_scores[m][idx[m]]
            if len(scores_at_idx.shape) == 1:
                scores_at_idx = scores_at_idx.reshape(1, -1)
            all_probs.append(scores_at_idx)
            all_categories.append(categories[m][idx[m]])

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
    logger.info("Computing the confidence dictionary.")
    bandwidth = np.arange(0, 5, 0.05)[1:]
    c_dict = predictor.build_confidence_dict(all_categories, all_labels, all_probs, bandwidth, phrog_integer)
    return c_dict


def save_confidence_dict(c_dict, output_path):
    logger.info("Saving the confidence dictionary.")
    with open(output_path, 'wb') as f:
        pickle.dump(c_dict, f)


if __name__ == '__main__':
    main()
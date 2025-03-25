#!/usr/bin/env python

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src import format_data
from src import predictor
import click 
from loguru import logger 
import numpy as np 
import torch 
from importlib_resources import files
import pickle
from Bio import SeqIO


__author__ = "Susanna Grigson"
__maintainer__ = "Susanna Grigson"
__license__ = "MIT"
__version__ = "0"
__email__ = "susie.grigson@flinders.edu.au"
__status__ = "development"

@click.command()
@click.argument("infile", type=click.Path(exists=True))
@click.option(
    "-o",
    "--out",
    type=click.STRING,
    default="phynteny",
    help="output directory",
    required=True,
)
@click.option(
    "--prefix",
    type=click.STRING,
    default="phynteny",
    help="output prefix",
    required=False,
)
@click.option(
    "--esm_model",
    default="facebook/esm2_t33_650M_UR50D",
    type=click.Choice([
        "facebook/esm2_t48_15B_UR50D",
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D"
    ], case_sensitive=True),
    help="Specify path to esm model if not the default",
    required=False,
)

@click.option(
    "-m", 
    "--models",
    type=click.Path(exists=True),
    help="Path to the models to use for predictions",
    default = files("phynteny_utils").joinpath("models")
)

@click.option("-f", "--force", is_flag=True, help="Overwrite output directory")

@click.version_option(version=__version__)



def main(infile, out, esm_model, models, force, prefix):

    start_time = time.time()
    # instantiate output directory 
    format_data.instantiate_output_directory(out, force)
    logger.add(out + "/phynteny.log", level="DEBUG")

    # Check if the models directory exists
    if not os.path.exists(models):
        logger.error(f"Models directory does not exist: {models}")
        sys.exit(1)
    else:
        logger.info(f"Models directory exists: {models}")

    # read in needed info on PHROGS 
    category_dict, phrog_integer_category = format_data.read_annotations_information() # what this this method even doing? 

    # preamble 
    logger.info(f'Starting Phynteny transformer v{__version__}') 
    logger.info(f'Command executed: in={infile}, out={out}, esm_model={esm_model}, models={models}, force={force}')
    logger.info(f'Repository homepage is https://github.com/susiegriggo/Phynteny_transformer/tree/main')
    logger.info(f"Written by {__author__}: {__email__}")


    # read in genbank file
    gb_dict = format_data.read_genbank_file(infile, phrog_integer_category) # what categories the genes belong to is missing 
    logger.info("Genbank file read")

    # Extract fasta from the genbank files
    logger.info("Extracting fasta from genbank")
    data_keys = list(gb_dict.keys())
    fasta_out = os.path.join(out, f"{prefix}.fasta")
    records = [gb_dict.get(k).get("sequence") for k in data_keys]
    with open(fasta_out, "w") as output_handle:
        for r in records:
            SeqIO.write(r, output_handle, "fasta")
    output_handle.close()

    # Extract the embeddings from the outputted fasta files
    logger.info("Computing ESM embeddings")
    logger.info("... if step is being slow consider using GPU!")
    embeddings = format_data.extract_embeddings(
        fasta_out, out, model_name=esm_model
    )

    # Prepare data
    X, y = format_data.prepare_data(embeddings, gb_dict)
    src_key_padding = format_data.pad_sequence(list(y.values()))

    # Need to add a one-hot encoding of the categories to the X data
    X_one_hot = {key: torch.tensor(np.hstack((format_data.custom_one_hot_encode(y[key]), X[key]))) for key in X.keys()}

    logger.info("Creating predictor object")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = predictor.Predictor(device=device)
    
    logger.info(f"Reading models from directory: {models}")
    p.read_models_from_directory(models)

    # Read the confidence_dict from a pickled file
    confidence_dict_path = os.path.join(models, "confidence_dict.pkl")
    if os.path.exists(confidence_dict_path):
        with open(confidence_dict_path, 'rb') as f:
            confidence_dict = pickle.load(f)
        logger.info("Loaded confidence dictionary")
    else:
        logger.error(f"Confidence dictionary file does not exist: {confidence_dict_path}")
        sys.exit(1)

    logger.info("Making predictions")
    scores = p.predict(X_one_hot, src_key_padding)
    logger.info(f"f{scores}")

    logger.info("Calculating confidence scores")
    predictions, confidence_scores = p.compute_confidence(scores, confidence_dict, phrog_integer_category)

    logger.info("Writing predictions to genbank file")
    genbank_file = out + "/phynteny.gbk"
    annotated_genes_count = format_data.save_genbank(gb_dict, genbank_file, predictions, scores, confidence_scores)
    
    logger.info("Generating table...")
    table_file = out + "/phynteny.tsv"
    format_data.generate_table(table_file, gb_dict, category_dict, phrog_integer_category)
    
    logger.info(f"Number of genes annotated with confidence above 90%: {annotated_genes_count}")
    
    logger.info("Predictions completed")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()


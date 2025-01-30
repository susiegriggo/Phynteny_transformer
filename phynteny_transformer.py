#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import format_data
from src import predictor
import click 
from loguru import logger 
import pandas as pd
from Bio import SeqIO
import pickle
import numpy as np 
import torch 
from importlib_resources import files
from src.format_data import (
    instantiate_output_directory,
    read_annotations_information,
    read_genbank_file,
    extract_features_and_embeddings,
    pad_sequence,
    custom_one_hot_encode
)


__author__ = "Susanna Grigson"
__maintainer__ = "Susanna Grigson"
__license__ = "MIT"
__version__ = "0"
__email__ = "susie.grigson@gmail.com"
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
    "--esm_model",
    default = "esm2_t33_650M_UR50D",
    type=str,
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

def main(infile, out, esm_model, models, force):
    instantiate_output_directory(out, force)
    logger.add(out + "/phynteny.log", level="DEBUG")
    logger.info("Starting Phynteny")

    category_dict, phrog_integer_category = read_annotations_information() # what this this method even doing? 
    gb_dict = read_genbank_file(infile, phrog_integer_category) # what categories the genes belong to is missing 
    logger.info("Genbank file read")

    logger.info("Extracting features and embeddings")
    embeddings = extract_features_and_embeddings(gb_dict, out, esm_model)
    X, y = format_data.process_data(embeddings, gb_dict) # Think these here are missing the categories
    src_key_padding = pad_sequence(list(y.values()))

    # Need to add a one-hot encoding of the categories to the X data
    X_one_hot = {key: torch.tensor(np.hstack((custom_one_hot_encode(y[key]), X[key]))) for key in X.keys()}

    logger.info("Creating predictor object")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = predictor.Predictor(device=device)
    
    logger.info(f"Reading models from directory: {models}")
    p.read_models_from_directory(models)

    logger.info("Making predictions")
    scores = p.predict(X_one_hot, src_key_padding)
    logger.info(f"f{scores}")

    logger.info("Writing predictions to genbank file")
    genbank_file = out + "/phynteny.gbk"

    logger.info("Generating table...")
    table_file = out + "/phynteny.tsv"
    
    logger.info("Predictions completed")

if __name__ == "__main__":
    main()


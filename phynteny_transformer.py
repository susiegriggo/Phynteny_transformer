#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import format_data
from src import predictor
from src import model_onehot
import click 
from loguru import logger 
import pandas as pd
from Bio import SeqIO
import pickle
import numpy as np 
import torch 
from importlib_resources import files


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
    X, y = format_data.process_data(embeddings, gb_dict) # TODO - up to here, bug in np.hstack 
    src_key_padding = pad_sequence(list(y.values()))

    logger.info("Creating predictor object")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = predictor.Predictor(device=device)
    
    logger.info(f"Reading models from directory: {models}")
    p.read_models_from_directory(models)

    logger.info("Making predictions")
    scores = p.predict(X, src_key_padding)
    
    logger.info("Predictions completed")

def instantiate_output_directory(out, force):
    format_data.instantiate_dir(out, force)

def read_annotations_information():
    phrogs = pd.read_csv(
        "src/phrog_annotation_info/phrog_annot_v4.tsv",
        sep="\t",
    )
    category_dict = dict(zip(phrogs["phrog"], phrogs["category"]))

    phrog_integer = pickle.load(
        open(
            "phynteny_utils/integer_category.pkl", #TODO change this to a path
            "rb",
        )
    )
    phrog_integer = dict(
        zip(
            [i - 1 for i in list(phrog_integer.keys())],
            [i for i in list(phrog_integer.values())],
        )
    )
    phrog_integer_reverse = dict(
        zip(list(phrog_integer.values()), list(phrog_integer.keys()))
    )
    phrog_integer_reverse["unknown function"] = -1
    phrog_integer_category = dict(
        zip(
            [str(k) for k in list(category_dict.keys())],
            [phrog_integer_reverse.get(k) for k in list(category_dict.values())],
        )
    )
    phrog_integer_category["No_PHROG"] = -1

    return category_dict, phrog_integer_category

def read_genbank_file(infile, phrog_integer):
    logger.info("Reading genbank file!")
    #gb_dict = format_data.get_genbank(infile)
    logger.info("Infile: " + infile)
    gb_dict = format_data.get_data(infile, 0, phrog_integer)
    if not gb_dict:
        click.echo("Error: no sequences found in genbank file")
        logger.critcal("No sequences found in genbank file. Nothing to annotate")
        sys.exit()
    logger.info("Genbank file keys`")
    logger.info(gb_dict.keys())
    return gb_dict

def extract_features_and_embeddings(gb_dict, out, esm_model):
    logger.info('Extracting protein sequences as a fasta')
    keys = list(gb_dict.keys())

    extracted_embeddings = dict()

    for k in keys: 
   
        records = gb_dict[k].get('sequence')
        output_handle = out + '/' + k + '.faa'
        SeqIO.write(records, output_handle, "fasta")
        
        logger.info('Generating esm embeddings')
        embeddings = format_data.extract_embeddings(output_handle, out + '/' + k, model_name=esm_model)
        extracted_embeddings[k] = embeddings

    return  extracted_embeddings

#def convert_embeddings(embeddings, gb_features, phrog_integer_category):
#    return format_data.convert_embeddings(embeddings, gb_features, phrog_integer_category)

def pad_sequence(y):
    max_length = np.max([len(i) for i in y]) 
    src_key_padding = np.zeros((len(y), max_length)) - 2 
    for i in range(len(y)):
        src_key_padding[i][:len(y[i])] = 0 
        src_key_padding[i][torch.nonzero(y[i] == -1, as_tuple=False)] = 1
    return src_key_padding

if __name__ == "__main__":
    main()


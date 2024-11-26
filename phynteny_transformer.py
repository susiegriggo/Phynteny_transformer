from src import format_data
import click 
from esm import pretrained
from loguru import logger 
import sys
import pandas as pd
from Bio import SeqIO
import pickle
import gzip
import re 


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
    default="",
    help="output directory",
)

@click.option(
    "--esm_model",
    default = "",
    type=str,
    help="Specify path to esm model if not the default",
    required=False,
)
@click.option("-f", "--force", is_flag=True, help="Overwrite output directory")


@click.version_option(version=__version__)

def main(infile, out,  esm_model, force):


    # generate the output directory
    format_data.instantiate_dir(out, force)

    # generate the logging object
    logger.add(out + "/phynteny.log", level="DEBUG")
    logger.info("Starting Phynteny")

    # get annotations information
    phrogs = pd.read_csv(
        "src/phrog_annotation_info/phrog_annot_v4.tsv",
        sep="\t",
    )
    category_dict = dict(zip(phrogs["phrog"], phrogs["category"]))

    # read in integer encoding of the categories - #TODO try to automate this weird step
    phrog_integer = pickle.load(
        open(
            "src/phrog_annotation_info/integer_category.pkl",
            "rb",
        )
    )
    phrog_integer = dict(
        zip(
            [i - 1 for i in list(phrog_integer.keys())],
            [i for i in list(phrog_integer.values())],
        )
    )  # shuffling the keys down by one
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

    # read in the esm model
    #model, alphabet = pretrained.load_model_and_alphabet(esm_model)
    #model.eval()

    # get entries in the genbank file
    logger.info("Reading genbank file!")
    gb_dict = format_data.get_genbank(infile)
    if not gb_dict:
        click.echo("Error: no sequences found in genbank file")
        logger.critcal("No sequences found in genbank file. Nothing to annotate")
        sys.exit()


    # extract features 
    logger.info('Extracting protein sequences as a fasta')
    keys = list(gb_dict.keys())
    gb_features = {}
    for k in keys: 
        gb_features[k ]= format_data.extract_features(gb_dict.get(k),k )
        records = gb_features[k].get('sequence')
        output_handle = out + '/' + k+ '.faa'
        SeqIO.write(records, output_handle, "fasta")
        
         # generate esm embeddings 
        logger.info('Generating esm embeddings')
        embeddings = format_data.extract_embeddings(output_handle, out + '/' + k)
        
    

if __name__ == "__main__":
    main()

    
from esm import FastaBatchedDataset, pretrained
import torch 
import binascii
import gzip 
import pathlib
import os 
from Bio import SeqIO
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
import pandas as pd
from loguru import logger 
import click 
import re 
import pickle

def get_dict(dict_path):
    """
    Helper function to import dictionaries

    :param dict_path: path to the desired dictionary
    :return: loaded dictionary
    """

    with open(dict_path, "rb") as handle:
        dictionary = pickle.load(handle)
        if any(dictionary):
            logger.info(f"dictionary loaded from {dict_path}")
        else:
            logger.crtical(
                f"dictionary could not be loaded from {dict_path}. Is it empty?"
            )
    handle.close()
    return dictionary

def is_gzip_file(f):
    """
    Method copied from Phispy see https://github.com/linsalrob/PhiSpy/blob/master/PhiSpyModules/helper_functions.py

    This is an elegant solution to test whether a file is gzipped by reading the first two characters.
    I also use a version of this in fastq_pair if you want a C version :)
    See https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed for inspiration
    :param f: the file to test
    :return: True if the file is gzip compressed else false
    """
    with open(f, "rb") as i:
        return binascii.hexlify(i.read(2)) == b"1f8b"

def get_genbank(genbank):
    """
    Convert genbank file to a dictionary and save sequences to fasta files 

    :param genbank: path to a genbank file to read in 
    :return: genbank file as a dictionary
    """


    # if genbank.strip()[-3:] == ".gz":
    if is_gzip_file(genbank.strip()):
        try:
            with gzip.open(genbank.strip(), "rt") as handle:
                # read genbank to a dictionary 
                gb_dict = SeqIO.to_dict(SeqIO.parse(handle, "gb"))
            handle.close()
        except ValueError:
            logger.error(genbank.strip() + " is not a genbank file!")
            raise

    else:
        try:
            with open(genbank.strip(), "rt") as handle:
                # read genbank to a dictionary 
                gb_dict = SeqIO.to_dict(SeqIO.parse(handle, "gb"))
            handle.close()
        except ValueError:
            logger.error(genbank.strip() + " is not a genbank file!")
            raise

    return gb_dict

def get_fasta(input_data, out_dir): 

    with open(input_data, "r", errors="replace") as file:
        genbank_files = file.readlines()

        for genbank in genbank_files: 

            out_fasta = out_dir + '/' + re.split('/',genbank.strip())[-1] + '.faa'

            # if genbank.strip()[-3:] == ".gz":
            if is_gzip_file(genbank.strip()):
                try:
                    with gzip.open(genbank.strip(), "rt") as handle:
                        # read genbank to a dictionary 
                        SeqIO.write(SeqIO.parse(handle, "gb"), out_fasta, 'fasta')
                    handle.close()
                except ValueError:
                    logger.error(genbank.strip() + " is not a genbank file!")
                    raise

            else:
                try:
                    with open(genbank.strip(), "rt") as handle:
                        # read genbank to a dictionary 
                        SeqIO.write(SeqIO.parse(handle, "gb"), out_fasta, 'fasta')
                    handle.close()
                except ValueError:
                    logger.error(genbank.strip() + " is not a genbank file!")
                    raise

def phrog_to_integer(phrog_annot, phrog_integer):
    """
    Converts phrog annotation to its integer representation
    """

    return [phrog_integer.get(i) for i in phrog_annot]

def extract_features(this_phage, key):
    """
    Extract the required features and format as a dictionary

    param this_phage: phage genome extracted from genbank file
    return: dictionary with the features for this specific phage
    """

    phage_length = len(this_phage.seq)
    this_CDS = [i for i in this_phage.features if i.type == "CDS"]  # coding sequences

    position = [
        (int(this_CDS[i].location.start), int(this_CDS[i].location.end))
        for i in range(len(this_CDS))
    ]
    sense = [
        re.split("]", str(this_CDS[i].location))[1][1] for i in range(len(this_CDS))
    ]
    protein_id = [
        this_CDS[i].qualifiers.get("protein_id") for i in range(len(this_CDS))
    ]
    protein_id = [p[0] if p is not None else None for p in protein_id]
    phrogs = [this_CDS[i].qualifiers.get("phrog") for i in range(len(this_CDS))]
    phrogs = ["No_PHROG" if i is None else i[0] for i in phrogs]
    sequence = [SeqRecord(Seq(this_CDS[i].qualifiers.get('translation')[0]), id=key + '_' + str(i), description = key + '_' + str(i)) for i in range(len(this_CDS))] 
    
    return {
        "length": phage_length,
        "phrogs": phrogs,
        "protein_id": protein_id,
        "sense": sense,
        "position": position,
        "sequence": sequence
    }


def get_data(input_data, gene_categories, phrog_integer, maximum_genes=False):
    """
    Loop to fetch training and test data

    :param input_data: path to where the input files are located
    :param gene_categories: number of gene categories which must be included
    :return: curated data dictionary
    """

    training_data = {}  # dictionary to store all of the training data

    with open(input_data, "r", errors="replace") as file:
        genbank_files = file.readlines()
       
        for genbank in genbank_files:
            # convert genbank to a dictionary
            gb_dict = get_genbank(genbank)
            gb_keys = list(gb_dict.keys())
         

            for key in gb_keys:

                # extract the relevant features
                phage_dict = extract_features(gb_dict.get(key), key)
                
                # integer encoding of phrog categories
                integer = phrog_to_integer(phage_dict.get("phrogs"), phrog_integer)
                phage_dict["categories"] = integer
                
                # evaluate the number of categories present in the phage
                categories_present = set(integer)
                if 0 in categories_present:
                    categories_present.remove(0)

                if maximum_genes == False:
                    if len(categories_present) >= gene_categories:

                        # update dictionary with this entry
                        g = re.split(",|\.", re.split("/", genbank.strip())[-1])[0]
                        training_data[g + "_" + key] = phage_dict

                else:
                    # if above the minimum number of categories are included
                    if (
                        len(phage_dict.get("phrogs")) <= maximum_genes
                        and len(categories_present) >= gene_categories
                    ):
                 
                        # update dictionary with this entry
                        g = re.split(",|\.", re.split("/", genbank.strip())[-1])[0]
                        training_data[g + "_" + key] = phage_dict

    return training_data

# this code seems to batch from a fasta file better from the code later 
def extract_embeddings(fasta_file, output_dir, model_name = 'esm2_t33_650M_UR50D', tokens_per_batch=4096, repr_layers=[33]):
    """
    Extract ESM2 embeddings 
    """
    
    # read in the specified esm model 
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    # move model to gpu if available 
    if torch.cuda.is_available():
        model = model.cuda()
      
    # batch the fasta file 
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    # create data loader obj
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=alphabet.get_batch_converter(), 
        batch_sampler=batches
    )

    # make output directory 
    output_path =  pathlib.Path(output_dir + '/esm') 
    output_path.mkdir(parents=True, exist_ok=True)

    # start processing batches 
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            # move tokens to gpu if available 
            if torch.cuda.is_available():
                print('USING CUDA')
                toks = toks.to(device="cuda", non_blocking=True)
            else: 
                 print('Not using CUDA :(')

            print(toks)
            # Extract embeddings 
            with torch.no_grad():
                results = model(toks, repr_layers=repr_layers, return_contacts=False)
            token_representations = results["representations"][33]

            # update this to save dictionary for an entire fasta file 
            results = dict() 
            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                #filename = output_dir + '/' + entry_id + '.pt'
                
                representation = token_representations[i, 1 : len(strs[i]) - 1].mean(0)
                results[label] = representation 
                   
            torch.save(results, output_dir + '/esm/embeddings.pt') 

        return results 

@click.command()
@click.option(
    "-i",
    "--input_data",
    help="Text file containing genbank files to build model",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-m",
    "--maximum_genes",
    type=int,
    help="Specify the maximum number of genes in each genome. Default 120.",
    default=120,
)
@click.option(
    "-g",
    "--gene_categories",
    type=int,
    help="Specify the minimum number of categories in each genome. Default 4.",
    default=4,
)
@click.option(
    "--prefix",
    "-p",
    default="data",
    type=str,
    help="Prefix for the output files",
)
@click.option(
    "--out",
    "-o",
    default="out",
    type=str,
    help="Directory for the output files",
)

def main(input_data, gene_categories, maximum_genes, prefix, out): 

    # read in information for the phrog annotations 
    # read in annotation file
    phrogs = pd.read_csv('/home/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/phrog_annot_v4.tsv', sep='\t')
    category_dict = dict(zip(phrogs['phrog'], phrogs['category']))

    # read in integer encoding of the categories - #TODO try to automate this weird step 
    phrog_integer = pickle.load(open('/home/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/integer_category.pkl', 'rb'))
    phrog_integer = dict(zip([i -1 for i in list(phrog_integer.keys())], [i for i in list(phrog_integer.values())]))
    phrog_integer_reverse = dict(zip(list(phrog_integer.values()), list(phrog_integer.keys())))
    phrog_integer_reverse['unknown function'] = -1
    phrog_integer_category = dict(zip([str(k) for k in list(category_dict.keys())], [phrog_integer_reverse.get(k) for k in list(category_dict.values())]))
    phrog_integer_category['No_PHROG'] = -1

    # Create the output directory if it doesn't exist
    if not os.path.exists(out):
        os.makedirs(out)
        print(f"Directory created: {out}")
    else:
        print(f"Warning: Directory {out} already exists.")
    
    # read in the genbank file/files 
    data = get_data(
        input_data, gene_categories, phrog_integer_category, maximum_genes
    )

    # extract fasta from the genbank files 
    data_keys = list(data.keys()) 
    fasta_out = out + '/' + prefix + '.fasta' 
    records = [data.get(k).get('sequence') for k in data_keys]
    with open(fasta_out, 'w') as output_handle: 
        for r in records:
            SeqIO.write(r, output_handle, "fasta")
    output_handle.close()

    # Extract the embeddings from the outputted fasta files 
    extract_embeddings(fasta_out, out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main() 

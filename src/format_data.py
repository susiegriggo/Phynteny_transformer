from esm import FastaBatchedDataset, pretrained
import torch
import binascii
import gzip
import random
import numpy as np
import pathlib
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from loguru import logger
import re
import shutil
import pickle
import os
import pandas as pd
import sys


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

def instantiate_dir(output_dir, force):
    """
    Generate output directory releative to whether force has been specififed
    """

    # remove the existing outdir on force
    if force == True:
        if os.path.isdir(output_dir) == True:
            shutil.rmtree(output_dir)

        else:
            print(
                "\n--force was specficied even though the output directory does not exist \n"
            )

    # make directory if force is not specified
    else:
        if os.path.isdir(output_dir) == True:
            sys.exit(
                "\nOutput directory already exists and force was not specified. Please specify -f or --force to overwrite the output directory. \n"
            )

    # instantiate the output directory
    os.mkdir(output_dir)



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
            out_fasta = out_dir + "/" + re.split("/", genbank.strip())[-1] + ".faa"

            # if genbank.strip()[-3:] == ".gz":
            if is_gzip_file(genbank.strip()):
                try:
                    with gzip.open(genbank.strip(), "rt") as handle:
                        # read genbank to a dictionary
                        SeqIO.write(SeqIO.parse(handle, "gb"), out_fasta, "fasta")
                    handle.close()
                except ValueError:
                    logger.error(genbank.strip() + " is not a genbank file!")
                    raise

            else:
                try:
                    with open(genbank.strip(), "rt") as handle:
                        # read genbank to a dictionary
                        SeqIO.write(SeqIO.parse(handle, "gb"), out_fasta, "fasta")
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

    # get sequence and replace ambiguous amino acid J with X
    sequence = [
        SeqRecord(
            Seq(this_CDS[i].qualifiers.get("translation")[0].replace("J", "X").rstrip('*')),
            id=key + "_" + str(i),
            description=key + "_" + str(i),
        )
        for i in range(len(this_CDS))
    ]

    return {
        "length": phage_length,
        "phrogs": phrogs,
        "protein_id": protein_id,
        "sense": sense,
        "position": position,
        "sequence": sequence,
    }


def encode_strand(strand):
    """
    One hot encode the direction of each gene

    :param strand: sense encoded as a vector of + and - symbols
    :return: one hot encoding as two separate numpy arrays
    """

    encode = np.array([2 if i == "+" else 1 for i in strand])

    strand_encode = np.array([1 if i == 1 else 0 for i in encode]), np.array(
        [1 if i == 2 else 0 for i in encode]
    )

    return np.array(strand_encode[0]).reshape(-1, 1), np.array(
        strand_encode[1]
    ).reshape(-1, 1)


def encode_length(gene_positions):
    """
    Extract the length of each of each gene so that it can be included in the embedding

    :param gene_positions: list of start and end position of each gene
    :return: length of each gene
    """

    return np.array(
        [round(np.abs(i[1] - i[0]) / 1000, 3) for i in gene_positions]
    ).reshape(-1, 1)

def is_genbank_file(file_path): 
    """
    Check if the file is a Genbank file by looking for specific keywords in the file 

    :param file_path: path to the file to check
    :return: True if the file is a Genbank file, False otherwise
    """

    file_size = os.path.getsize(file_path)
    update_interval = file_size / 10

    with open(file_path, "r") as file: 
        bytes_read = 0
        next_update = update_interval
        for line in file: 
            bytes_read += len(line)
            if line.startswith("LOCUS") or line.startswith("ORIGIN"): 
                return True 
            if bytes_read >= next_update:
                logger.info(f"Read {bytes_read} bytes ({(bytes_read / file_size) * 100:.1f}%) from {file_path}")
                next_update += update_interval
    
    logger.info(f"{file_path} is not a Genbank file")
    return False


def get_data(input_data, gene_categories, phrog_integer, maximum_genes=False):
    """
    Loop to fetch training and test data

    :param input_data: path to a text file or a single genbank path
    :param gene_categories: number of gene categories which must be included
    :return: curated data dictionary
    """

    training_data = {}  # dictionary to store all of the training data
    genbank_files = []

    # check if the input data is a genbank file 
    if is_genbank_file(input_data):
        logger.info("Input data is a genbank file")
        genbank_files = [input_data]    
    else:
        logger.info("Input data is a text file")
        # read in the genbank files from the input data
        with open(input_data, "r") as file:
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
                    # training_data[g + "_" + key] = phage_dict
                    training_data[key] = phage_dict

    return training_data


# this code seems to batch from a fasta file better from the code later
def extract_embeddings(
    fasta_file,
    output_dir,
    model_name="esm2_t33_650M_UR50D",
    tokens_per_batch=4096,
    repr_layers=[33],
):
    """
    Extract ESM2 embeddings
    """

    # read in the specified esm model
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    # move model to gpu if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("USING CUDA :)")
    else:
        print("NO CUDA :()")

    # batch the fasta file
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
    logger.info(f"Number of batches: {len(batches)}")

    # create data loader obj
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )

    # make output directory
    output_path = pathlib.Path(output_dir + "/esm")
    output_path.mkdir(parents=True, exist_ok=True)

    # dictionary to write to
    results = dict()

    # start processing batches
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            logger.info(f"Processing batch {batch_idx + 1} of {len(batches)}")
            logger.info(f"Batch size (tokens): {toks.size()}")

            # move tokens to gpu if available
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            # Log memory usage before processing the batch
            if torch.cuda.is_available():
                logger.info(f"CUDA memory allocated before batch: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
                logger.info(f"CUDA memory reserved before batch: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

            # Extract embeddings
            try:
                model_predictions = model(
                    toks, repr_layers=repr_layers, return_contacts=False
                )
                token_representations = model_predictions["representations"][33]

                # update this to save dictionary for an entire fasta file
                for i, label in enumerate(labels):
                    representation = token_representations[i, 1 : len(strs[i]) - 1].mean(0)
                    if torch.cuda.is_available():
                        results[label] = representation.detach().cpu()
                    else:
                        results[label] = representation

                # Log memory usage after processing the batch
                if torch.cuda.is_available():
                    logger.info(f"CUDA memory allocated after batch: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
                    logger.info(f"CUDA memory reserved after batch: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory: {e}")
                torch.cuda.empty_cache()
                raise

    torch.save(results, output_dir + "/esm/embeddings.pt")
    logger.info("Embeddings saved successfully")

    return results


def process_data(
    esm_vectors, genome_details, extra_features=True, exclude_embedding=False
):
    """
    frame as a supervised learning problem
    """

    genomes = list(genome_details.keys())


    X = list()
    y = list()
    removed = []

    logger.info(f'ESm vectors: {esm_vectors}')
    logger.info(f'Esm vector keys: {esm_vectors.keys()}')

    for g in genomes:

        logger.info(f"Processing genome {g}")   

        # get the genes in this genome
        this_genes = genome_details.get(g)

        # get the corresponding functions - this is included in the object
        # this_categories = [plm_integer.get(i) if plm_integer.get(i) is not None else -1 for i in this_genes]
        this_categories = genome_details.get(g).get("categories")

        # handle if extra features are specified separateley
        if extra_features:
            # print(genome_details.get(genome_label))
            strand1, strand2 = encode_strand(genome_details.get(g).get("sense"))

            # get the length of each gene
            gene_lengths = encode_length(genome_details.get(g).get("position"))

            # fetch the embeddings for this genome
            if exclude_embedding:
                this_vectors = [[] for i in this_genes]
            else:
             
                #this_vectors = [
                #    esm_vectors.get(g + "_" + str(i)).numpy().astype(np.float32)
                #    for i in range(len(this_categories))
                #]
                
                this_keys = [k for k in list(esm_vectors.keys()) if f"{g}_" in k]
                this_vectors = [esm_vectors.get(k) for k in this_keys]

            # merge these columns into a numpy array
            embedding = np.hstack( #TODO change the shapping of this vectors 
                (strand1, strand2, gene_lengths, np.array(this_vectors).reshape(len(this_vectors),len(this_vectors[0])))
            )

        else:
            # merge vectors into a numpy array
            embedding = np.array(this_vectors).astype(np.float32)

        # store the info in the dataset
        X.append(torch.tensor(np.array(embedding)))
        # y.append(torch.tensor(np.array(this_categories)))
        y.append(torch.tensor(this_categories))

    print("Numer of genomes removed: " + str(len(removed)))
    print("Numer of genomes kept: " + str(len(X)))

    # convert to dictionary inc the dictionary names
    X = dict(zip(genomes, X))
    y = dict(zip(genomes, y))

    return X, y


def derep_data(data):
    """
    Deduplicate repeated gene orders for a phynteny dictionary
    """

    # get the training keys and encodings
    keys = list(data.keys())

    # randomly shuffle the keys
    random.shuffle(keys)

    # get the cagtegories in each phage
    category_encodings = [data.get(k).get("categories") for k in keys]

    # remove duplicates in the data
    category_str = ["".join([str(j) for j in i]) for i in category_encodings]

    # get the deduplicated keys
    dedup_keys = list(dict(zip(category_str, keys)).values())

    return dict(zip(dedup_keys, [data.get(d) for d in dedup_keys]))


def instantiate_output_directory(out, force):
    instantiate_dir(out, force)

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
    #gb_dict = get_genbank(infile)
    logger.info("Infile: " + infile)
    gb_dict = get_data(infile, 0, phrog_integer)
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
        embeddings = extract_embeddings(output_handle, out + '/' + k, model_name=esm_model)
        extracted_embeddings[k] = embeddings

    return  extracted_embeddings

def pad_sequence(y):
    max_length = np.max([len(i) for i in y]) 
    src_key_padding = np.zeros((len(y), max_length)) - 2 
    for i in range(len(y)):
        src_key_padding[i][:len(y[i])] = 0 
        src_key_padding[i][torch.nonzero(y[i] == -1, as_tuple=False)] = 1
    return src_key_padding

def custom_one_hot_encode(data, num_classes=10):
    """
    Generate a one-hot encoding for the given data.

    Parameters:
    data (torch.Tensor): Data tensor to encode.
    num_classes (int): Number of classes to encode.

    Returns:
    np.array: One-hot encoded array.
    """

    one_hot_encoded = []
    for value in data:
        if value == -1:
            one_hot_encoded.append([0] * num_classes)
        else:
            one_hot_row = [0] * num_classes
            one_hot_row[value] = 1
            one_hot_encoded.append(one_hot_row)
    return np.array(one_hot_encoded)

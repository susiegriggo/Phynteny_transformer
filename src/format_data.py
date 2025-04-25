import torch
import binascii
import gzip
import random
import numpy as np
import pathlib
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from loguru import logger
import traceback
import re
import shutil
import pickle
import os
import pandas as pd
import sys
import click  # Add this import for click.open_file
from transformers import EsmModel, EsmTokenizer
from transformers import EsmForMaskedLM
import pkg_resources


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
    Generate output directory relative to whether force has been specified

    :param output_dir: path to the output directory
    :param force: boolean indicating whether to force overwrite existing directory
    """
    # Check if directory exists first, before any processing
    if os.path.isdir(output_dir):
        # If directory exists and force is not specified, exit immediately
        if not force:
            sys.exit(
                "\nOutput directory already exists and force was not specified. "
                "Please specify -f or --force to overwrite the output directory.\n"
            )
        else:
            # Only try to remove if force is true and dir exists
            try:
                # Try to remove the directory with its contents
                shutil.rmtree(output_dir, ignore_errors=True)
                
                # If the directory still exists, try a more aggressive approach
                if os.path.exists(output_dir):
                    logger.warning(f"Could not completely remove {output_dir} with shutil.rmtree. Trying manual deletion.")
                    for root, dirs, files in os.walk(output_dir, topdown=False):
                        for name in files:
                            try:
                                os.remove(os.path.join(root, name))
                            except OSError as e:
                                logger.warning(f"Could not remove file {os.path.join(root, name)}: {e}")
                        for name in dirs:
                            try:
                                os.rmdir(os.path.join(root, name))
                            except OSError as e:
                                logger.warning(f"Could not remove directory {os.path.join(root, name)}: {e}")
                    try:
                        os.rmdir(output_dir)
                    except OSError as e:
                        logger.warning(f"Could not remove root directory {output_dir}: {e}")
            except Exception as e:
                logger.error(f"Error removing directory {output_dir}: {e}")
                sys.exit(1)
    elif force:
        # If directory doesn't exist but force was specified, just inform user
        print("\n--force was specified even though the output directory does not exist\n")

    # Create the output directory
    try:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        elif force:
            # If we get here, it means force=True but directory still exists
            logger.warning(f"Directory {output_dir} still exists despite removal attempt. Continuing since --force was specified.")
    except Exception as e:
        logger.error(f"Error creating directory {output_dir}: {e}")
        sys.exit(1)


def is_gzip_file(f):
    """
    Check if a file is gzip compressed

    :param f: the file to test
    :return: True if the file is gzip compressed, else False
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
    """
    Convert genbank files listed in input_data to fasta format and save to out_dir

    :param input_data: path to a text file containing paths to genbank files
    :param out_dir: directory to save the fasta files
    """
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

    :param phrog_annot: list of phrog annotations
    :param phrog_integer: dictionary mapping phrog annotations to integers
    :return: list of integer representations of phrog annotations
    """

    return [phrog_integer.get(i) for i in phrog_annot]


def extract_features(this_phage, key):
    """
    Extract the required features and format as a dictionary

    :param this_phage: phage genome extracted from genbank file
    :param key: unique identifier for the phage
    :return: dictionary with the features for this specific phage
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
    
    # Save all original feature qualifiers - critical for preserving information
    all_qualifiers = []
    for i in range(len(this_CDS)):
        feature_qualifiers = {}
        for key_qual, value_qual in this_CDS[i].qualifiers.items():
            # Copy all qualifiers to preserve them for later
            feature_qualifiers[key_qual] = value_qual
        all_qualifiers.append(feature_qualifiers)

    logger.debug(f"Extracted {len(all_qualifiers)} genes from {key}")

    # get sequence and replace ambiguous amino acid J with X
    sequence = []
    for i in range(len(this_CDS)):
        # Get the translation if it exists
        translation = this_CDS[i].qualifiers.get("translation", [""])[0].replace("J", "X").rstrip('*')
        
        record = SeqRecord(
            Seq(translation),
            id=key + "_" + str(i),
            description=key + "_" + str(i),
        )
        
        # Important: Add original qualifiers to the record features
        if translation:
            # Create a CDS feature with all the original qualifiers
            feature = SeqFeature(
                FeatureLocation(0, len(translation)),
                type="CDS",
                qualifiers=all_qualifiers[i]
            )
            record.features = [feature]
            
        sequence.append(record)

    # Store the original genome sequence
    original_sequence = str(this_phage.seq)

    return {
        "length": phage_length,
        "phrogs": phrogs,
        "protein_id": protein_id,
        "sense": sense,
        "position": position,
        "sequence": sequence,
        "all_qualifiers": all_qualifiers,  # Store all qualifiers explicitly
        "original_sequence": original_sequence,  # Store the original genome sequence
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
    Extract the length of each gene so that it can be included in the embedding

    :param gene_positions: list of start and end positions of each gene
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


def fetch_data(input_data, gene_categories, phrog_integer, maximum_genes=False):
    """
    Loop to fetch training and test data

    :param input_data: path to a text file or a single genbank path
    :param gene_categories: number of gene categories which must be included
    :param phrog_integer: dictionary mapping phrog annotations to integers
    :param maximum_genes: maximum number of genes to include, default is False
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
    
    skipped_genomes = []

    for genbank in genbank_files:
        # convert genbank to a dictionary
        gb_dict = get_genbank(genbank)
        gb_keys = list(gb_dict.keys())

        for key in gb_keys:
            # extract the relevant features
            phage_dict = extract_features(gb_dict.get(key), key)

            # Check for CDS with more than 10,000 amino acids
            skip_genome = False
            sequences = phage_dict.get("sequence")
            for seq in sequences:
                if len(seq.seq) > 10000:
                    logger.warning(f"Genome {key} contains a CDS with more than 10,000 amino acids and will be excluded from further analysis.")
                    skip_genome = True
                    break

            if skip_genome:
                skipped_genomes.append(key)
                continue

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
                    g = re.split(r",|\.", re.split("/", genbank.strip())[-1])[0]

                    training_data[key] = phage_dict
                    #training_data[f"{g}_{key}"] = phage_dict
                else:
                    skipped_genomes.append(key)

            else:
                # if above the minimum number of categories are included
                if (
                    len(phage_dict.get("phrogs")) <= maximum_genes
                    and len(categories_present) >= gene_categories
                ):
                    # update dictionary with this entry
                    g = re.split(r",|\.", re.split("/", genbank.strip())[-1])[0]
                    logger.info(f"g: {g}")
                    logger.info(f"genbank.strip(): {genbank.strip()}")
                    logger.info(f're.split("/", genbank.strip())[-1]: {re.split("/", genbank.strip())[-1]}')
                    

                    training_data[key] = phage_dict
                    #training_data[f"{g}_{key}"] = phage_dict
                else:
                    skipped_genomes.append(key)

    # Log skipped genomes
    with open("skipped_genomes.txt", "w") as file:
        for genome in skipped_genomes:
            file.write(f"{genome}\n")
            logger.info(f"Skipped genome: {genome}")

    return training_data


### this code seems to batch from a fasta file better from the code later
def read_fasta(file_path):
    """
    Read sequences from a fasta file.

    :param file_path: path to the fasta file
    :return: list of headers and list of sequences
    """
    headers = []
    sequences = []
    with open(file_path, "r") as file:
        sequence = ""
        header = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    headers.append(header)
                    sequence = ""
                header = line[1:].strip()
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
            headers.append(header)
    return headers, sequences

def batch_sequences(sequences, tokens_per_batch, tokenizer):
    """
    Batch sequences for processing.

    :param sequences: list of sequences
    :param tokens_per_batch: maximum number of tokens per batch
    :param tokenizer: tokenizer to use for encoding sequences
    :return: list of batches
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for seq in sequences:
        tokenized_seq = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
        num_tokens = tokenized_seq.input_ids.size(1)
        if current_tokens + num_tokens > tokens_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(seq)
        current_tokens += num_tokens

    if current_batch:
        batches.append(current_batch)

    return batches

def load_model_from_checkpoint(checkpoint_path, base_model_name, cache_dir="/path/to/your/cache/directory"):
    """
    Load a model from a specified checkpoint.

    :param checkpoint_path: Path to the checkpoint file
    :param base_model_name: Name of the base model
    :param cache_dir: Directory to use for caching models and other files
    :return: Loaded model and tokenizer
    """
    ## Set the cache directory
    #os.environ['TRANSFORMERS_CACHE'] = cache_dir

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = EsmModel.from_pretrained(base_model_name, cache_dir=cache_dir)
    #model = EsmForMaskedLM.from_pretrained(base_model_name, cache_dir=cache_dir)
    tokenizer = EsmTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(new_state_dict, strict=False)
    return model, tokenizer

def extract_embeddings(
    fasta_file,
    output_dir,
    model_name="facebook/esm2_t33_650M_UR50D",
    tokens_per_batch=4096,
    max_length=1024,
    checkpoint_path=None,
    cache_dir="cache/"  # Add cache_dir parameter
):
    """
    Extract ESM2 embeddings using HuggingFace

    :param fasta_file: Path to the fasta file
    :param output_dir: Directory to save the embeddings
    :param model_name: Name of the base model
    :param tokens_per_batch: Maximum number of tokens per batch
    :param max_length: Maximum length of the sequences
    :param checkpoint_path: Path to the model checkpoint
    :param cache_dir: Directory to use for caching models and other files
    :return: Dictionary of embeddings
    """
    # Set the cache directory if provided
    #if cache_dir:
    #    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    if checkpoint_path:
        model, tokenizer = load_model_from_checkpoint(checkpoint_path, model_name, cache_dir)
        logger.info(f"Loaded model: {model_name}")
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}") 
    else:
        #tokenizer = EsmTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info(f"Loaded model: {model_name}")
        logger.info(f"Loaded model without checkpoint")
    model.eval()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("USING CUDA :)")
    else:
        logger.info("CUDA NOT IN USE :(")

    # Read and batch the fasta file
    headers, sequences = read_fasta(fasta_file)
    batches = batch_sequences(sequences, tokens_per_batch, tokenizer)
    logger.info(f"Number of batches: {len(batches)}")

    # Make output directory
    output_path = pathlib.Path(output_dir + "/esm")
    output_path.mkdir(parents=True, exist_ok=True)

    # Dictionary to write to
    results = dict()

    # Start processing batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(batches):
            logger.debug(f"Processing batch {batch_idx + 1} of {len(batches)}")

            # Tokenize the batch
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            if torch.cuda.is_available():
                inputs = {key: val.cuda() for key, val in inputs.items()}

            # Extract embeddings
            try:
                
                if checkpoint_path is None:
                    outputs = model(**inputs)
                    token_representations = outputs.last_hidden_state
                
                else: 
                    outputs = model(**inputs, output_hidden_states=True)
                    token_representations = outputs.hidden_states[-1]
                    
                #pooled_representations = outputs.pooler_output

                # Update this to save dictionary for an entire fasta file
                for i, seq in enumerate(batch):
                    representation = token_representations[i, 1 : len(seq) - 1].mean(0)
                    #representation = pooled_representations[i]
                    header = headers.pop(0)  # Use the header as the key
                    if torch.cuda.is_available():
                        results[header] = representation.detach().cpu()
                    else:
                        results[header] = representation

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory: {e}")
                torch.cuda.empty_cache()
                raise

    torch.save(results, output_dir + "/esm/embeddings.pt")
    logger.info("Embeddings saved successfully")

    return results


def prepare_data(
    esm_vectors, genome_details, extra_features=True, exclude_embedding=False
):
    """
    Frame as a supervised learning problem

    :param esm_vectors: dictionary of ESM vectors
    :param genome_details: dictionary with genome details
    :param extra_features: boolean indicating whether to include extra features
    :param exclude_embedding: boolean indicating whether to exclude embeddings
    :return: tuple of X and y data
    """

    genomes = list(genome_details.keys())


    X = list()
    y = list()
    removed = []

    for g in genomes:

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
                

                # extra and sort the keys with a lambda function 
                this_keys = sorted([k for k in list(esm_vectors.keys()) if f"{g}_" in k], key=lambda x: int(x.split('_')[-1]))
                this_vectors = [esm_vectors.get(k) for k in this_keys] 
                if not this_vectors:
                    removed.append(g)
                    continue
           
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


    # convert to dictionary inc the dictionary names
    X = dict(zip(genomes, X))
    y = dict(zip(genomes, y))

    return X, y


def prepare_y_data(genome_details):
    """
    Prepare the y data only.

    :param genome_details: dictionary with genome details
    :return: y data
    """

    genomes = list(genome_details.keys())
    y = list()

    for g in genomes:
        # get the corresponding functions - this is included in the object
        this_categories = genome_details.get(g).get("categories")

        # Replace None with -1
        this_categories = [-1 if category is None else category for category in this_categories]
        y.append(torch.tensor(this_categories))

    # convert to dictionary inc the dictionary names
    y = dict(zip(genomes, y))

    return y


def derep_data(data):
    """
    Deduplicate repeated gene orders for a phynteny dictionary

    :param data: dictionary with phynteny data
    :return: deduplicated data dictionary
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
    """
    Instantiate the output directory

    :param out: path to the output directory
    :param force: boolean indicating whether to force overwrite existing directory
    """
    instantiate_dir(out, force)

def read_annotations_information():
    """
    Read annotations information

    :return: tuple of category_dict and phrog_integer_category
    """
    # Try to find the phrog_annot_v4.tsv file using pkg_resources
    try:
        phrog_file_path = pkg_resources.resource_filename('src', 'phrog_annotation_info/phrog_annot_v4.tsv')
        if not os.path.exists(phrog_file_path):
            # Fallback to alternate locations
            alternate_paths = [
                pkg_resources.resource_filename('phynteny_utils', 'phrog_annotation_info/phrog_annot_v4.tsv'),
                os.path.join(os.path.dirname(__file__), 'phrog_annotation_info/phrog_annot_v4.tsv'),
                "src/phrog_annotation_info/phrog_annot_v4.tsv"  # Original path as last resort
            ]
            
            for path in alternate_paths:
                if os.path.exists(path):
                    phrog_file_path = path
                    break
        
        logger.info(f"Using phrog annotation file: {phrog_file_path}")
        phrogs = pd.read_csv(phrog_file_path, sep="\t")
        
    except (pkg_resources.DistributionNotFound, FileNotFoundError) as e:
        logger.error(f"Error finding phrog_annot_v4.tsv: {e}")
        # Fallback to original path
        logger.warning("Falling back to hardcoded path")
        phrogs = pd.read_csv("src/phrog_annotation_info/phrog_annot_v4.tsv", sep="\t")
    
    category_dict = dict(zip(phrogs["phrog"], phrogs["category"]))

    # manually add additional categories that may be found with phold 
    category_dict['vfdb'] = 'moron, auxiliary metabolic gene and host takeover'
    category_dict['netflax'] = 'moron, auxiliary metabolic gene and host takeover'
    category_dict['acr'] = 'moron, auxiliary metabolic gene and host takeover'
    category_dict['card'] = 'moron, auxiliary metabolic gene and host takeover'
    category_dict['defensefinder'] = 'moron, auxiliary metabolic gene and host takeover'

    # Try to find the integer_category.pkl file
    try:
        integer_file_path = pkg_resources.resource_filename('phynteny_utils', 'integer_category.pkl')
        if not os.path.exists(integer_file_path):
            alternate_paths = [
                "phynteny_utils/integer_category.pkl",  # Original path
                os.path.join(os.path.dirname(__file__), '../phynteny_utils/integer_category.pkl'),
                os.path.join(os.path.dirname(__file__), 'phynteny_utils/integer_category.pkl')
            ]
            for path in alternate_paths:
                if os.path.exists(path):
                    integer_file_path = path
                    break
            else:
                raise FileNotFoundError("integer_category.pkl not found in any known locations.")
        
        logger.info(f"Using integer category file: {integer_file_path}")
        phrog_integer = pickle.load(open(integer_file_path, "rb"))
        
    except (pkg_resources.DistributionNotFound, FileNotFoundError) as e:
        logger.error(f"Error finding integer_category.pkl: {e}")
        sys.exit("Critical error: integer_category.pkl file is missing. Please ensure it is correctly installed.")
    
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
    """
    Read genbank file

    :param infile: path to the genbank file
    :param phrog_integer: dictionary mapping phrog annotations to integers
    :return: genbank dictionary
    """
    logger.info("Reading genbank file!")

    logger.info("Infile: " + infile)
    gb_dict = fetch_data(infile, 0, phrog_integer)
    if not gb_dict:
        click.echo("Error: no sequences found in genbank file")
        logger.critcal("No sequences found in genbank file. Nothing to annotate")
        sys.exit()

    return gb_dict

def extract_features_and_embeddings(gb_dict, out, esm_model):
    """
    Extract features and embeddings

    :param gb_dict: dictionary with genbank data
    :param out: output directory
    :param esm_model: name of the ESM model
    :return: dictionary of extracted embeddings
    """
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
    """
    Pad sequences to the same length

    :param y: list of sequences
    :return: padded sequences
    """
    max_length = np.max([len(i) for i in y]) 
    src_key_padding = np.zeros((len(y), max_length)) - 2 
    for i in range(len(y)):
        src_key_padding[i][:len(y[i])] = 0 
        src_key_padding[i][torch.nonzero(y[i] == -1, as_tuple=False)] = 1
    return src_key_padding

def custom_one_hot_encode(data, num_classes=9):
    """
    Generate a one-hot encoding for the given data.

    :param data: Data tensor to encode
    :param num_classes: Number of classes to encode
    :return: One-hot encoded array
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

def save_genbank(gb_dict, genbank_file, predictions, scores, confidence_scores):
    """
    Save the genbank file with predictions.

    :param gb_dict: Dictionary of genbank records
    :param genbank_file: Path to save the genbank file
    :param predictions: List of predictions
    :param scores: List of scores
    :param confidence_scores: List of confidence scores
    :return: Number of genes annotated with confidence above threshold
    """
    # Threshold for considering a gene as annotated
    threshold = 0.9
    annotated = 0

    with open(genbank_file, "w") as handle:
        for i, k in enumerate(gb_dict.keys()):
            # Handle different possible structures of gb_dict
            if isinstance(gb_dict[k], dict) and "sequence" in gb_dict[k]:
                # Extract the sequence list from a dictionary structure
                sequence_records = gb_dict[k]["sequence"]
                
                # Process each SeqRecord in the sequence list
                for j, record in enumerate(sequence_records):
                    if j < len(scores[i]) and hasattr(record, 'features'):
                        # Add annotations directly to the record
                        if not hasattr(record, 'annotations'):
                            record.annotations = {}
                        
                        # Add required molecule_type annotation for GenBank format
                        if "molecule_type" not in record.annotations:
                            record.annotations["molecule_type"] = "protein"
                        
                        record.annotations["phynteny"] = str(int(predictions[i][j]))
                        record.annotations["phynteny_score"] = str(scores[i][j])
                        record.annotations["phynteny_confidence"] = str(confidence_scores[i])
                        
                        # Count genes annotated with high confidence
                        logger.info(f"Confidence score for {k}: {confidence_scores[i]}")
                        if confidence_scores[i][j] > threshold:
                            annotated += 1
                        
                        # Write record to handle
                        SeqIO.write(record, handle, "genbank")
            elif hasattr(gb_dict[k], "features"):
                # It's directly a SeqRecord object
                record = gb_dict[k]
                
                # Add required molecule_type annotation for GenBank format
                if not hasattr(record, 'annotations'):
                    record.annotations = {}
                if "molecule_type" not in record.annotations:
                    record.annotations["molecule_type"] = "DNA"
                
                # Get CDS features
                cds = [f for f in record.features if f.type == "CDS"]
                
                # Add annotations
                for j, feature in enumerate(cds):
                    if j < len(scores[i]):
                        if "phynteny" not in feature.qualifiers:
                            feature.qualifiers["phynteny"] = []
                        feature.qualifiers["phynteny"] = str(int(predictions[i][j]))
                        
                        if "phynteny_score" not in feature.qualifiers:
                            feature.qualifiers["phynteny_score"] = []
                        feature.qualifiers["phynteny_score"] = str(scores[i][j])
                        
                        if "phynteny_confidence" not in feature.qualifiers:
                            feature.qualifiers["phynteny_confidence"] = []
                        feature.qualifiers["phynteny_confidence"] = str(confidence_scores[i])
                        
                        # Count genes annotated with high confidence
                        if confidence_scores[i] > threshold:
                            annotated += 1
                
                # Write record to handle
                SeqIO.write(record, handle, "genbank")
            else:
                logger.warning(f"Skipping {k}: Could not extract SeqRecord object")
    
    return annotated

def generate_table(outfile, gb_dict, categories, phrog_integer, predictions=None, scores=None, confidence_scores=None, threshold=0.9, categories_map=None):
    """
    Generate table summary of the annotations made.

    :param outfile: Path to the output file
    :param gb_dict: Dictionary of Genbank records
    :param categories: Dictionary of categories
    :param phrog_integer: Dictionary mapping phrog annotations to integers
    :param predictions: List of predictions for each sequence (optional)
    :param scores: List of scores for each prediction (optional)
    :param confidence_scores: List of confidence scores for each prediction (optional) 
    :param threshold: Confidence threshold (default: 0.9)
    :param categories_map: Dictionary mapping prediction numbers to human-readable labels
    :return: None
    """
    # Get the list of phages to loop through
    keys = list(gb_dict.keys())

    # Convert annotations made to a text file
    with click.open_file(outfile, "wt") if outfile != ".tsv" else sys.stdout as f:
        f.write(
            "ID\tstart\tend\tstrand\tphrog_id\tphrog_category\tphynteny_category\tphynteny_score\tphynteny_confidence\tphage\n"
        )

        for genome_idx, k in enumerate(keys):
            logger.debug(f"Processing genome #{genome_idx}: {k}")
            
            try:
                # Check if gb_dict[k] is a dictionary with sequence field
                if isinstance(gb_dict.get(k), dict) and 'sequence' in gb_dict.get(k):
                    sequences = gb_dict.get(k)['sequence']
                    logger.debug(f"Genome {k} has {len(sequences)} sequences")
                    
                    # Check availability of prediction data for this genome
                    has_predictions = predictions is not None and genome_idx < len(predictions)
                    has_scores = scores is not None and genome_idx < len(scores)
                    has_confidence = confidence_scores is not None and genome_idx < len(confidence_scores)
                    
                    logger.debug(f"Has predictions: {has_predictions}, Has scores: {has_scores}, Has confidence: {has_confidence}")
                    
                    # For each sequence in the genome
                    for seq_idx, record in enumerate(sequences):
                        # Get sequence ID
                        seq_id = record.id if hasattr(record, 'id') else f"{k}_{seq_idx}"
                        
                        # Get position information
                        start = 0
                        end = len(record.seq)
                        if 'position' in gb_dict[k] and seq_idx < len(gb_dict[k]['position']):
                            start, end = gb_dict[k]['position'][seq_idx]
                        
                        # Get strand information
                        strand = "+"
                        if 'sense' in gb_dict[k] and seq_idx < len(gb_dict[k]['sense']):
                            strand = gb_dict[k]['sense'][seq_idx]
                        
                        # Get PHROG information
                        phrog_id = "No_PHROG"
                        phrog_category = "unknown function"
                        if 'phrogs' in gb_dict[k] and seq_idx < len(gb_dict[k]['phrogs']):
                            phrog_id = gb_dict[k]['phrogs'][seq_idx]
                            phrog_id = int(phrog_id) if phrog_id.isdigit() else phrog_id
                            if phrog_id in categories:
                                phrog_category = categories[phrog_id]

                        
                        # Default values for prediction columns
                        phynteny_category = "NA"
                        phynteny_score = "NA"
                        phynteny_confidence = "NA"
                        
                        # if the phrog is unknown in the genbank file 
                        if phrog_category == "unknown function":

                            # Extract Category if available 
                            if has_predictions and seq_idx < len(predictions[genome_idx]):
                                pred = predictions[genome_idx][seq_idx]
                                pred_int = int(pred)
                                logger.debug(f"Prediction for {k}, seq {seq_idx}: {pred_int}")
                                
                                # Convert prediction number to label if categories_map is provided
                                if categories_map is not None and pred_int in categories_map:
                                    phynteny_category = categories_map[pred_int]
                                    logger.debug(f"Category mapped to: {phynteny_category}")
                                else:
                                    phynteny_category = str(pred_int)
                                    if categories_map is not None:
                                        logger.warning(f"Category {pred_int} not found in categories_map ({list(categories_map.keys())[:5]}...)")
                            else:
                                logger.debug(f"No predictions array for genome {k}")
                                
                            # Extract score if available
                            if has_scores and seq_idx < len(scores[genome_idx]):
                                score_arr = scores[genome_idx][seq_idx]
                                logger.debug(f"Score array for {k}, seq {seq_idx}: {type(score_arr)}")
                                
                                if isinstance(score_arr, np.ndarray) and len(score_arr) > 0 and 'pred_int' in locals():
                                    if pred_int < len(score_arr):
                                        max_score = float(score_arr[pred_int])
                                        phynteny_score = f"{max_score:.4f}"
                                        logger.debug(f"Score: {phynteny_score}")
                                    else:
                                        logger.warning(f"Score index out of range: pred_int {pred_int} >= len(score_arr) {len(score_arr)}")
                                else:
                                    try:
                                        phynteny_score = f"{float(score_arr):.4f}"
                                    except (TypeError, ValueError):
                                        logger.warning(f"Could not convert score to float: {score_arr}")
                                        
                            # Extract confidence if available
                            if has_confidence:
                                if isinstance(confidence_scores[genome_idx], np.ndarray):
                                    if seq_idx < len(confidence_scores[genome_idx]):
                                        conf = confidence_scores[genome_idx][seq_idx]
                                        phynteny_confidence = f"{float(conf):.4f}"
                                        logger.debug(f"Confidence: {phynteny_confidence}")
                                    else:
                                        logger.warning(f"Confidence index out of bounds: seq_idx {seq_idx} >= len(confidence_scores[genome_idx]) {len(confidence_scores[genome_idx])}")
                                else:
                                    conf = confidence_scores[genome_idx]
                                    phynteny_confidence = f"{float(conf):.4f}"
                                    logger.debug(f"Genome-level confidence: {phynteny_confidence}")
                            
                        # Write row to table file
                        f.write(f"{seq_id}\t{start}\t{end}\t{strand}\t{phrog_id}\t{phrog_category}\t{phynteny_category}\t{phynteny_score}\t{phynteny_confidence}\t{k}\n")
                
                else:
                    logger.warning(f"Genome {k} does not have expected structure. Skipping...")
            
            except Exception as e:
                logger.error(f"Error processing genome {k} for table: {str(e)}")
                logger.error(traceback.format_exc())
    
    logger.info(f"Table generation complete.")
    return None

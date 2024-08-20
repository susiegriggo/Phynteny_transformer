from esm import FastaBatchedDataset, pretrained
import torch 
import binascii
import gzip 
import numpy as np
import pathlib
from Bio import SeqIO
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
from loguru import logger 
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

    # get sequence and replace ambiguous amino acid J with X 
    sequence = [SeqRecord(Seq(this_CDS[i].qualifiers.get('translation')[0].replace('J', 'X')),  
                          id=key + '_' + str(i), 
                          description = key + '_' + str(i)) for i in range(len(this_CDS))] 
    
    return {
        "length": phage_length,
        "phrogs": phrogs,
        "protein_id": protein_id,
        "sense": sense,
        "position": position,
        "sequence": sequence
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

    return np.array(strand_encode[0]).reshape(-1,1), np.array(strand_encode[1]).reshape(-1,1)

def encode_length(gene_positions): 
    """
    Extract the length of each of each gene so that it can be included in the embedding 

    :param gene_positions: list of start and end position of each gene 
    :return: length of each gene 
    """

    return np.array([round(np.abs(i[1] - i[0])/1000, 3) for i in gene_positions]).reshape(-1,1)
 

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
                        #training_data[g + "_" + key] = phage_dict
                        training_data[key] = phage_dict

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
        print('USING CUDA :)')
    else: 
        print('NO CUDA :()')
      
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

    # dictionary to write to 
    results = dict() 
    
    # start processing batches 
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            # move tokens to gpu if available 
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            # Extract embeddings 
            with torch.no_grad():
                model_predictions = model(toks, repr_layers=repr_layers, return_contacts=False)
            token_representations = model_predictions["representations"][33]

            # update this to save dictionary for an entire fasta file 
            for i, label in enumerate(labels):
                representation = token_representations[i, 1 : len(strs[i]) - 1].mean(0)
                if torch.cuda.is_available(): 
                    results[label] = representation.detach().cpu()
                else: 
                    results[label] = representation
                   
    torch.save(results, output_dir + '/esm/embeddings.pt') 

    return results 

def process_data(esm_vectors, genome_details, extra_features = True, exclude_embedding=False):
    """
    frame as a supervised learning problem
    """
    
    genomes = list(genome_details.keys())
    
    X = list()
    y = list()
    removed = []

    for g in genomes:

        # get the genes in this genome
        this_genes = genome_details.get(g)

        # get the corresponding functions - this is included in the object 
        #this_categories = [plm_integer.get(i) if plm_integer.get(i) is not None else -1 for i in this_genes]
        this_categories = genome_details.get(g).get('categories')

        # handle if extra features are specified separateley
        if extra_features: 
                
            #print(genome_details.get(genome_label))
            strand1, strand2  = encode_strand(genome_details.get(g).get('sense'))

            # get the length of each gene 
            gene_lengths = encode_length(genome_details.get(g).get('position'))

            # fetch the embeddings for this genome 
            if exclude_embedding: 
                this_vectors = [[] for i in this_genes]
            else: 
                this_vectors = [esm_vectors.get(g + '_' + str(i)) for i in range(len(this_categories))]

            # merge these columns into a numpy array 
            embedding = np.hstack((strand1, strand2, gene_lengths, np.array(this_vectors)))
   
        else: 
            
            # merge vectors into a numpy array 
            embedding = np.array(this_vectors)

        # store the info in the dataset
        X.append(torch.tensor(np.array(embedding)))
        #y.append(torch.tensor(np.array(this_categories)))
        y.append(torch.tensor(this_categories))
    
    print('Numer of genomes removed: ' + str(len(removed)))
    print('Numer of genomes kept: ' + str(len(X)))

    # convert to dictionary inc the dictionary names 
    X = dict(zip(genomes, X))
    y = dict(zip(genomes, y ))

    return X,y


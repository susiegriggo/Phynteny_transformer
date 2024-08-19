
import torch
import pickle
import click
import re
import random 
import torch
import numpy as np 
import pandas as pd
from src import model_onehot
import os 

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


def process_data(plm_vectors, plm_integer, contig_details, max_genes=1000, extra_features = True, exclude_embedding=False):
    """
    frame as a supervised learning problem
    """
    # Organise info
    df = pd.DataFrame({'key': list(plm_vectors.keys()), 'genome': ['_'.join(re.split('_', i)[:-1]) for i in plm_vectors.keys()]})
    genome_genes = df.groupby('genome')['key'].apply(list).to_dict()
    # genomes = list(set(['_'.join(re.split('_', i)[:-1]) for i in plm_vectors.keys()]))
    genomes = [ re.split('pharokka_phrogs_genomes_', i)[1] for i in list(contig_details.keys())]
    
    X = list()
    y = list()
    removed = []

    for g in genomes:

        # get the genes in this genome
        this_genes = genome_genes.get(g)

        # get the corresponding vectors
        if exclude_embedding: 
            this_vectors = [[] for i in this_genes]
            
        else: 
            this_vectors = [plm_vectors.get(i) for i in this_genes]
        
        if len(this_vectors) > max_genes: 
            print('Number of genes in ' + g + ' is ' + str(len(this_vectors)))

        else: 

            # get the corresponding functions
            this_categories = [plm_integer.get(i) if plm_integer.get(i) is not None else -1 for i in this_genes]

            # handle if extra features are specified separateley
            if extra_features: 
                
                # get the strand information 
                genome_label  = 'pharokka_phrogs_genomes_' + g 
                #print(contig_details.get(genome_label))
                strand1, strand2  = encode_strand(contig_details.get(genome_label).get('sense'))

                # get the length of each gene 
                gene_lengths = encode_length(contig_details.get(genome_label).get('position'))

                # merge these columns into a numpy array 
                embedding = np.hstack((strand1, strand2, gene_lengths, np.array(this_vectors)))
   
            else: 
        
                # assemble array 
                embedding = np.array(this_vectors)
                
            # keep the information for the genomes that are not entirely unknown
            if all(element == -1 for element in this_categories):
                removed.append(g)
            else:
                # store the info in the dataset
                X.append(torch.tensor(np.array(embedding)))
                y.append(torch.tensor(np.array(this_categories)))
    
    print('Numer of genomes removed: ' + str(len(removed)))
    print('Numer of genomes kept: ' + str(len(X)))
    return X,y

@click.command()
@click.option("--x_path", "-x", help="File path to X training data")
@click.option("--y_path", "-y", help="File path to y training data")
@click.option('--shuffle', is_flag=True, default = False, help='Shuffle order of the genes. Helpful for determining if gene order increases predictive power')
@click.option('--unmask_unknowns', is_flag=True, default = False, help='Do not mask unknown gene functions from the model')
@click.option('--lr', default=1e-6, help='Learning rate for the optimizer.', type=float)
@click.option('--epochs', default=10, help='Number of training epochs.', type=int)
@click.option('--dropout', default=0.1, help='Dropout value for dropout layer.', type=int)
@click.option('--hidden_dim', default=512, help='Hidden dimension size for the transformer model.', type=int)
@click.option('--num_heads', default=4, help='Number of attention heads in the transformer model.', type=int)
@click.option('-o', '--out', default='train_out', help='Path to save the output.', type=click.STRING)
@click.option( "--device", default='cuda', help="specify cuda or cpu.", type=click.STRING)

def main(x_path, y_path, shuffle, lr, epochs, hidden_dim, num_heads, out, unmask_unknowns, dropout, device):
    
    # Input phrog info 
    phrog_integer = pickle.load(open('/home/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/integer_category.pkl', 'rb'))
    phrog_integer = dict(zip([i -1 for i in list(phrog_integer.keys())], [i for i in list(phrog_integer.values())])) # shuffling the keys down by one 

    # Create the output directory if it doesn't exist
    if not os.path.exists(out):
        os.makedirs(out)
        print(f"Directory created: {out}")
    else:
        print(f"Warning: Directory {out} already exists.")
    

    print('Reading in data', flush = True)
    X = pickle.load(open(x_path, 'rb'))
    y = pickle.load(open(y_path, 'rb'))

    
    # Shuffle if specified # TODO adjust this now that X and y are dictionaries 
    if shuffle: 
        print('Shuffling gene orders...', flush=True)
        for  key in list(X.keys()): 
            
            # generate indices for shuffling 
            indices = list(range(len(X.get(key))))
            random.shuffle(indices)
            X[key] = X[key][indices]

        print('\t Done shuffling gene orders', flush=True)
    else: 
        print('Gene orders not shuffled!', flush=True)


    # Produce the dataset object 
    train_dataset = model_onehot.VariableSeq2SeqEmbeddingDataset(list(X.values()),list(y.values()))
    train_dataset.set_training(True)
    print('Training model...', flush=True)

    if unmask_unknowns: 
        mask_unknowns = False
    else: 
        mask_unknowns = True
        
    model_onehot.train_crossValidation(train_dataset, phrog_integer, n_splits=10, batch_size=1, epochs=epochs, lr=lr, save_path=out, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout, device=device, mask_unknowns=mask_unknowns)

    # Evaluate the model
    #print('Evaluating the model', flush=True)
    #model_onehot.evaluate(transformer_model, test_dataloader)

    print('FINISHED! :D')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

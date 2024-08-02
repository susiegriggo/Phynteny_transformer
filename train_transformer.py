from sklearn.model_selection import train_test_split
import torch
import pickle
import click
import re
import random 
import torch
import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader
from src import model_masker 
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

    return strand_encode[0], strand_encode[1]

def encode_length(gene_positions): 
    """
    Extract the length of each of each gene so that it can be included in the embedding 

    :param gene_positions: list of start and end position of each gene 
    :return: length of each gene 
    """

    return [np.abs(i[1] - i[0]) for i in gene_positions]


def process_data(plm_vectors, plm_integer, contig_details, max_genes=1000):
    """
    frame as a supervised learning problem
    """
    # Organise info
    df = pd.DataFrame({'key': list(plm_vectors.keys()), 'genome': ['_'.join(re.split('_', i)[:-1]) for i in plm_vectors.keys()]})
    genome_genes = df.groupby('genome')['key'].apply(list).to_dict()
    genomes = list(set(['_'.join(re.split('_', i)[:-1]) for i in plm_vectors.keys()]))

    X = list()
    y = list()
    removed = []

    for g in genomes[:10000]:
        # get the genes in this genome
        this_genes = genome_genes.get(g)

        # get the corresponding vectors
        this_vectors = [plm_vectors.get(i) for i in this_genes]
        if len(this_vectors) > max_genes: 
            print('Number of genes in ' + g + ' is ' + str(len(this_vectors)))

        else: 

            # get the corresponding functions
            this_categories = [plm_integer.get(i) if plm_integer.get(i) is not None else -1 for i in this_genes]

            # get the strand information 
            strand1, strand2  = encode_strand(contig_details.get(g).get('sense'))

            # get the relative position of each gene in the genome 
            
            
            # get the length of each gene 
            gene_lengths = encode_length(contig_details.get(g).get('position'))

            # keep the information for the genomes that are not entirely unknown
            if all(element == -1 for element in this_categories):
                removed.append(g)
            else:
                # store the info in the dataset
                X.append(torch.tensor(np.array(this_vectors)))
                y.append(torch.tensor(np.array(this_categories)))
    
    return X,y

def process_data_with_category(plm_vectors, plm_integer, max_genes=1000):
    """
    frame as a supervised learning problem
    """
    # Organise info
    df = pd.DataFrame({'key': list(plm_vectors.keys()), 'genome': ['_'.join(re.split('_', i)[:-1]) for i in plm_vectors.keys()]})
    genome_genes = df.groupby('genome')['key'].apply(list).to_dict()
    genomes = list(set(['_'.join(re.split('_', i)[:-1]) for i in plm_vectors.keys()]))

    X = list()
    y = list()
    removed = []

    for g in genomes[:10000]:
        # get the genes in this genome
        this_genes = genome_genes.get(g)

        # get the corresponding vectors
        this_vectors = [plm_vectors.get(i) for i in this_genes]
        if len(this_vectors) > max_genes: 
            print('Number of genes in ' + g + ' is ' + str(len(this_vectors)))

        else: 

            # get the corresponding functions
            this_categories = [plm_integer.get(i) if plm_integer.get(i) is not None else -1 for i in this_genes]

            # keep the information for the genomes that are not entirely unknown
            if all(element == -1 for element in this_categories):
                removed.append(g)
            else:
                # store the info in the dataset
                this_X = np.array(this_vectors) 
                
                # Reshape the new column to be a 2D array with one column
                category_column = np.array(this_categories)[:, np.newaxis]

                # Append the new column as the first column
                this_X = np.hstack((category_column, this_X))
                
                X.append(torch.tensor(this_X))
                y.append(torch.tensor(np.array(this_categories)))
                
    
    return X,y

@click.command()
@click.option('--max_genes', default=1000, help='Maximum number of genes per genome included in training - COMING SOON', type=int)
@click.option('--shuffle', is_flag=True, default = False, help='Shuffle order of the genes. Helpful for determining if gene order increases predictive power')
@click.option('--inc_category', is_flag=True, default = False, help='Include category in the first column of the dataframe')
@click.option('--lr', default=1e-6, help='Learning rate for the optimizer.', type=float)
@click.option('--epochs', default=10, help='Number of training epochs.', type=int)
@click.option('--dropout', default=0.1, help='Dropout value for dropout layer.', type=int)
@click.option('--hidden_dim', default=512, help='Hidden dimension size for the transformer model.', type=int)
@click.option('--num_heads', default=4, help='Number of attention heads in the transformer model.', type=int)
@click.option('-o', '--out', default='train_out', help='Path to save the output.', type=click.STRING)
@click.option( "--device", default='cuda', help="specify cuda or cpu.", type=click.STRING)

def main(max_genes, shuffle, inc_category, lr, epochs, hidden_dim, num_heads, out, dropout, device):
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(out):
        os.makedirs(out)
        print(f"Directory created: {out}")
    else:
        print(f"Warning: Directory {out} already exists.")
    
    ###########################
    print('Reading in data', flush = True)

    # some scrappy code which needs fixing to get the plm vectors
    plm_vectors = pickle.load(open('/home/grig0076/scratch/glm_embeddings/LSTM_test_example/data/phrogs_genomes/pLM_embs.pkl', 'rb'))
    plm_ogg = pickle.load(open('/home/grig0076/scratch/glm_embeddings/LSTM_test_example/data/phrogs_genomes/ogg_assignment.pkl', 'rb'))
    contig_details = phynteny_dict = pickle.load(open('/home/grig0076/scratch/poznan_phynteny/phynteny_formatted/poznan_phynteny_all_data.pkl', 'rb')) 

    # read in annotation file
    phrogs = pd.read_csv('/home/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/phrog_annot_v4.tsv', sep='\t')
    category_dict = dict(zip(phrogs['phrog'], phrogs['category']))

    # read in integer encoding of the categories
    phrog_integer = pickle.load(open('/home/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/integer_category.pkl', 'rb'))
    phrog_integer = dict(zip([i -1 for i in list(phrog_integer.keys())], [i for i in list(phrog_integer.values())]))
    phrog_integer_reverse = dict(zip(list(phrog_integer.values()), list(phrog_integer.keys())))
    phrog_integer_reverse['unknown function'] = -1

    # plm integer
    plm_integer = dict(zip(list(plm_ogg.keys()), [phrog_integer_reverse.get(category_dict.get(int(i[0][6:]))) for i in plm_ogg.values()]))

    ##########################
    # Checks that the GPU is available 
    #print(torch.cuda.is_available())  # Should return True if GPU is available
    #print(torch.cuda.current_device())  # Should return the index of the current GPU device
    #print(torch.cuda.device_count())  # Should return the number of GPU devices
    #print(torch.cuda.get_device_name(0))  # Should return the name of the GPU device

    ##############################

    # Train and test split
    #print('Test and train split', flush=True)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    #print('Size of training data: ' + str(len(X_train)) + ' genomes')
    #print('Size of testing data: ' + str(len(X_test)) + ' genomes')

    # Generate datasets
    print('Building dataset', flush = True)
    if inc_category: 
        X, y = process_data_with_category(plm_vectors, plm_integer, max_genes=1000) 
    else: 
        X, y = process_data(plm_vectors, plm_integer, contig_details,max_genes=1000)

    # Shuffle if specified 
    if shuffle: 
        print('Shuffling gene orders...', flush=True)
        for i in range(len(X)): 
            
            # generate indices for shuffling 
            indices = list(range(len(X[i])))
            random.shuffle(indices)

            # Use shuffled indices to reorder X and y 
            X[i] = X[i][indices]
            y[i] = y[i][indices]

        print('\t Done shuffling gene orders', flush=True)
    else: 
        print('Gene orders not shuffled!', flush=True)

    if inc_category: 
        train_dataset = model_masker.VariableSeq2SeqEmbeddingDataset(X,y) 
        train_dataset.set_training(True) # this step allows us to mask 
        print('Training model...', flush=True)
        model_masker.train_crossValidation(train_dataset, phrog_integer, n_splits=10, batch_size=1, epochs=epochs, lr=lr, save_path=out, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout, device=device)
        
    else: 
        # Produce the dataset object 
        train_dataset = model_onehot.VariableSeq2SeqEmbeddingDataset(X,y)
        train_dataset.set_training(True)
        print('Training model...', flush=True)
        model_onehot.train_crossValidation(train_dataset, phrog_integer, n_splits=10, batch_size=1, epochs=epochs, lr=lr, save_path=out, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout, device=device)


    #test_dataset = model.VariableSeq2SeqEmbeddingDataset(X_test, y_test)

    # Return the size of the test dataset to try and find 
    #print(np.max([len(i) for i in X_train]))
    #print(np.min([len(i) for i in X_train]))
    #print(np.max([len(i) for i in y_train]))
    #print(np.min([len(i) for i in y_train]))


    # Create dataloader objects
    #print('Creating dataloader objects', flush=True)
    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=model.collate_fn)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=model.collate_fn)

    # Create model
    #print('Creating model', flush=True)
    #num_classes = len(phrog_integer)
    #transformer_model = model.VariableSeq2SeqTransformerClassifier(input_dim=1280, num_classes=num_classes, num_heads=num_heads, hidden_dim=hidden_dim)

    # Train the model
    
    #model.train(transformer_model, train_dataloader, test_dataloader, epochs=epochs, lr=lr, save_path=out)
    
    # Train the model wqith kfold cross validataion 
    # use batch size = 1 as it makes results easier to analyse with respsect to genomes of different sizes 
    

    # Evaluate the model
    #model.evaluate_with_metrics_and_save(transformer_model, test_dataloader, threshold=0.5, output_dir='metrics_output')
    #model.evaluate_with_optimal_thresholds(transformer_model, test_dataloader,phrog_integer, output_dir=out)

    # Evaluate the model
    print('Evaluating the model', flush=True)
    #model.evaluate(transformer_model, test_dataloader)

    print('FINISHED! :D')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

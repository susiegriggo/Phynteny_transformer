import torch
import pickle
import click
import re
import torch
import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src import model

def split(plm_vectors, plm_integer, test_size=0.2):
    """
    :return: X_train, X_test, y_train, y_test
    """
    # Organise info
    df = pd.DataFrame({'key': list(plm_vectors.keys()), 'genome': [re.split('_', i)[0] for i in plm_vectors.keys()]})
    genome_genes = df.groupby('genome')['key'].apply(list).to_dict()
    genomes = list(set([re.split('_', i)[0] for i in plm_vectors.keys()]))

    X = list()
    y = list()
    removed = []

    for g in genomes[:2000]:
        # get the genes in this genome
        this_genes = genome_genes.get(g)

        # get the corresponding vectors
        this_vectors = [plm_vectors.get(i) for i in this_genes]

        # get the corresponding functions
        this_categories = [plm_integer.get(i) if plm_integer.get(i) is not None else -1 for i in this_genes]

        # keep the information for the genomes that are not entirely unknown
        if all(element == -1 for element in this_categories):
            removed.append(g)
        else:
            # store the info in the dataset
            X.append(torch.tensor(np.array(this_vectors)))
            y.append(torch.tensor(np.array(this_categories)))

    return train_test_split(X, y, test_size=test_size, random_state=42)

@click.command()
@click.option('--batch_size', default=16, help='Batch size for training.', type=int)
@click.option('--lr', default=1e-4, help='Learning rate for the optimizer.', type=float)
@click.option('--epochs', default=5, help='Number of training epochs.', type=int)
@click.option('--hidden_dim', default=512, help='Hidden dimension size for the transformer model.', type=int)
@click.option('--num_heads', default=4, help='Number of attention heads in the transformer model.', type=int)
def main(batch_size, lr, epochs, hidden_dim, num_heads):
    ###########################
    # some scrappy code which needs fixing to get the plm vectors
    plm_vectors = pickle.load(open('/home/grig0076/scratch/glm_embeddings/LSTM_test_example/data/phrogs_genomes/pLM_embs.pkl', 'rb'))
    plm_ogg = pickle.load(open('/home/grig0076/scratch/glm_embeddings/LSTM_test_example/data/phrogs_genomes/ogg_assignment.pkl', 'rb'))

    # read in annotation file
    phrogs = pd.read_csv('/home/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/phrog_annot_v4.tsv', sep='\t')
    category_dict = dict(zip(phrogs['phrog'], phrogs['category']))

    # read in integer encoding of the categories
    phrog_integer = pickle.load(open('/home/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/integer_category.pkl', 'rb'))
    phrog_integer_reverse = dict(zip(list(phrog_integer.values()), list(phrog_integer.keys())))
    phrog_integer_reverse['unknown function'] = -1

    # plm integer
    plm_integer = dict(zip(list(plm_ogg.keys()), [phrog_integer_reverse.get(category_dict.get(int(i[0][6:]))) for i in plm_ogg.values()]))

    ##########################
    # Train and test split
    X_train, X_test, y_train, y_test = split(plm_vectors, plm_integer)

    # Generate datasets
    train_dataset = model.VariableSeq2SeqEmbeddingDataset(X_train, y_train)
    test_dataset = model.VariableSeq2SeqEmbeddingDataset(X_test, y_test)

    # Create dataloader objects
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=model.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=model.collate_fn)

    # Create model
    transformer_model = model.VariableSeq2SeqTransformerClassifier(input_dim=1280, num_classes=10, num_heads=num_heads, hidden_dim=hidden_dim)

    # Train the model
    model.train(transformer_model, train_dataloader, epochs=epochs, lr=lr)

    # Evaluate the model
    model.evaluate_with_threshold(transformer_model, test_dataloader, threshold=0.8)

    # Save the model
    # TODO update the part that has been added 

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

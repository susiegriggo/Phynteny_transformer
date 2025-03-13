import click
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from src import model_onehot
from loguru import logger

@click.command()
@click.option('--pharokka_x_path', required=True, type=click.Path(exists=True), help='Path to pharokka X data.')
@click.option('--pharokka_y_path', required=True, type=click.Path(exists=True), help='Path to pharokka y data.')
@click.option('--phold_y_path', required=True, type=click.Path(exists=True), help='Path to phold y data.')
@click.option('--model_dir', required=True, type=click.Path(exists=True), help='Directory containing the models.')
@click.option('--output_dir', required=True, type=click.Path(), help='Directory to save the ROC data.')
def main(pharokka_x_path, pharokka_y_path, phold_y_path, model_dir, output_dir):
    
    # Load data
    logger.info(f"Parameters: pharokka_x_path={pharokka_x_path}, pharokka_y_path={pharokka_y_path}, phold_y_path={phold_y_path}, model_dir={model_dir}, output_dir={output_dir}")
    pharokka_y = pickle.load(open(pharokka_y_path, 'rb'))
    pharokka_X = pickle.load(open(pharokka_x_path, 'rb'))
    phold_y = pickle.load(open(phold_y_path, 'rb'))

    num_classes = 9
    device = 'cpu'
    #phrog_categories = [f"Category {i}" for i in range(num_classes)]

    # Initialize dictionaries to store precision, recall, average precision, false positive rate, true positive rate, and ROC AUC
    precision_dict = {i: [] for i in range(num_classes)}
    recall_dict = {i: [] for i in range(num_classes)}
    average_precision_dict = {i: [] for i in range(num_classes)}
    fpr_dict = {i: [] for i in range(num_classes)}
    tpr_dict = {i: [] for i in range(num_classes)}
    roc_auc_dict = {i: [] for i in range(num_classes)}

    for k in range(10):
        logger.info(f"Processing fold {k+1}...")

        # Get validation data
        val_labels = [line.strip() for line in open(f'{model_dir}/val_kfold_keys/val_kfold_labels_fold{k+1}.txt').readlines()]
        
        # Remove labels that are not in phold_y
        val_labels = [v for v in val_labels if phold_y.get(v) != None]
        
        # Get embeddings and categories
        validation_embeddings = [pharokka_X.get(v) for v in val_labels]
        validation_categories = [pharokka_y.get(v) for v in val_labels]
        validation_phold = [phold_y.get(v) for v in val_labels]

        # Create validation dataset
        validation_dataset = model_onehot.EmbeddingDataset(validation_embeddings, validation_categories, mask_portion=0, labels=val_labels)
        validation_dataset.set_validation(validation_phold)
        validation_loader = DataLoader(validation_dataset, batch_size=64, collate_fn=model_onehot.collate_fn)

        # Load model
        m = model_onehot.TransformerClassifierCircularRelativeAttention(input_dim=1280, num_classes=num_classes, num_heads=4, hidden_dim=256, dropout=0.1, use_lstm=True)
        m.load_state_dict(torch.load(f'{model_dir}/fold_{k+1}transformer.model', map_location=torch.device('cpu')))
        m.eval()
        logger.info("Model and validation data read")

        # Get predictions
        all_probs = []
        all_categories = []

        logger.info(f"Number of batches {len(validation_loader)}")
        batch_count = 0

    
        for embeddings, categories, masks, idx in validation_loader:
            
            batch_size = embeddings.shape[0]
            
            # Move data to device
            embeddings, categories, masks = (
                embeddings.to(device).float(),
                categories.to(device).long(),
                masks.to(device).float(),
            )
            src_key_padding_mask = (masks != -2)
            outputs = m(embeddings, src_key_padding_mask=src_key_padding_mask.to(device))
            probs = F.softmax(outputs, dim=2)
            
            # Get predictions
            for i in range(batch_size):
                all_probs.extend(probs[i][idx[i]].tolist())
                all_categories.extend(categories[i][idx[i]].tolist())

            batch_count += 1
            if batch_count % 100 == 0:
                logger.info(f"...processing batch {batch_count}")

        all_probs = np.array(all_probs)
        all_categories = np.array(all_categories)
        y_true = label_binarize(all_categories, classes=np.arange(num_classes))

        # Compute precision-recall data
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true[:, i], all_probs[:, i])
            avg_precision = average_precision_score(y_true[:, i], all_probs[:, i])
            precision_dict[i].append(np.interp(np.linspace(0, 1, 100), recall[::-1], precision[::-1]))
            recall_dict[i].append(recall)
            average_precision_dict[i].append(avg_precision)

        # Compute ROC data
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            fpr_dict[i].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            tpr_dict[i].append(tpr)
            roc_auc_dict[i].append(roc_auc)

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_dict = {}
    mean_auc_dict = {}
    std_tpr_dict = {}
    mean_precision_dict = {}
    mean_avg_precision_dict = {}
    std_precision_dict = {}

    for i in range(num_classes):
        mean_tpr_dict[i] = np.mean(fpr_dict[i], axis=0)
        std_tpr_dict[i] = np.std(fpr_dict[i], axis=0)
        mean_auc_dict[i] = np.mean(roc_auc_dict[i])
        mean_precision_dict[i] = np.mean(precision_dict[i], axis=0)
        std_precision_dict[i] = np.std(precision_dict[i], axis=0)
        mean_avg_precision_dict[i] = np.mean(average_precision_dict[i])

    with open(f'{output_dir}/precision_dict.pkl', 'wb') as f:
        pickle.dump(precision_dict, f)
    with open(f'{output_dir}/recall_dict.pkl', 'wb') as f:
        pickle.dump(recall_dict, f)
    with open(f'{output_dir}/average_precision_dict.pkl', 'wb') as f:
        pickle.dump(average_precision_dict, f)
    with open(f'{output_dir}/fpr_dict.pkl', 'wb') as f:
        pickle.dump(fpr_dict, f)
    with open(f'{output_dir}/tpr_dict.pkl', 'wb') as f:
        pickle.dump(tpr_dict, f)
    with open(f'{output_dir}/roc_auc_dict.pkl', 'wb') as f:
        pickle.dump(roc_auc_dict, f)
    with open(f'{output_dir}/std_tpr_dict.pkl', 'wb') as f:
        pickle.dump(std_tpr_dict, f)
    with open(f'{output_dir}/mean_precision_dict.pkl', 'wb') as f:
        pickle.dump(mean_precision_dict, f)
    with open(f'{output_dir}/std_precision_dict.pkl', 'wb') as f:
        pickle.dump(std_precision_dict, f)
    with open(f'{output_dir}/mean_avg_precision_dict.pkl', 'wb') as f:
        pickle.dump(mean_avg_precision_dict, f)

    logger.info("Precision-Recall and ROC data saved successfully.")

if __name__ == '__main__':
    main()
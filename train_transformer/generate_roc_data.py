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
import os
import shutil

@click.command()
@click.option('--pharokka_x_path', required=True, type=click.Path(exists=True), help='Path to pharokka X data.')
@click.option('--pharokka_y_path', required=True, type=click.Path(exists=True), help='Path to pharokka y data.')
@click.option('--phold_y_path', required=True, type=click.Path(exists=True), help='Path to phold y data.')
@click.option('--model_dir', required=True, type=click.Path(exists=True), help='Directory containing the models.')
@click.option('--output_dir', required=True, type=click.Path(), help='Directory to save the ROC data.')
@click.option('--force', is_flag=True, help='Force overwrite the output directory if it exists.')
# Add model parameter options
@click.option('--input_dim', default=1280, help='Input dimension for the model.')
@click.option('--hidden_dim', default=256, help='Hidden dimension for the model.')
@click.option('--lstm_hidden_dim', default=512, help='LSTM hidden dimension for the model.')
@click.option('--num_heads', default=4, help='Number of attention heads for the model.')
@click.option('--num_layers', default=2, help='Number of transformer layers for the model.')
@click.option('--dropout', default=0.1, help='Dropout rate for the model.')
@click.option('--use_lstm/--no_lstm', default=True, help='Whether to use LSTM in the model.')
@click.option('--positional_encoding_type', type=click.Choice(['fourier', 'sinusoidal']), default='fourier', help='Type of positional encoding to use.')
@click.option('--pre_norm', is_flag=True, default=False, help='Use pre-normalization in transformer layers.')
@click.option('--protein_dropout_rate', default=0.0, help='Dropout rate for protein features.')
@click.option('--progressive_dropout', is_flag=True, default=False, help='Enable progressive dropout for protein features.')
@click.option('--initial_dropout_rate', default=1.0, help='Initial dropout rate when using progressive dropout.')
@click.option('--final_dropout_rate', default=0.4, help='Final dropout rate when using progressive dropout.')
@click.option('--max_len', default=1500, help='Maximum sequence length.')
@click.option('--output_dim', default=None, type=int, help='Output dimension for the model. Defaults to num_classes if not specified.')
def main(pharokka_x_path, pharokka_y_path, phold_y_path, model_dir, output_dir, force, 
         input_dim, hidden_dim, lstm_hidden_dim, num_heads, dropout, use_lstm,
         positional_encoding_type, pre_norm, protein_dropout_rate, progressive_dropout,
         initial_dropout_rate, final_dropout_rate, max_len, output_dim, num_layers):
    # Create output directory if it does not exist, or clear it if force is specified
    if os.path.exists(output_dir):
        if force:
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            raise FileExistsError(f"Output directory {output_dir} already exists. Use --force to overwrite.")
    else:
        os.makedirs(output_dir)
    
    # Load data with error handling
    try:
        logger.info(f"Parameters: pharokka_x_path={pharokka_x_path}, pharokka_y_path={pharokka_y_path}, phold_y_path={phold_y_path}, model_dir={model_dir}, output_dir={output_dir}")
        pharokka_y = pickle.load(open(pharokka_y_path, 'rb'))
        logger.info(f"Number of pharokka_y samples: {len(pharokka_y)}")
        pharokka_X = pickle.load(open(pharokka_x_path, 'rb'))
        logger.info(f"Number of pharokka_X samples: {len(pharokka_X)}")
        phold_y = pickle.load(open(phold_y_path, 'rb'))
        logger.info(f"Number of phold_y samples: {len(phold_y)}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    num_classes = 9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Log model parameters
    logger.info(f"Model parameters: input_dim={input_dim}, hidden_dim={hidden_dim}, lstm_hidden_dim={lstm_hidden_dim}, "
                f"num_heads={num_heads}, num_layers={num_layers}, dropout={dropout}, use_lstm={use_lstm}, positional_encoding_type={positional_encoding_type}, "
                f"pre_norm={pre_norm}, protein_dropout_rate={protein_dropout_rate}, progressive_dropout={progressive_dropout}, "
                f"initial_dropout_rate={initial_dropout_rate}, final_dropout_rate={final_dropout_rate}, max_len={max_len}, "
                f"output_dim={output_dim or num_classes}")

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
        logger.info(f'Number of validation samples: {len(validation_phold)}')

        # Create validation dataset
        validation_dataset = model_onehot.EmbeddingDataset(validation_embeddings, validation_categories, mask_portion=0, labels=val_labels)
        validation_dataset.set_validation(validation_phold)
        validation_loader = DataLoader(validation_dataset, batch_size=64, collate_fn=model_onehot.collate_fn)

        # Select positional encoding function
        positional_encoding_func = model_onehot.fourier_positional_encoding if positional_encoding_type == 'fourier' else model_onehot.sinusoidal_positional_encoding
        
        # Load model with all parameters
        m = model_onehot.TransformerClassifierCircularRelativeAttention(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            dropout=dropout,
            max_len=max_len,
            use_lstm=use_lstm,
            positional_encoding=positional_encoding_func,
            protein_dropout_rate=protein_dropout_rate,
            pre_norm=pre_norm,
            progressive_dropout=progressive_dropout,
            initial_dropout_rate=initial_dropout_rate,
            final_dropout_rate=final_dropout_rate,
            output_dim=output_dim or num_classes
        )
        
        # Create dummy forward pass to initialize buffers if necessary
        if hasattr(m, 'protein_feature_dropout'):
            logger.info("Initializing protein_feature_dropout buffers with dummy forward pass")
            batch_size, seq_len = 2, 10
            feature_dim = m.gene_feature_dim + (hidden_dim - m.gene_feature_dim)
            dummy_input = torch.zeros((batch_size, seq_len, feature_dim), device=device)
            dummy_idx = [torch.tensor([0, 1]) for _ in range(batch_size)]
            m.protein_feature_dropout(dummy_input, dummy_idx)
            
        # Load state dict with strict=False to handle parameter differences
        #state_dict = torch.load(f'{model_dir}/fold_{k+1}/transformer_state_dict.pth', map_location=torch.device(device))
        state_dict = torch.load(f'{model_dir}/fold_{k+1}transformer.model', map_location=device)
        m.load_state_dict(state_dict, strict=False)
        m.to(device)
        m.eval()
        logger.info(f"Model loaded from {model_dir}/fold_{k+1}transformer.model")

        # Get predictions
        all_probs = []
        all_categories = []

        logger.info(f"Number of batches {len(validation_loader)}")
        batch_count = 0

        # Initialize a dictionary to count predictions for each category
        category_counts = {i: 0 for i in range(num_classes)}
        # Initialize a dictionary to count occurrences of each label in the predictions
        label_counts = {i: 0 for i in range(num_classes)}

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
                # Update category counts
                for cat in categories[i][idx[i]].tolist():
                    category_counts[cat] += 1
                # Update label counts
                for prob in probs[i][idx[i]].tolist():
                    label_counts[np.argmax(prob)] += 1

            batch_count += 1
            if batch_count % 50 == 0:
                logger.info(f"...processing batch {batch_count}")

                # Log the number of predictions made for each category
                logger.info(f"Category counts: {category_counts}")

        logger.info(f"Label counts in predictions: {label_counts}")

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
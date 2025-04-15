import pickle
import click
import random
import os
from loguru import logger
os.chdir('/home/grig0076/susie_scratch/GitHubs/latest/Phynteny_transformer')
from src import model_onehot
from src.model_onehot import fourier_positional_encoding  # Add import for Fourier positional encoding
import torch

def setup_output_directory(out, force):
    try:
        if not os.path.exists(out):
            os.makedirs(out)
            print(f"Directory created: {out}")
        else:
            if not force:
                print(f"Warning: Directory {out} already exists.")
                raise Exception(f"Directory {out} already exists.")
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        raise

def load_data(x_path, y_path):
    try:
        logger.info("Reading in data")
        X = pickle.load(open(x_path, "rb"))
        y = pickle.load(open(y_path, "rb"))

        # Compute input size of embeddings
        input_size = list(X.values())[0].shape[1] - 3  # 3 is the number of strand info and gene length
        logger.info(f"Computed input size of embeddings: {input_size}")

        return X, y, input_size, list(X.keys())  # Return keys as labels
    except Exception as e:
        logger.error(f"Error reading input files: {e}")
        raise

def validate_num_heads(ctx, param, value):
    if value % 4 != 0:
        raise click.BadParameter("Number of attention heads must be divisible by 4.")
    return value


def test_model(out, params, fold=None):  # Default fold is None
    """
    Load the trained model and test it on the validation data.

    Parameters:
    out (str): Output directory where the model and validation data are saved.
    params (dict): Dictionary of model parameters.
    fold (str): The fold to test (e.g., "fold_1"). If None, defaults to "fold_1".
    """
    try:
        # Determine the fold directory
        fold_dir = f"fold_{fold}" if fold is not None else "fold_1"

        # Log the parameters used to load the model
        logger.info("Model loading parameters:")
        for key, value in params.items():
            logger.info(f"{key}: {value}")

        # Load the model
        model_path = os.path.join(out, fold_dir, "transformer_state_dict.pth")
        model = load_model(model_path, params)
        logger.info(f"Model loaded successfully from {model_path}.")

        # Load the validation data
        val_loader_path = os.path.join(out, fold_dir, "val_kfold_loader.pkl")
        with open(val_loader_path, "rb") as f:
            val_loader = pickle.load(f)
        logger.info(f"Validation data loaded successfully from {val_loader_path}.")

        # Initialize counters for predictions, correct predictions, and counts
        category_counts_masked = torch.zeros(params["num_classes"], dtype=torch.int)
        category_counts_predicted = torch.zeros(params["num_classes"], dtype=torch.int)
        category_counts_correct = torch.zeros(params["num_classes"], dtype=torch.int)  # Renamed from category_counts_true

        # Evaluate the model
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for embeddings, categories, masks, idx in val_loader:
                embeddings = embeddings.to(device).float()
                categories = categories.to(device).long()
                masks = masks.to(device).float()
                src_key_padding_mask = (masks != -2).bool().to(device)

                outputs = model(embeddings, idx=idx, src_key_padding_mask=src_key_padding_mask)
                predictions = outputs.argmax(dim=-1)

                # Evaluate only rows with idx
                for batch_idx, indices in enumerate(idx):
                    for i in indices:
                        pred = predictions[batch_idx, i].item()
                        true_label = categories[batch_idx, i].item()
                        if pred != -1:  # Ignore padding or invalid predictions
                            if pred == true_label:
                                category_counts_correct[true_label] += 1  # Updated to category_counts_correct
                            category_counts_predicted[pred] += 1  # Count predictions
                            category_counts_masked[true_label] += 1  # Count masked categories


        logger.info("Accuracy per category:")
        for i in range(params["num_classes"]):
            if category_counts_masked[i] > 0:
                accuracy = category_counts_correct[i].item() / category_counts_masked[i].item()
                logger.info(f"Category {i}: {accuracy:.2f}")
            else:
                logger.info(f"Category {i}: No predictions made")

        logger.info("Masked category counts (true labels):")
        for i in range(params["num_classes"]):
            logger.info(f"Category {i}: {category_counts_masked[i].item()} masked")

        logger.info("Prediction counts:")
        for i in range(params["num_classes"]):
            logger.info(f"Category {i}: {category_counts_predicted[i].item()} predicted")

        logger.info("True label counts:")
        for i in range(params["num_classes"]):
            logger.info(f"Category {i}: {category_counts_correct[i].item()} correct")  # Updated to category_counts_correct

    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        raise

def load_model(model_path, params):
    """
    Load the trained model from the specified path.

    Parameters:
    model_path (str): Path to the saved model file.
    params (dict): Dictionary of model parameters.

    Returns:
    nn.Module: Loaded model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    attention = params["attention"]
    input_dim = params["input_size"]
    num_classes = params["num_classes"]
    num_heads = params["num_heads"]
    hidden_dim = params["hidden_dim"]
    lstm_hidden_dim = params["lstm_hidden_dim"]
    dropout = params["dropout"]
    num_layers = params["num_layers"]
    output_dim = params["output_dim"]
    use_lstm = params["use_lstm"]
    use_positional_encoding = params["use_positional_encoding"]
    protein_dropout_rate = params.get("protein_dropout_rate", 0.0)  # Get with default value
    

    # Initialize the model
    if attention == "absolute":
        model = model_onehot.TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            output_dim=output_dim,
            use_lstm=use_lstm,
            use_positional_encoding=use_positional_encoding
        )
    elif attention == "relative":
        model = model_onehot.TransformerClassifierRelativeAttention(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            output_dim=output_dim,
            use_lstm=use_lstm,
            use_positional_encoding=use_positional_encoding
        )
    elif attention == "circular":
        model = model_onehot.TransformerClassifierCircularRelativeAttention(
            input_dim=input_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            output_dim=output_dim,
            use_lstm=use_lstm,
            use_positional_encoding=use_positional_encoding,
            protein_dropout_rate=protein_dropout_rate  # Add this parameter
        )
    else:
        raise ValueError(f"Invalid attention type: {attention}")
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)  # Load the state_dict into the model
    model.to(device)
    model.eval()
    return model

@click.command()
@click.option("--x_path", "-x", help="File path to X training data")
@click.option("--y_path", "-y", help="File path to y training data")
@click.option(
    "--mask_portion", default=0.15, help="Portion of knowns to mask during training"
)
@click.option(
    "--attention",
    type=click.Choice(["absolute", "relative", "circular"]),
    default="circular",
    help="Choose a type of positional embedding for the model.",
)
@click.option(
    "--shuffle",
    is_flag=True,
    default=False,
    help="Shuffle order of the genes. Helpful for determining if gene order increases predictive power",
)
@click.option("--lr", default=1e-6, help="Learning rate for the optimizer.", type=float)
@click.option("--min_lr_ratio", default=0.1, help="Minimum learning rate ratio for the cosine scheduler.", type=float)
@click.option("--epochs", default=15, help="Number of training epochs.", type=int)
@click.option(
    "--dropout", default=0.1, help="Dropout value for dropout layer.", type=float
)
@click.option(
    "--hidden_dim",
    default=512,
    help="Hidden dimension size for the transformer model.",
    type=int,
)
@click.option(
    "--batch_size",
    default=16,  # Revert batch size to 16
    help="Batch size used to train the model.",
    type=int,
)
@click.option(
    "--num_heads",
    default=4,
    help="Number of attention heads in the transformer model.",
    type=int,
    callback=validate_num_heads,
)
@click.option(
    "-o",
    "--out",
    default="train_out",
    help="Path to save the output.",
    type=click.STRING,
)
@click.option(
    "--intialisation",
    default="random",
    help="Specify whether to initialise attention mechanism with zeros or random values.",
    type=click.Choice(['random', 'zeros'], case_sensitive=False),
)
@click.option(
    "--lambda_penalty",
    default=0.1,
    help="Specify the lambda penalty for the diagonal loss function. Set to 0 to disable diagonal loss.",
    type=float,
)
@click.option(
    "--parallel_kfolds",
    is_flag=True,
    default=False,
    help="Distribute kfolds across available GPUs or train sequentially.",
)
@click.option(
    "--device", default="cuda", help="specify cuda or cpu.", type=click.STRING
)
@click.option(
    "--num_layers",
    default=2,
    help="Number of transformer layers.",
    type=int,
)
@click.option(
    "--checkpoint_interval",
    default=20,  
    help="Number of epochs between saving checkpoints.",
    type=int,
)
@click.option(
    "--fold_index",
    default=None,
    help="Specify a single fold index to train.",
    type=int,
)
@click.option(
    "--output_dim",
    default=None,
    help="Specify the output dimension for the model.",
    type=int,
)
@click.option(
    "--lstm_hidden_dim",
    default=512,
    help="Hidden dimension size for the LSTM layer.",
    type=int,
)
@click.option(
    "-f", "--force", is_flag=True, help="Overwrite output directory if it exists."
)
@click.option(
    "--use_lstm",
    is_flag=True,
    default=False,
    help="Include LSTM layers in the model.",
)
@click.option(
    "--use_positional_encoding",
    is_flag=True,
    default=True,
    help="Include positional encoding in the model.",
)
@click.option(
    "--noise_std",
    default=0.0,
    help="Standard deviation of the Gaussian noise to add to the embeddings.",
    type=float,
)
@click.option(
    "--zero_idx",
    is_flag=True,
    default=False,
    help="Set the embeddings of the masked tokens to zeros.",
)
@click.option(
    "--ignore_strand_gene_length",
    is_flag=True,
    default=False,
    help="Ignore the strand and gene length features in the embeddings.",
)
@click.option(
    "--run_test_model",  # Update the option name
    is_flag=True,
    default=False,
    help="Test the trained model on the validation data.",
)
@click.option(
    "--protein_dropout_rate",
    default=0.0,
    help="Dropout rate for protein features (only applied to protein embeddings).",
    type=float,
)
@click.option(
    "--pre_norm",
    is_flag=True,
    default=False,
    help="Use pre-normalization in transformer layers instead of post-normalization."
)
def main(
    x_path,
    y_path,
    mask_portion,
    attention,
    shuffle,
    lr,
    min_lr_ratio,
    epochs,
    hidden_dim,
    num_heads,
    batch_size, 
    out,
    dropout,
    device,
    intialisation, 
    lambda_penalty,
    parallel_kfolds,
    num_layers,
    fold_index,
    checkpoint_interval,
    output_dim,  # Add output_dim parameter
    lstm_hidden_dim,  # Add lstm_hidden_dim parameter
    force,  # Add force parameter
    use_lstm,  # Add use_lstm parameter
    use_positional_encoding,  # Add use_positional_encoding parameter
    noise_std,  # Add noise_std parameter
    zero_idx,  # Add zero_idx parameter
    ignore_strand_gene_length,  # Add ignore_strand_gene_length parameter
    run_test_model,  # Use the updated parameter name
    protein_dropout_rate,  # Add protein_dropout_rate parameter
    pre_norm,  # Add pre_norm parameter
):
    setup_output_directory(out, force)

    # Generate the script with training parameters
    params = {
        "attention": attention,
        "input_size": None,  # Will be set after loading data
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "lstm_hidden_dim": lstm_hidden_dim,
        "dropout": dropout,
        "num_layers": num_layers,
        "output_dim": output_dim,
        "use_lstm": use_lstm,
        "use_positional_encoding": use_positional_encoding,
        "protein_dropout_rate": protein_dropout_rate,  # Add this parameter to params dictionary
        "num_classes": 9,  # Hardcoded for now
        "pre_norm": pre_norm  # Add pre_norm to params
    }

    # generate loguru object
    logger.add(out + "/trainer.log", level="DEBUG")

    # Log parameter values
    logger.info(f"Parameters: x_path={x_path}, y_path={y_path}, mask_portion={mask_portion}, attention={attention}, shuffle={shuffle}, lr={lr}, min_lr_ratio={min_lr_ratio}, epochs={epochs}, hidden_dim={hidden_dim}, num_heads={num_heads}, batch_size={batch_size}, out={out}, dropout={dropout}, device={device}, intialisation={intialisation}, lambda_penalty={lambda_penalty}, parallel_kfolds={parallel_kfolds}, num_layers={num_layers}, fold_index={fold_index}, output_dim={output_dim}, lstm_hidden_dim={lstm_hidden_dim}, use_lstm={use_lstm}, use_positional_encoding={use_positional_encoding}, noise_std={noise_std}, zero_idx={zero_idx}, ignore_strand_gene_length={ignore_strand_gene_length}, protein_dropout_rate={protein_dropout_rate}, pre_norm={pre_norm}")  # Log use_lstm

    X, y, input_size, labels = load_data(x_path, y_path)  # Get labels
    params["input_size"] = input_size  # Set input size in params

    shuffled_data = {}  # Dictionary to save shuffled data

    # Shuffle if specified
    if shuffle:
        logger.info("Shuffling gene orders...")
        for key in list(X.keys()):
            # generate indices for shuffling
            indices = list(range(len(X.get(key))))
            random.Random(4).shuffle(indices)  # shuffle with a random seed of 4
            X[key] = X[key][indices]
            y[key] = y[key][indices]
        shuffled_data['X'] = X  # Save shuffled X
        shuffled_data['y'] = y  # Save shuffled y
        logger.info("\t Done shuffling gene orders")
        # Save shuffled data to a file
        with open(os.path.join(out, "shuffled_data.pkl"), "wb") as f:
            pickle.dump(shuffled_data, f)
    else:
        logger.info("Gene orders not shuffled!")

    # Produce the dataset object
    train_dataset = model_onehot.EmbeddingDataset(
        list(X.values()), list(y.values()), list(y.keys()), mask_portion=mask_portion, noise_std=noise_std, zero_idx=zero_idx, strand_gene_length=not ignore_strand_gene_length  # Pass ignore_strand_gene_length
    )
    train_dataset.set_training(True)
    logger.info(f"Total dataset size: {len(train_dataset)} samples")

    # Train the model
    logger.info("\nTraining model...")

    try:
        model_onehot.train_crossValidation(
            train_dataset,
            attention,
            n_splits=10,
            batch_size=batch_size,  # have changed this batch size to 16 
            epochs=epochs,
            lr=lr,
            save_path=out,
            num_heads=num_heads,
            min_lr_ratio=min_lr_ratio,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device,
            intialisation=intialisation, 
            lambda_penalty=lambda_penalty,
            parallel_kfolds=parallel_kfolds,
            checkpoint_interval=checkpoint_interval,
            num_layers=num_layers,
            single_fold=fold_index,
            output_dim=output_dim,  # Pass output_dim
            input_size=input_size,  # Pass input_size
            lstm_hidden_dim=lstm_hidden_dim,  # Pass lstm_hidden_dim
            use_lstm=use_lstm,  # Pass use_lstm
            positional_encoding=fourier_positional_encoding,  # Use Fourier positional encoding
            use_positional_encoding=use_positional_encoding,  # Pass use_positional_encoding
            noise_std=noise_std,  # Pass noise_std
            zero_idx=zero_idx,  # Pass zero_idx
            strand_gene_length=not ignore_strand_gene_length,  # Pass ignore_strand_gene_length
            protein_dropout_rate=protein_dropout_rate,  # Add this parameter
            pre_norm=pre_norm  # Add pre_norm parameter
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

    if run_test_model:  # Use the updated parameter name

        test_model(out, params, fold=fold_index)  # Pass fold_index to test_model

    logger.info("FINISHED! :D")

if __name__ == "__main__":
    main()

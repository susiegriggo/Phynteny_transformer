import os
import pickle
import torch
from loguru import logger
from src.model_onehot import TransformerClassifier, TransformerClassifierRelativeAttention, TransformerClassifierCircularRelativeAttention
from tqdm import tqdm

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

    if attention == "absolute":
        model = TransformerClassifier(
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
        model = TransformerClassifierRelativeAttention(
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
        model = TransformerClassifierCircularRelativeAttention(
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
    else:
        raise ValueError(f"Invalid attention type: {attention}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def test_model(model_path, val_loader_path, params):
    """
    Load the trained model and test it on the validation data.

    Parameters:
    model_path (str): Path to the trained model file.
    val_loader_path (str): Path to the validation data loader.
    params (dict): Dictionary of model parameters.
    """
    try:
        logger.info("Model loading parameters:")
        for key, value in params.items():
            logger.info(f"{key}: {value}")

        model = load_model(model_path, params)
        logger.info(f"Model loaded successfully from {model_path}.")

        with open(val_loader_path, "rb") as f:
            val_loader = pickle.load(f)
        logger.info(f"Validation data loaded successfully from {val_loader_path}.")
        
        # Log the mode of val_loader
        if hasattr(val_loader, 'dataset') and hasattr(val_loader.dataset, 'training') and val_loader.dataset.training:
            logger.info("val_loader's dataset is already in training mode.")
        else:
            logger.info("val_loader's dataset is not in training mode. Setting it to training mode.")
            if hasattr(val_loader, 'dataset') and hasattr(val_loader.dataset, 'set_training'):
                val_loader.dataset.set_training(True)
            else:
                logger.warning("val_loader's dataset does not have a set_training method.")

        category_counts_masked = torch.zeros(params["num_classes"], dtype=torch.int)
        category_counts_predicted = torch.zeros(params["num_classes"], dtype=torch.int)
        category_counts_correct = torch.zeros(params["num_classes"], dtype=torch.int)

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for embeddings, categories, masks, idx in tqdm(val_loader, desc="Testing Progress"):
                embeddings = embeddings.to(device).float()
                categories = categories.to(device).long()
                masks = masks.to(device).float()
                src_key_padding_mask = (masks != -2).bool().to(device)

                outputs = model(embeddings, src_key_padding_mask=src_key_padding_mask)
                predictions = outputs.argmax(dim=-1)

                for batch_idx, indices in enumerate(idx):
                    for i in indices:
                        pred = predictions[batch_idx, i].item()
                        true_label = categories[batch_idx, i].item()
                        if pred != -1:  # Ignore padding or invalid predictions
                            if pred == true_label:
                                category_counts_correct[true_label] += 1  # Count correct predictions
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

        logger.info("Correct label counts:")
        for i in range(params["num_classes"]):
            logger.info(f"Category {i}: {category_counts_correct[i].item()} correct")

        logger.info("Testing process completed successfully.")
        print("Testing process completed successfully.")

    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        raise

if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--model_path", "-m", required=True, help="Path to the trained model.")
    @click.option("--val_loader_path", "-v", required=True, help="Path to the validation data loader.")
    @click.option("--attention", default="circular", type=click.Choice(["absolute", "relative", "circular"]), help="Attention type.")
    @click.option("--input_size", type=int, required=True, help="Input size of the embeddings.")
    @click.option("--num_classes", default=9, type=int, help="Number of output classes.")
    @click.option("--num_heads", default=4, type=int, help="Number of attention heads.")
    @click.option("--hidden_dim", default=512, type=int, help="Hidden dimension size.")
    @click.option("--lstm_hidden_dim", default=512, type=int, help="LSTM hidden dimension size.")
    @click.option("--dropout", default=0.1, type=float, help="Dropout rate.")
    @click.option("--num_layers", default=2, type=int, help="Number of transformer layers.")
    @click.option("--output_dim", type=int, help="Output dimension.")
    @click.option("--use_lstm", is_flag=True, default=False, help="Use LSTM layers.")
    @click.option("--use_positional_encoding", is_flag=True, default=True, help="Use positional encoding.")
    def main(model_path, val_loader_path, attention, input_size, num_classes, num_heads, hidden_dim, lstm_hidden_dim, dropout, num_layers, output_dim, use_lstm, use_positional_encoding):
        params = {
            "attention": attention,
            "input_size": input_size,
            "num_classes": num_classes,
            "num_heads": num_heads,
            "hidden_dim": hidden_dim,
            "lstm_hidden_dim": lstm_hidden_dim,
            "dropout": dropout,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "use_lstm": use_lstm,
            "use_positional_encoding": use_positional_encoding,
        }
        test_model(model_path, val_loader_path, params)

    main()
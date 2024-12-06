# Phynteny Transformer

### Overview

The Phynteny Transformer is a deep learning model designed to predict the missing functions of genes in phage genomes. It leverages a transformer architecture with various attention mechanisms (absolute, relative, and circular) to capture the positional information of genes. The model can be trained using K-Fold cross-validation to ensure robust performance.

### Model Architecture

The model consists of the following components:
- **Embedding Layers**: These layers embed the functional, strand, and length information of genes.
- **Positional Encoding**: Learnable or sinusoidal positional encodings are used to capture the positional information of genes.
- **LSTM Layer**: A bidirectional LSTM layer processes the embedded sequences.
- **Transformer Encoder**: Multiple transformer encoder layers with different attention mechanisms (absolute, relative, and circular) are used to capture the dependencies between genes.
- **Classification Layer**: A final linear layer classifies the genes into different functional categories.

### Training the Model

To train the model, follow these steps:

1. **Prepare the Data**: Ensure you have the training data files for X (input features) and y (target labels) in pickle format.

2. **Install Dependencies**: Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Training Script**: Use the `train_transformer.py` script to train the model. The script accepts various command-line options to fine-tune the model parameters.

    ```bash
    python train_transformer.py --x_path <path_to_X_data> --y_path <path_to_y_data> [OPTIONS]
    ```

    **Options**:
    - `--x_path`, `-x`: File path to X training data.
    - `--y_path`, `-y`: File path to y training data.
    - `--mask_portion`: Portion of knowns to mask during training (default: 0.15).
    - `--attention`: Type of positional embedding for the model (`absolute`, `relative`, `circular`).
    - `--shuffle`: Shuffle order of the genes (default: False).
    - `--gamma`: Gamma for the learning rate scheduler (default: 0.1).
    - `--step_size`: Step size for the learning rate scheduler (default: 5).
    - `--lr`: Learning rate for the optimizer (default: 1e-6).
    - `--epochs`: Number of training epochs (default: 15).
    - `--dropout`: Dropout value for dropout layer (default: 0.1).
    - `--hidden_dim`: Hidden dimension size for the transformer model (default: 512).
    - `--batch_size`: Batch size used to train the model (default: 1).
    - `--num_heads`: Number of attention heads in the transformer model (default: 4).
    - `--num_layers`: Number of transformer layers (default: 2).
    - `--lstm_hidden_dim`: LSTM hidden dimension size (default: 512).
    - `--num_splits`: Number of splits for cross-validation (default: 10).
    - `--checkpoint_interval`: Interval for saving checkpoints (default: 1).
    - `--out`, `-o`: Path to save the output (default: `train_out`).
    - `--intialisation`: Specify whether to initialise attention mechanism with zeros or random values (`random`, `zeros`).
    - `--diagonal_loss`: Specify whether to use the diagonal loss function.
    - `--lambda_penalty`: Specify the lambda penalty for the diagonal loss function (default: 0.1).
    - `--device`: Specify `cuda` or `cpu` (default: `cuda`).

### Example Command

```bash
python train_transformer.py --x_path data/X_train.pkl --y_path data/y_train.pkl --attention circular --epochs 20 --batch_size 16 --num_heads 8 --hidden_dim 1024 --out results/
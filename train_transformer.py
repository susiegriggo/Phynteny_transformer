import pickle
import click
import random
from loguru import logger
from src import model_onehot
import os


def validate_num_heads(ctx, param, value):
    if value % 4 != 0:
        raise click.BadParameter("Number of attention heads must be divisible by 4.")
    return value
 

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
@click.option("--gamma", default=0.1, help="Gamma for the learning rate scheduler.", type=float)
@click.option("--step_size", default=5, help="Step size for the learning rate scheduler.", type=int)
@click.option("--lr", default=1e-6, help="Learning rate for the optimizer.", type=float)
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
    default=1,
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
    "--diagonal_loss",
    is_flag=True,
    help="Specify whether to use the diagonal loss function.",
)
@click.option(
    "--lambda_penalty",
    default=0.1,
    help="Specify the lambda penalty for the diagonal loss function.",
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
def main(
    x_path,
    y_path,
    mask_portion,
    attention,
    shuffle,
    lr,
    gamma, 
    step_size,
    epochs,
    hidden_dim,
    num_heads,
    batch_size, 
    out,
    dropout,
    device,
    intialisation, 
    diagonal_loss,
    lambda_penalty,
    parallel_kfolds,
    num_layers,
):
    # Create the output directory if it doesn't exist
    if not os.path.exists(out):
        os.makedirs(out)
        print(f"Directory created: {out}")
    else:
        print(f"Warning: Directory {out} already exists.")

    # generate loguru object
    logger.add(out + "/trainer.log", level="DEBUG")

    # Input phrog info
    phrog_integer = pickle.load(
        open(
            "/home/grig0076/GitHubs/Phynteny_transformer/phynteny_utils/integer_category.pkl",
            "rb",
        )
    )
    phrog_integer = dict(
        zip(
            [i - 1 for i in list(phrog_integer.keys())],
            [i for i in list(phrog_integer.values())],
        )
    )  # shuffling the keys down by one

    # read in data
    logger.info("Reading in data")
    X = pickle.load(open(x_path, "rb"))
    y = pickle.load(open(y_path, "rb"))

    # Shuffle if specified
    if shuffle:
        logger.info("Shuffling gene orders...")
        for key in list(X.keys()):
            # generate indices for shuffling
            indices = list(range(len(X.get(key))))
            random.Random(4).shuffle(indices)  # shuffle with a random seed of 4
            X[key] = X[key][indices]
            y[key] = y[key][indices]

        logger.info("\t Done shuffling gene orders")
    else:
        logger.info("Gene orders not shuffled!")

    # Produce the dataset object
    # note that this is a very small training size
    
    train_dataset = model_onehot.VariableSeq2SeqEmbeddingDataset(
        list(X[0].values()), list(y[0].values()), mask_portion=mask_portion
    )
    train_dataset.set_training(True)
    logger.info("\nTraining model...")


    #TODO add step size as another parameter 
    model_onehot.train_crossValidation(
        train_dataset,
        attention,
        n_splits=10,
        batch_size=batch_size, # have changed this batch size to 16 
        epochs=epochs,
        lr=lr,
        save_path=out,
        num_heads=num_heads,
        gamma=gamma, 
        step_size=step_size,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device,
        intialisation=intialisation, 
        diagonal_loss=diagonal_loss,
        lambda_penalty=lambda_penalty,
        parallel_kfolds=parallel_kfolds,
        num_layers=num_layers,
    )

    # Evaluate the model
    # print('Evaluating the model', flush=True)
    # model_onehot.evaluate(transformer_model, test_dataloader)

    logger.info("FINISHED! :D")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()

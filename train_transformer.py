import pickle
import click
import random
from loguru import logger
from src import model_onehot
import os


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
@click.option(
    "--unmask_unknowns",
    is_flag=True,
    default=False,
    help="Do not mask unknown gene functions from the model",
)
@click.option("--lr", default=1e-6, help="Learning rate for the optimizer.", type=float)
@click.option("--epochs", default=15, help="Number of training epochs.", type=int)
@click.option(
    "--dropout", default=0.1, help="Dropout value for dropout layer.", type=int
)
@click.option(
    "--hidden_dim",
    default=512,
    help="Hidden dimension size for the transformer model.",
    type=int,
)
@click.option(
    "--num_heads",
    default=4,
    help="Number of attention heads in the transformer model.",
    type=int,
)
@click.option(
    "-o",
    "--out",
    default="train_out",
    help="Path to save the output.",
    type=click.STRING,
)
@click.option(
    "--device", default="cuda", help="specify cuda or cpu.", type=click.STRING
)
def main(
    x_path,
    y_path,
    mask_portion,
    attention,
    shuffle,
    lr,
    epochs,
    hidden_dim,
    num_heads,
    out,
    unmask_unknowns,
    dropout,
    device,
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
            "/home/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/integer_category.pkl",
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
        list(X.values()), list(y.values()), mask_portion=mask_portion
    )
    train_dataset.set_training(True)
    logger.info("\nTraining model...")

    if unmask_unknowns:
        mask_unknowns = False
    else:
        mask_unknowns = True

    model_onehot.train_crossValidation(
        train_dataset,
        attention,
        n_splits=10,
        batch_size=1, # have changed this batch size to 16 
        epochs=epochs,
        lr=lr,
        save_path=out,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device,
        mask_unknowns=mask_unknowns,
    )

    # Evaluate the model
    # print('Evaluating the model', flush=True)
    # model_onehot.evaluate(transformer_model, test_dataloader)

    logger.info("FINISHED! :D")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()

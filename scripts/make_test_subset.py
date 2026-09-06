import pickle
import random

import click
from loguru import logger


@click.command()
@click.option("--x_path", "-x", required=True, type=click.Path(exists=True), help="File path to the full X data (pickled dict).")
@click.option("--y_path", "-y", required=True, type=click.Path(exists=True), help="File path to the full y data (pickled dict).")
@click.option("--n_genomes", "-n", default=20, type=int, help="Number of genomes to keep in the subset.")
@click.option("--seed", default=42, type=int, help="Random seed used to choose which genomes are kept.")
@click.option("--out_x", default="data.X.subset.pkl", type=click.Path(), help="Output path for the subset X data.")
@click.option("--out_y", default="data.y.subset.pkl", type=click.Path(), help="Output path for the subset y data.")
def main(x_path, y_path, n_genomes, seed, out_x, out_y):
    """
    Take a small random subset of genomes from an existing X.pkl/y.pkl pair, for
    quickly smoke-testing train_transformer.py (e.g. on a memory-limited dev queue)
    without needing to load the full training data.

    This still has to load the full X.pkl/y.pkl once, so run it somewhere with
    enough memory to do that (the same partition normal training runs on, or the
    login node for a quick subset) - the point is to only have to pay that cost once,
    so every subsequent test run can use the much smaller subset files instead.
    """
    logger.info(f"Reading in full data from {x_path} and {y_path}")
    X = pickle.load(open(x_path, "rb"))
    y = pickle.load(open(y_path, "rb"))
    logger.info(f"Full dataset has {len(X)} genomes")

    keys = list(X.keys())
    if n_genomes > len(keys):
        logger.warning(f"Requested {n_genomes} genomes but only {len(keys)} are available - using all of them")
        n_genomes = len(keys)

    random.Random(seed).shuffle(keys)
    subset_keys = keys[:n_genomes]

    X_subset = {k: X[k] for k in subset_keys}
    y_subset = {k: y[k] for k in subset_keys}

    logger.info(f"Writing subset of {len(X_subset)} genomes to {out_x} and {out_y}")
    pickle.dump(X_subset, open(out_x, "wb"))
    pickle.dump(y_subset, open(out_y, "wb"))

    total_genes = sum(len(v) for v in y_subset.values())
    logger.info(f"Subset written: {len(X_subset)} genomes, {total_genes} genes total")
    logger.info("Done")


if __name__ == "__main__":
    main()

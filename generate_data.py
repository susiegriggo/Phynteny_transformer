from src import format_data
import click
import pickle
import pandas as pd
import os
from Bio import SeqIO
from loguru import logger
import transformers


@click.command()
@click.option(
    "-i",
    "--input_data",
    help="Text file containing genbank files to build model",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-d",
    "--dereplicate",
    is_flag=True,
    help="Specify whether to dedeuplicate phages whether there is duplicated gene orders",
    required=False,
)
@click.option(
    "--include_genomes", 
    help="Specify a text file of genes to only include from the input data",
    required=False,
)
@click.option(
    "--model",
    type=click.Choice([
        "facebook/esm2_t48_15B_UR50D",
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D"
    ]),
    default="facebook/esm2_t33_650M_UR50D",
    help="Specify path to model if not the default",
    required=False,
)
@click.option(
    "-m",
    "--maximum_genes",
    type=int,
    help="Specify the maximum number of genes in each genome. Default 120.",
    default=10000,
)
@click.option(
    "-g",
    "--gene_categories",
    type=int,
    help="Specify the minimum number of categories in each genome",
    default=0,
)
@click.option(
    "--prefix",
    "-p",
    default="data",
    type=str,
    help="Prefix for the output files",
)
@click.option(
    "--out",
    "-o",
    default="out",
    type=str,
    help="Directory for the output files",
)
@click.option(
    "--exclude_embedding", is_flag=True, default=False, help="Exclude esm embeddings"
)
@click.option(
    "--extra_features",
    is_flag=True,
    default=False,
    help="Whether to include extra features alongside the embedding including the strand information, orientation and gene length",
)
@click.option(
    "--data_only", 
    is_flag=True,
    default=False,
    help="Only generate the data and not the embeddings. Does not output X a y files"
)
@click.option(
    "--tokens_per_batch",
    type=int,
    default=1024,
    help="Specify the number of tokens per batch for extracting embeddings",
)
@click.option(
    "--y_only", 
    is_flag=True,
    default=False,
    help="Only generate the y object without the embeddings or the X arrays"
)
@click.option(
    "--checkpoint_path",
    type=click.Path(exists=True),
    help="Path to the model checkpoint",
    required=False,
)
@click.option(
    "--cache_dir",
    type=click.Path(exists=True),
    help="Specify the cache directory for HuggingFace transformers",
    required=False,
    default=None,  # Update default to None
)
def main(
    input_data,
    include_genomes,
    dereplicate,
    gene_categories,
    maximum_genes,
    prefix,
    out,
    model,
    exclude_embedding,
    extra_features,
    data_only,
    tokens_per_batch,
    y_only,
    checkpoint_path,
    cache_dir,
):
    # Set the cache directory for HuggingFace transformers if provided
    if cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # read in information for the phrog annotations
    # read in annotation file
    phrogs = pd.read_csv(
        "/scratch/pawsey1018/grig0076/GitHubs/Phynteny_transformer/phynteny_utils/phrog_annot_v4.tsv",
        sep="\t",
    )
    category_dict = dict(zip(phrogs["phrog"], phrogs["category"]))
    category_dict['vfdb'] = 'moron, auxiliary metabolic gene and host takeover'
    category_dict['netflax'] = 'moron, auxiliary metabolic gene and host takeover'
    category_dict['acr'] = 'moron, auxiliary metabolic gene and host takeover'
    category_dict['card'] = 'moron, auxiliary metabolic gene and host takeover'
    category_dict['defensefinder'] = 'moron, auxiliary metabolic gene and host takeover'


    # read in integer encoding of the categories - #TODO try to automate this weird step
    phrog_integer = pickle.load(
        open(
            "/scratch/pawsey1018/grig0076/GitHubs/Phynteny_transformer/phynteny_utils/integer_category.pkl",
            "rb",
        )
    )
    phrog_integer = dict(
        zip(
            [i - 1 for i in list(phrog_integer.keys())],
            [i for i in list(phrog_integer.values())],
        )
    )  # shuffling the keys down by one
    phrog_integer_reverse = dict(
        zip(list(phrog_integer.values()), list(phrog_integer.keys()))
    )
    phrog_integer_reverse["unknown function"] = -1
    phrog_integer_category = dict(
        zip(
            [str(k) for k in list(category_dict.keys())],
            [phrog_integer_reverse.get(k) for k in list(category_dict.values())],
        )
    )
    phrog_integer_category["No_PHROG"] = -1

    # Create the output directory if it doesn't exist
    try:
        if not os.path.exists(out):
            os.makedirs(out)
            logger.info(f"Directory created: {out}")
        else:
            logger.warning(f"Directory {out} already exists.")
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        raise

    # read in the genbank file/files
    logger.info("Extracting info from genbank files")
    data = format_data.fetch_data(
        input_data, gene_categories, phrog_integer_category, maximum_genes
    )
    pickle.dump(data, open(out + "/" + prefix + ".data.pkl", "wb"))

    # filter duplicate orders and include specific genomes if specified
    if dereplicate or include_genomes:
        if dereplicate:
            logger.info("Deduplicating repeated gene orders")
            data = format_data.derep_data(data)
        
        if include_genomes:
            logger.info("Filtering genomes")
            with open(include_genomes, "r") as f:
                genomes = f.read().splitlines()
            data = {k: v for k, v in data.items() if k in genomes}
        
        pickle.dump(data, open(out + "/" + prefix + ".data.pkl", "wb"))

    if data_only:
        logger.info("Data only flag set. Exiting now")
        return

    if y_only:
        logger.info("Generating y object only")
        y = format_data.prepare_y_data(data)
        pickle.dump(y, open(out + "/" + prefix + ".y.pkl", "wb"))
        logger.info('y object saved to file')
        return

    else: 
        # extract fasta from the genbank files
        logger.info("Extracting fasta from genbank")
        data_keys = list(data.keys())
        fasta_out = out + "/" + prefix + ".fasta"
        records = [data.get(k).get("sequence") for k in data_keys]
        with open(fasta_out, "w") as output_handle:
            for r in records:
                SeqIO.write(r, output_handle, "fasta")
        output_handle.close()

        # Extract the embeddings from the outputted fasta files
        logger.info("Computing ESM embeddings")
        logger.info("... if step is being slow consider using GPU!")
        embeddings = format_data.extract_embeddings(
            fasta_out, out, model_name=model, tokens_per_batch=tokens_per_batch, checkpoint_path=checkpoint_path, cache_dir=cache_dir
        )

        # move on to create training and testing data
        if extra_features:
            X, y = format_data.prepare_data(
                embeddings, data, exclude_embedding=exclude_embedding
            )

        else:
            X, y = format_data.prepare_data(
                embeddings, data, extra_features=False, exclude_embedding=exclude_embedding
            )

        # save the generated data to file
        if not y_only:
            pickle.dump(X, open(out + "/" + prefix + ".X.pkl", "wb"))
            
        pickle.dump(y, open(out + "/" + prefix + ".y.pkl", "wb"))
        logger.info('Data saved to file')


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()

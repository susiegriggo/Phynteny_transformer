from src import format_data
import click
import pickle
import pandas as pd
import os
from Bio import SeqIO


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
    help="Specify whethre to dedeuplicate phages whether there is duplicated gene orders",
    required=False,
)
@click.option(
    "--model",
    type=str,
    help="Specify path to model if not the default",
    required=False,
)
@click.option(
    "-m",
    "--maximum_genes",
    type=int,
    help="Specify the maximum number of genes in each genome. Default 120.",
    default=120,
)
@click.option(
    "-g",
    "--gene_categories",
    type=int,
    help="Specify the minimum number of categories in each genome. Default 4.",
    default=4,
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
def main(
    input_data,
    dereplicate,
    gene_categories,
    maximum_genes,
    prefix,
    out,
    model,
    exclude_embedding,
    extra_features,
):
    # read in information for the phrog annotations
    # read in annotation file
    phrogs = pd.read_csv(
        "~/susie_scratch/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/phrog_annot_v4.tsv",
        sep="\t",
    )
    category_dict = dict(zip(phrogs["phrog"], phrogs["category"]))

    # read in integer encoding of the categories - #TODO try to automate this weird step
    phrog_integer = pickle.load(
        open(
            "/scratch/pawsey1018/grig0076/GitHubs/Phynteny/phynteny_utils/phrog_annotation_info/integer_category.pkl",
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
    if not os.path.exists(out):
        os.makedirs(out)
        print(f"Directory created: {out}")
    else:
        print(f"Warning: Directory {out} already exists.")

    # read in the genbank file/files
    print("Extracting info from genbank files", flush=True)
    data = format_data.get_data(
        input_data, gene_categories, phrog_integer_category, maximum_genes
    )
    pickle.dump(data, open(out + "/" + prefix + ".data.pkl", "wb"))

    # filter duplicate orders
    if dereplicate:
        print("Deduplicating repeated gene orders", flush=True)
        data = format_data.derep_data(data)
        pickle.dump(data, open(out + "/" + prefix + ".data.pkl", "wb"))

    # extract fasta from the genbank files
    print("Extract fasta from genbank", flush=True)
    data_keys = list(data.keys())
    fasta_out = out + "/" + prefix + ".fasta"
    records = [data.get(k).get("sequence") for k in data_keys]
    with open(fasta_out, "w") as output_handle:
        for r in records:
            SeqIO.write(r, output_handle, "fasta")
    output_handle.close()

    # Extract the embeddings from the outputted fasta files
    print("Computing ESM embeddings", flush=True)
    print("... if step is being slow consider using GPU!")
    embeddings = format_data.extract_embeddings(fasta_out, out, model_name=model)

    # move on to create training and testing data
    if extra_features:
        X, y = format_data.process_data(
            embeddings, data, exclude_embedding=exclude_embedding
        )

    else:
        X, y = format_data.process_data(
            embeddings, data, extra_features=False, exclude_embedding=exclude_embedding
        )

    # save the generated data to file
    print(X)
    print(y)
    pickle.dump(X, open(out + "/" + prefix + ".X.pkl", "wb"))
    pickle.dump(y, open(out + "/" + prefix + ".y.pkl", "wb"))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()

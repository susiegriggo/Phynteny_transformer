from src import format_data 
import pickle 
import click 
import os

@click.command()
@click.option(
    "-i",
    "--input_data",
    help="Input faa file to be embedded",
    required=True,
    type=click.Path(exists=True),
) 
@click.option(
    "--out",
    "-o",
    default="out",
    type=str,
    help="Directory for the output files",
)

def main(input_data, out): 

    # Create the output directory if it doesn't exist
    if not os.path.exists(out):
        os.makedirs(out)
        print(f"Directory created: {out}")
    else:
        print(f"Warning: Directory {out} already exists.")
 
    # extract the embeddings 
    embeddings = format_data.extract_embeddings(input_data, out)
    
    # save the embedings to file 
    embeddings_out = out + fasta_out[:-4] + 'pkl' 
    pickle.dump(embeddings, open(embeddings_out, 'wb'))

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
 
# Phynteny-Transformer
![Phynteny Transformer Logo](phynteny_logo.png)

`phynteny` is annotation tool for bacteriophage genomes that integrates protein language models and synteny. 

`phynteny` leverages a transformer architecture with attention mechanisms and long short term memory to capture the positional information of genes.

`phynteny` takes a genbank file with PHROG annotations as input. If you haven't already annotaated your phage(s) with [Pharokka](https://github.com/gbouras13/pharokka) and [phold](https://github.com/gbouras13/phold) go do that and come back here! 

### Dependencies

To run the Phynteny Transformer, you need the following dependencies:

- Python 3.8+
- torch
- numpy
- pandas
- click
- loguru
- BioPython
- transformers
- importlib_resources
- scikit-learn
- tqdm

You can install the dependencies using pip:

### Installation 

#### Install Models 



### Running Phynteny 


### Containers 


### Model Architecture

The model consists of the following components:
- **Embedding Layers**: These layers embed the functional, strand, and length information of genes.
- **Positional Encoding**: Learnable or sinusoidal positional encodings are used to capture the positional information of genes.
- **LSTM Layer**: A bidirectional LSTM layer processes the embedded sequences.
- **Transformer Encoder**: Multiple transformer encoder layers with different attention mechanisms (absolute, relative, and circular) are used to capture the dependencies between genes.
- **Classification Layer**: A final linear layer classifies the genes into different functional categories.





## Bugs and Suggestions 
If you break Phynteny or would like to make any suggestions please open an issue or email me at susie.grigson@gmail.com and I'll try to get back to you. 

## Wow! how can I cite this?
Preprint available at ...

[![Edwards Lab](https://img.shields.io/badge/Bioinformatics-EdwardsLab-03A9F4)](https://edwards.flinders.edu.au)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub language count](https://img.shields.io/github/languages/count/susiegriggo/Phynteny_transformer)
[![install with pip](https://img.shields.io/static/v1?label=Install%20with&message=PIP&color=success)](https://pypi.org/project/phynteny_transformer/)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/susiegriggo/phynteny_transformer/main)
[![PyPI version](https://badge.fury.io/py/phynteny.svg)](https://badge.fury.io/py/phynteny_transformer)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/phynteny/badges/version.svg)](https://anaconda.org/bioconda/phynteny_transformer)
![Conda](https://img.shields.io/conda/dn/bioconda/phynteny_transformer)

## !! Patch 

This branch contains a patch that will eventually be properly integrated to deal with the deprecated `pkg_resources`. To implement this patch: 
```
# create new env for patched phynteny
conda create -n phynteny_transformer_pkgresource_patch -c conda-forge -c bioconda \
   python=3.10 \
     transformers=4.49.0 \
     phynteny_transformer 


# activate the new env
conda activate phynteny_transformer_pkgresource_patch 

# clone the patched version 
git clone https://github.com/susiegriggo/Phynteny_transformer.git
cd Phynteny_transformer 

# switch to the patched branch 
git checkout phynteny_importlib 

# install the patch
pip install . 

# install models (should work now without pkg resources)
install_models 

``` 
Then you should be able to run this here without any issues 

```

phynteny_transformer test_data/test_phage.gbk -o patched_test 

``` 

# Phynteny-Transformer 
![Phynteny Transformer Logo](phynteny_logo.png)

Phynteny is annotation tool for bacteriophage genomes that integrates protein language models and gene synteny. ```phynteny-transformer``` leverages a transformer architecture with attention mechanisms and long short term memory to capture the positional information of genes.

```phynteny-transformer``` takes a genbank file with PHROG annotations as input. If you haven't already annotated your phage(s) with [Pharokka](https://github.com/gbouras13/pharokka) and [Phold](https://github.com/gbouras13/phold) go do that and then come right back here! 

## Dependencies

To run ```phynteny-transformer```, you need the following dependencies:

- Python 3.9+
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

## How do I use Phynteny? 

### Installation 

#### Option 1: Installing Phynteny using conda

You can install Phynteny from bioconda at [https://anaconda.org/bioconda/phynteny](https://anaconda.org/bioconda/phynteny_transformer). Make sure you have [`conda`](https://docs.conda.io/en/latest/) installed. 
```bash
# create conda environment and install phynteny
conda create -n phynteny_transformer -c bioconda phynteny_transformer
 
# activate environment
conda activate phynteny_transformer

# install phynteny
conda install -c bioconda phynteny_transformer
```

Now you can go to [Install Models](#install-models) to install pre-trained phynteny-transformer models. 

#### Option 2: Installing Phynteny using pip
You can install Phynteny from PyPI at [https://pypi.org/project/phynteny/](https://pypi.org/project/phynteny_transformer/). 
```
pip install phynteny_transformer
```

Now you can go to [Install Models](#install-models) to install pre-trained phynteny models. 

#### Option 3: Installing Phynteny from source

You can install Phynteny Transformer from source. 

```
git clone https://github.com/susiegriggo/Phynteny_transformer 
cd Phynteny_transformer 
pip install . 
```
**NOTE:** Source installation is recommended if you would like to train your own ```phynteny-trasformer``` models. 

### Install Models 

Before you can run ```phynteny-transformer``` you'll need to install some databases

```
install_models
```

If you would like to install them to a specific location  

```
install_models -o <path/to/database_dir>
 ```

If this doesn't work you can download the models directly from [Zenodo](https://zenodo.org/records/15584824/files/phynteny_transformer_models_2025-06-03.tar.gz) and untar them yourself and point Phynteny to them with the `-m` flag. 

### Quick Start 
```
phynteny_transformer  test_data/test_phage.gbk -o test_output
```

### Output 

* ```phynteny_transformer.gbk``` contains a GenBank format file that has been updated to include annotations generated using Phynteny along with their Phynteny score and confidence. 
* ```phynteny_per_cds_funcions.tsv``` provides a table of the annotations generated (similar to the ```pharokka_cds_functions.tsv from Pharokka```)


### Advanced Usage

Phynteny Transformer provides an advanced mode for specifying the parameters of a model that you trained yourself. To see all advanced options:
```
phynteny_transformer --help --advanced
```

## How do I run Phynteny with other phage annotation tools? 

Phynteny leverages existing phage annotations to predict unknown genes. For this reason, we recommend using Phynteny after annotating using [Pharokka](https://github.com/gbouras13/pharokka) and [Phold](https://github.com/gbouras13/phold/tree/main). 

The plot below shows the annotation rate of different tools across 4 benchmarked datasets ((a) INPHARED 1419, (b) Cook, (c) Crass and (d) Tara - see the [Phold preprint](https://www.biorxiv.org/content/10.1101/2025.08.05.668817v1) for more information). The final Phynteny plots combine the benefits of annotation with Pharokka (with HMM, the second violin) followed by Phold (with structures, the fourth violin)followed by Phynteny. 

![Pharokka_Phold_Phynteny](Pharokka_Phold_Phynteny.png)


Thank you to @gbouras13 for the awesome figure! 

You can run `pharokka` + `phold` + `phynteny` in an interactive notebook using [this link](https://colab.research.google.com/github/gbouras13/phold/blob/main/run_pharokka_and_phold_and_phynteny.ipynb)

## How does Phynteny work? 
The following figure summarises how Phynteny was created! 
![Brief Overview](Phynteny_overview.png)



## Training Custom Models

Phynteny Transformer allows you to train your own custom models. To train a model, you need to provide a dataset in the required format and specify the training parameters. For more details, refer to the documentation in the train_transformer directory. 

## Bugs and Suggestions 
If you break Phynteny or would like to make any suggestions please open an issue or email me at susie.grigson@gmail.com and I'll try to get back to you. 

## Acknowledgements 
Thankyou to Laura Inglis for designing the Phynteny logo! 

Phynteny was trained using resources provided by the Pawsey Supercomputing Research Centre (Perth, Australia) which is funded by the Australian Government. Analysis was performed using the Flinders University DeepThought High Performance Cluster (https://doi.org/10. 25957/FLINDERS.HPC.DEEPTHOUGHT).

## Wow! how can I cite this?
Preprint for Phynteny is available [here](https://www.biorxiv.org/content/10.1101/2025.07.28.667340v1][https://www.biorxiv.org/content/10.1101/2025.07.28.667340v1).<br>
You can cite Phynteny as:  <br>
Grigson, S.R., Bouras, G., Papudeshi, B., Mallawaarachchi, V., Roach, M.J., Decewicz, P., & Edwards, R.A. (2025). Synteny-aware functional annotation of bacteriophage genomes with Phynteny. bioRxiv, 2025.07.28.667340. https://doi.org/10.1101/2025.07.28.667340.


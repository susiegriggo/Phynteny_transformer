#!/usr/bin/env python3
import setuptools
import os 

def get_version():
    with open("VERSION", "r") as f:
        return f.readline().strip()
    
with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(include=['phynteny_utils', 'phynteny_utils.*', 'train_transformer', 'train_transformer.*', 'src'])
print("Packages found:", packages)

# Create the phynteny_utils directory and models subdirectory if they don't exist
phynteny_utils_dir = "phynteny_utils"
if not os.path.exists(phynteny_utils_dir):
    os.makedirs(phynteny_utils_dir, exist_ok=True)
    # Create an empty __init__.py file
    with open(os.path.join(phynteny_utils_dir, "__init__.py"), "w") as f:
        pass

# Create the models subdirectory
models_dir = os.path.join(phynteny_utils_dir, "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    # Create an empty __init__.py file
    with open(os.path.join(models_dir, "__init__.py"), "w") as f:
        pass

# Define package_data to include files in the phynteny_utils directory
package_data = {
    "phynteny_utils": ["*.pkl", "*.tsv"],  # Include all .pkl and .tsv files in phynteny_utils
    "phynteny_utils.models": ["*"],  # Include all files in models subdirectory
    "phynteny_utils.phrog_annotation_info": ["*"],  # Include all files in phrog_annotation_info
}

# Remove data_files and rely on package_data for installation
data_files = []

install_requires = [
        "loguru",
        "click",
        "torch>=2.0.0",
        "numpy",
        "biopython",
        "scikit-learn",
        "transformers",
        "pandas",
        "tqdm",
    ]

setuptools.setup(
    name="phynteny",
    version=get_version(),
    zip_safe=True,
    author="Susanna Grigson",
    author_email="susie.grigson@gmail.com",
    description="Phynteny: Synteny-based prediction of bacteriophage genes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/susiegriggo/Phynteny_transformer",
    license="MIT",
    packages=packages,
    py_modules=["phynteny_transformer"],
    package_data=package_data,
    include_package_data=True,  # Ensure package data is included
    scripts=["phynteny_transformer"],
    entry_points={
        "console_scripts": [
            "generate_training_data=train_phynteny.generate_training_data:main",
            "train_model=train_phynteny.train_phyntenty:main",
            "compute_confidence=train_phynteny.compute_confidence:main",
            "install_models=src.install_models:main",
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires=">=3.8",
)

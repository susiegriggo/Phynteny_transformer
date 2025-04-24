#!/usr/bin/env python3

import glob
import os
import setuptools 

def get_version():
    with open("VERSION", "r") as f:
        return f.readline().strip()
    
with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(include=['phynteny_utils', 'phynteny_utils.*', 'train_transformer', 'train_transformer.*', 'src'])
print("Packages found:", packages)

# Define package_data to include directory structure but not necessarily the model files
package_data = {
    "phynteny_utils": ["*"],
    "phynteny_utils.models": ["__init__.py"],  # Just include the package marker
    "phynteny_utils.phrog_annotation_info": ["*"],
}

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

# Don't include model files in data_files
data_files = [
    (".", ["LICENSE", "README.md"])
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
    data_files=data_files,
    include_package_data=True,
    scripts=["phynteny_transformer"],
    entry_points={
        "console_scripts": [
            "generate_training_data=train_phynteny.generate_training_data:main",
            "train_model=train_phynteny.train_phyntenty:main",
            "compute_confidence=train_phynteny.compute_confidence:main",
            "install_models=phynteny_utils.install_models:main",
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

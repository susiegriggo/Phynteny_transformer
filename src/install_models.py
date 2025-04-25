"""
Module for handling models.

Based on code from https://github.com/gbouras13/pharokka/blob/master/bin/databases.py
"""
#!/usr/bin/env python3
import hashlib
import os
import re
import shutil
import subprocess as sp
import sys
import tarfile
from pathlib import Path
import requests
from loguru import logger
import click
import pkg_resources


PHYNTENY_MODEL_NAMES = ['fold_10transformer.model' ,
                        'fold_2transformer.model',
                        'fold_4transformer.model',
                        'fold_6transformer.model',
                         'fold_8transformer.model', 
                         'fold_1transformer.model',
                         'fold_3transformer.model',
                         'fold_5transformer.model',
                         'fold_7transformer.model',
                         'fold_9transformer.model'
]

VERSION_DICTIONARY = {
    "0.1.1": {
        "md5": "493657e916bbb9c63ef5304726da2671",
        "db_url": "https://zenodo.org/records/15276214/files/phynteny_transformer_model0.1.1_2025-04-24.tar.gz",
        #"dir_name": "phynteny_transformer_models_zenodo",
        "dir_name": ".",
    }
}


def instantiate_install(db_dir, force=False):
    """
    Begin model install

    :param db_dir: path to install the models
    :param force: if True, reinstall models even if they already exist
    """
    # Show absolute path for clarity
    abs_path = os.path.abspath(db_dir)
    logger.info(f"Model installation directory (absolute path): {abs_path}")
    
    instantiate_dir(db_dir)
    downloaded_flag = check_db_installation(db_dir) 
    if downloaded_flag == True and not force:
        logger.info(f"All phynteny models have already been downloaded and checked in: {abs_path}")
    else:
        if force and downloaded_flag:
            logger.info("Force reinstall requested. Reinstalling models...")
        else:
            logger.info("Some models are missing.")
        get_model_zenodo(db_dir)
        logger.info(f"Models successfully downloaded to: {abs_path}")


def instantiate_dir(db_dir):
    """
    Create directory to download models

    :param db_dir: path to the model directory
    """

    if os.path.isdir(db_dir) == False:
        logger.info(f"Creating models directory: {db_dir}")
        try:
            os.makedirs(db_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {db_dir}: {e}")
            sys.exit(1)


def check_db_installation(db_dir):
    """
    Check that models have been installed

    :param db_dir: path to the models directory
    """
    downloaded_flag = True

    for file_name in PHYNTENY_MODEL_NAMES:
        path = os.path.join(db_dir, file_name)
        if os.path.isfile(path) == False:
            logger.warning("Phynteny models are missing.")
            downloaded_flag = False
            break

    return downloaded_flag

"""
code from the marvellous bakta https://github.com/oschwengers/bakta, db.py specifically
"""

def download(db_url: str, tarball_path: Path):
    try:
        with tarball_path.open("wb") as fh_out, requests.get(
            db_url, stream=True
        ) as resp:
            total_length = resp.headers.get("content-length")
            if total_length is not None:  # content length header is set
                total_length = int(total_length)
                total_length_mb = total_length / (1024 * 1024)
            
            downloaded = 0
            logger.info(f"Downloading file, total size: {total_length_mb:.2f} MB")
            
            for data in resp.iter_content(chunk_size=1024 * 1024):
                fh_out.write(data)
                downloaded += len(data)
                
                if total_length is not None:
                    percent = (downloaded / total_length) * 100
                    downloaded_mb = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded_mb:.1f}/{total_length_mb:.1f} MB)")
                    sys.stdout.flush()
            
            if total_length is not None:
                sys.stdout.write("\n")  # New line after progress tracking is complete
                sys.stdout.flush()
    except IOError:
        logger.error(f"Could not download file from Zenodo! url={db_url}, path={tarball_path}")
        sys.exit(
            f"Please try again or use the manual option detailed at https://github.com/susiegriggo/Phynteny/tree/main to download from {db_url}. "
        )


def calc_md5_sum(tarball_path: Path, buffer_size: int = 1024 * 1024) -> str:
    """
    calculates the md5 for a tarball
    """
    md5 = hashlib.md5()
    with tarball_path.open("rb") as fh:
        data = fh.read(buffer_size)
        while data:
            md5.update(data)
            data = fh.read(buffer_size)
    return md5.hexdigest()


def remove_directory(dir_path):
    """
    removes directory if it exists
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def check_paths_permissions(tarball_path, output_path):
    """
    Check if paths exist and have proper permissions.
    """
    results = {}
    # Check tarball
    results["tarball_exists"] = os.path.exists(tarball_path)
    results["tarball_readable"] = os.access(tarball_path, os.R_OK) if results["tarball_exists"] else False
    results["tarball_size"] = os.path.getsize(tarball_path) if results["tarball_exists"] else 0
    
    # Check output directory
    results["output_exists"] = os.path.exists(output_path)
    results["output_writable"] = os.access(output_path, os.W_OK) if results["output_exists"] else False
    
    return results

def inspect_tarball(tarball_path):
    """
    Inspect the structure of the tarball and return details.
    """
    try:
        with tarfile.open(tarball_path, 'r:gz') as tar:
            members = tar.getmembers()
            file_count = len(members)
            top_level_dirs = {member.name.split('/')[0] for member in members if '/' in member.name}
            sample_files = [member.name for member in members[:5]]
            return {
                "status": "success",
                "file_count": file_count,
                "top_level_dirs": list(top_level_dirs),
                "sample_files": sample_files
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def untar(tarball_path: Path, output_path: Path):
    """
    untars a tarball and saves in the output_path
    """
    try:
        # Check permissions and existence first
        path_checks = check_paths_permissions(tarball_path, output_path)
        logger.info(f"Path checks: {path_checks}")
        
        if not path_checks["tarball_exists"]:
            logger.error(f"Tarball does not exist: {tarball_path}")
            sys.exit(f"Tarball not found at {tarball_path}")
            
        if not path_checks["tarball_readable"]:
            logger.error(f"Tarball is not readable: {tarball_path}")
            sys.exit(f"Cannot read tarball at {tarball_path}. Check permissions.")
            
        if not path_checks["output_exists"] or not path_checks["output_writable"]:
            logger.error(f"Output directory does not exist or is not writable: {output_path}")
            sys.exit(f"Cannot write to output directory: {output_path}. Check permissions.")
        
        # Inspect tarball structure
        tarball_info = inspect_tarball(tarball_path)
        logger.info(f"Tarball inspection: {tarball_info}")
        
        # Extract the tarball
        logger.info(f"Starting extraction: {tarball_path} to {output_path}")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            # Get a list of all members for inspection
            all_members = tar.getmembers()
            logger.info(f"Found {len(all_members)} items in tarball")
            
            # Safe extractall implementation
            for member in all_members:
                # Security check
                if member.name.startswith('/') or '..' in member.name:
                    logger.warning(f"Skipping potentially insecure path: {member.name}")
                    continue
                
                # Extract the file
                logger.debug(f"Extracting: {member.name}")
                try:
                    tar.extract(member, path=str(output_path))
                except Exception as e:
                    logger.error(f"Failed to extract {member.name}: {str(e)}")
            
            logger.info("Extraction completed")

        # Handle the extracted content based on dir_name
        dir_name = VERSION_DICTIONARY["0.1.1"]["dir_name"]
        tarpath = os.path.join(output_path, dir_name)
        
        # List contents of output_path to debug
        logger.info(f"Contents of output directory after extraction: {os.listdir(output_path)}")
        
        # If dir_name is ".", we need to handle files directly in the output_path
        if dir_name == ".":
            # Files may already be in the root, check for model files
            root_model_files = [f for f in os.listdir(output_path) if f.endswith('.model')]
            if len(root_model_files) > 0:
                logger.info(f"Found {len(root_model_files)} model files directly in output path")
                # Files already in the right place, no need to move
                return
                
        # Check if we need to look for a subdirectory
        possible_dirs = [d for d in os.listdir(output_path) 
                        if os.path.isdir(os.path.join(output_path, d)) 
                        and (d.startswith("phynteny") or "models" in d.lower())]
        
        if possible_dirs:
            logger.info(f"Found potential model directories: {possible_dirs}")
            # Use the first matching directory
            tarpath = os.path.join(output_path, possible_dirs[0])
            
        # Check if directory exists before trying to list files
        if not os.path.exists(tarpath):
            logger.error(f"Expected directory not found after extraction: {tarpath}")
            logger.error(f"Available directories: {[d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]}")
            sys.exit(
                f"Please try again or use the manual option detailed at https://github.com/susiegriggo/Phynteny/tree/main to download the models."
            )

        # Get a list of all files in the source directory
        files_to_move = [
            f for f in os.listdir(tarpath) if os.path.isfile(os.path.join(tarpath, f))
        ]
        
        logger.info(f"Found {len(files_to_move)} files to move from {tarpath} to {output_path}")
        
        # Move each file to the destination directory
        for file_name in files_to_move:
            source_path = os.path.join(tarpath, file_name)
            destination_path = os.path.join(output_path, file_name)
            logger.debug(f"Moving {source_path} to {destination_path}")
            try:
                shutil.move(source_path, destination_path)
            except Exception as e:
                logger.error(f"Failed to move {file_name}: {str(e)}")
        
        # Log completion
        logger.info(f"Successfully moved {len(files_to_move)} files to {output_path}")
        
        # remove the directory if it's not the output path
        if tarpath != output_path:
            logger.info(f"Removing temporary directory: {tarpath}")
            remove_directory(tarpath)

    except tarfile.ReadError as e:
        logger.error(f"Failed to read tarfile: {str(e)}")
        logger.error(f"Tarball may be corrupted or not a valid tar.gz file: {tarball_path}")
        sys.exit(f"Invalid tarfile. Please try downloading again.")
        
    except Exception as e:
        import traceback
        logger.error(f"Error extracting tarball: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Could not extract {tarball_path} to {output_path}")
        sys.exit(
            f"Please try again or use the manual option detailed at https://github.com/susiegriggo/Phynteny/tree/main to download the models."
        )


def get_model_zenodo(db_dir):
    """
    Download the phynteny model using the zenodo url
    confidence_dict_exists = os.path.exists(os.path.join(db_dir, "confidence_dict.pkl"))

    :param db_dir: directory to install the models
    """
    abs_path = os.path.abspath(db_dir)
    logger.info(f"Models will be installed to: {abs_path}")

    # Get parent directory for downloading the tarball
    # We want to download to phynteny_utils/ but extract to phynteny_utils/models/
    download_dir = os.path.dirname(db_dir)
    download_path = os.path.abspath(download_dir)
    logger.info(f"Downloading tarball to: {download_path}")

    db_url = VERSION_DICTIONARY["0.1.1"]["db_url"]
    requiredmd5 = VERSION_DICTIONARY["0.1.1"]["md5"]

    tarball = re.split("/", db_url)[-1]
    # Download tarball to parent directory
    tarball_path = Path(f"{download_dir}/{tarball}")

    # download the tarball
    logger.info(f"Downloading Phynteny Models from {db_url}.")
    logger.debug('path')
    logger.debug(db_url)
    download(db_url, tarball_path)

    # check md5
    md5_sum = calc_md5_sum(tarball_path)

    if md5_sum == requiredmd5:
        logger.info(f"Phynteny Models tarball download OK: {md5_sum}")
    else:
        logger.error(f"Corrupt file! MD5 should be '{requiredmd5}' but is '{md5_sum}'")
        sys.exit(
            f"Please try again or use the manual option detailed at https://github.com/susiegriggo/Phynteny/tree/main to download from {db_url}. "
        )

    logger.info(f"Extracting Phynteny Models tarball: file={tarball_path}, output={abs_path}")

    # Extract to the models directory
    untar(tarball_path, db_dir)
    
    # Don't delete the tarball - user wants it preserved
    logger.info(f"Keeping tarball at {tarball_path}")

    # List models after installation
    model_files = [f for f in os.listdir(db_dir) if f.endswith('.model')]
    logger.info(f"Installed {len(model_files)} model files")
    confidence_dict_path = os.path.join(db_dir, "confidence_dict.pkl")
    if os.path.exists(confidence_dict_path):
        logger.info("Confidence dictionary successfully installed")

    logger.info("Done.")


@click.command()
@click.option(
    "-o",
    "--outfile",
    type=click.Path(),
    help="Path to install Phynteny models",
    default=None,
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Force reinstallation of models even if they already exist",
    default=False,
)
def main(outfile, force):
    if outfile == None:
        print("Downloading Phynteny models to the default location")
        try:
            # Get the phynteny_utils directory
            phynteny_utils_dir = pkg_resources.resource_filename("phynteny_utils", "")
            # Create models subdirectory
            db_dir = os.path.join(phynteny_utils_dir, "models")
            print(f"Default package location determined as: {os.path.abspath(db_dir)}")
            # Make sure the directory exists
            os.makedirs(db_dir, exist_ok=True)
        except (ImportError, pkg_resources.DistributionNotFound) as e:
            print(f"Could not determine package location: {e}")
            # Fallback to a local directory
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            phynteny_utils_dir = os.path.join(script_dir, "phynteny_utils")
            db_dir = os.path.join(phynteny_utils_dir, "models")
            print(f"Falling back to local directory: {os.path.abspath(db_dir)}")
    else:
        # For user-specified locations, we'll add a models subdirectory if needed
        if os.path.basename(outfile) != "models":
            db_dir = os.path.join(outfile, "models")
            print(f"Creating 'models' subdirectory in specified path")
        else:
            db_dir = outfile
            
        abs_path = os.path.abspath(db_dir)
        print(f"Downloading Phynteny models to: {abs_path}")
    
    # Create the models directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)
    
    if force:
        print("Force reinstall requested. Will reinstall models even if they exist.")

    instantiate_install(db_dir, force)
    
    # Final verification and summary
    model_count = sum(1 for f in os.listdir(db_dir) if f.endswith('.model'))
    confidence_dict_exists = os.path.exists(os.path.join(db_dir, "confidence_dict.pkl"))
    
    print(f"\nInstallation Summary:")
    print(f"---------------------")
    print(f"Models directory: {os.path.abspath(db_dir)}")
    print(f"Model files installed: {model_count}")
    print(f"Confidence dictionary: {'Installed' if confidence_dict_exists else 'Missing'}")
    print(f"\nWhen running phynteny_transformer, use:")
    print(f"  phynteny_transformer -m {os.path.abspath(db_dir)} [other options] INPUT_FILE")


if __name__ == "__main__":
    main()
    
    print(f"\nInstallation Summary:")
    print(f"---------------------")
    print(f"Models directory: {os.path.abspath(db_dir)}")
    print(f"Model files installed: {model_count}")
    print(f"Confidence dictionary: {'Installed' if confidence_dict_exists else 'Missing'}")
    print(f"\nWhen running phynteny_transformer, use:")
    print(f"  phynteny_transformer -m {os.path.abspath(db_dir)} [other options] INPUT_FILE")


if __name__ == "__main__":
    main()
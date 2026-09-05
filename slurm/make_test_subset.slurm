#!/bin/bash
#SBATCH --job-name=make_test_subset
#SBATCH --output=make_test_subset_%j.log
#SBATCH --error=make_test_subset_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1

# One-off job: build a small subset of X.pkl/y.pkl for fast smoke-testing on the
# dev queue. This still has to load the FULL data once, so it runs on the same
# partition normal training runs on (not the dev queue) - after this runs once,
# the small subset files it produces are cheap to load anywhere, dev queue included.

module load singularity/4.1.0-slurm

export containerImage=/home/grig0076/susie_scratch/docker/phynteny_transformer_v1_alive_progress_2.sif

# Work around the numpy/scipy ABI mismatch in the container (harmless here since
# this script doesn't import scipy/sklearn, but cheap insurance if that changes).
export SINGULARITYENV_PYTHONPATH=/home/grig0076/susie_scratch/pyfix

DATA_DIR=/home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1
OUT_DIR=/home/grig0076/susie_scratch/phynteny_transformer/dev_test_data
mkdir -p $OUT_DIR

singularity exec --rocm $containerImage python3 scripts/make_test_subset.py \
  --x_path $DATA_DIR/data.X.pkl \
  --y_path $DATA_DIR/data.y.pkl \
  --n_genomes 20 \
  --seed 42 \
  --out_x $OUT_DIR/data.X.subset.pkl \
  --out_y $OUT_DIR/data.y.subset.pkl

echo "Subset written to $OUT_DIR"

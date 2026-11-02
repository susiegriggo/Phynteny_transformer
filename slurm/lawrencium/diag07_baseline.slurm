#!/bin/bash
#SBATCH --job-name=diag07_baseline
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --account=pc_microbiome
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1

set -euo pipefail   # fail the job (and report a nonzero exit code) if any command fails

# Diagnostic run 7/7: context-free baseline (ESM protein embedding + strand +
# length -> MLP). No attention, no positional encoding, no access to any other
# gene in the genome. Needs retraining since the old Setonix checkpoints were
# purged by the scratch-wipe. This is the reference point every other run in
# the sweep (diag01-06) is measured against: if a configuration can't clearly
# beat this, genomic context/gene order isn't adding anything in that
# configuration.

module load gcc/11.4.0
module load cuda/12.8.0

# Activate the phynteny_transformer conda env (at $SCRATCH/big_conda/) yourself
# before submitting this job with sbatch - sbatch inherits your shell's
# environment, so no conda activation happens inside the script itself.
SCRATCH=/global/home/users/susiegrigson/scratch

REPO=/global/home/users/susiegrigson/GitHubs/Phynteny_transformer
cd $REPO
export PYTHONPATH=$REPO:$PYTHONPATH   # make sure `from src...` resolves this checkout, not any installed copy

DATA_DIR=$SCRATCH/Phynteny_data/phynteny_processed_training_files_and_splits_v0.1
OUT_DIR=$SCRATCH/phynteny_diagnostic_runs
mkdir -p $OUT_DIR

FOLD=1   # same 10-fold split (random_state=42) as every other run
BATCH_SIZE=64
MASK_PORTION=0.3
LR=1e-5
DROPOUT=0.05
HIDDEN_DIM=256
EPOCHS=50

OUTPUT_DIR="$OUT_DIR/diag07_baseline_fold_${FOLD}"

echo "Diagnostic 7/7: context-free baseline MLP, fold=$FOLD"
echo "Output directory: $OUTPUT_DIR"

python3 train_transformer/train_baseline.py \
  --fold_index $FOLD \
  --x_path $DATA_DIR/data.X.pkl \
  --y_path $DATA_DIR/data.y.pkl \
  --mask_portion $MASK_PORTION \
  --epochs $EPOCHS \
  --lr $LR \
  --dropout $DROPOUT \
  --hidden_dim $HIDDEN_DIM \
  --batch_size $BATCH_SIZE \
  --out $OUTPUT_DIR \
  --force

echo "Completed diagnostic 7/7 (fold $FOLD)"

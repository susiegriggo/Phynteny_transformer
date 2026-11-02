#!/bin/bash
#SBATCH --job-name=troubleshoot_h100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=16:00:00
#SBATCH --account=pc_microbiome
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:H100:1

set -euo pipefail   # fail the job (and report a nonzero exit code) if any command fails

# Troubleshooting run only - identical config to diag01, but on the es2/H100
# partition instead of es1/A40, to check whether the failures we've been
# seeing (logging import crash on n0002.es1, OOM on n0001.es1) are specific
# to those two es1 nodes, or something more general. If this runs cleanly,
# that confirms the fault is localised to those two bad es1 nodes (already
# excluded via --exclude in the other 10 scripts) rather than anything wrong
# with the conda env, repo checkout, or training code itself.

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
NUM_HEADS=4
NUM_LAYERS=2
EPOCHS=50
ATTENTION=circular
LAMBDA_PENALTY=100
USE_PE=True

OUTPUT_DIR="$OUT_DIR/troubleshoot_h100_fold_${FOLD}"

echo "H100 troubleshooting run (same config as diag01): attention=$ATTENTION lambda_penalty=$LAMBDA_PENALTY use_positional_encoding=$USE_PE fold=$FOLD"
echo "Output directory: $OUTPUT_DIR"

python3 train_transformer/train_transformer.py \
  --fold_index $FOLD \
  --x_path $DATA_DIR/data.X.pkl \
  --y_path $DATA_DIR/data.y.pkl \
  --mask_portion $MASK_PORTION \
  --epochs $EPOCHS \
  --attention $ATTENTION \
  --lr $LR \
  --dropout $DROPOUT \
  --hidden_dim $HIDDEN_DIM \
  --num_heads $NUM_HEADS \
  --num_layers $NUM_LAYERS \
  --batch_size $BATCH_SIZE \
  --lambda_penalty $LAMBDA_PENALTY \
  --use_positional_encoding $USE_PE \
  --out $OUTPUT_DIR \
  --force

echo "Completed H100 troubleshooting run (fold $FOLD)"

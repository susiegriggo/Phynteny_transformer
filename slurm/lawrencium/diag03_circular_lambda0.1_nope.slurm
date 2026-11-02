#!/bin/bash
#SBATCH --job-name=diag03_circular_l0.1_nope
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=16:00:00
#SBATCH --account=pc_microbiome
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1

# Diagnostic run 3/7: circular attention, lambda_penalty=0.1, PE OFF.
# Lambda fix plus dropping the redundant absolute sinusoidal PE (since the
# circular relative-position bias already encodes position - stacking both is
# the "two mechanisms that don't necessarily go together" issue). This is the
# cleanest circular-attention configuration in the sweep.

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
LAMBDA_PENALTY=0.1
USE_PE=False

OUTPUT_DIR="$OUT_DIR/diag03_circular_l0.1_nope_fold_${FOLD}"

echo "Diagnostic 3/7: attention=$ATTENTION lambda_penalty=$LAMBDA_PENALTY use_positional_encoding=$USE_PE fold=$FOLD"
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

echo "Completed diagnostic 3/7 (fold $FOLD)"

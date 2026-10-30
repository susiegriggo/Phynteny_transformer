#!/bin/bash
#SBATCH --job-name=diag03_circular_l0.1_nope
#SBATCH --output=diag03_circular_l0.1_nope_%j.log
#SBATCH --error=diag03_circular_l0.1_nope_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=pc_microbiome      # TODO: confirm account/allocation on Lawrencium
#SBATCH --partition=es2              # TODO: confirm partition
#SBATCH --qos=es2_normal             # TODO: confirm qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:H100:1

# Diagnostic run 3/7: circular attention, lambda_penalty=0.1, PE OFF.
# Lambda fix plus dropping the redundant absolute sinusoidal PE (since the
# circular relative-position bias already encodes position - stacking both is
# the "two mechanisms that don't necessarily go together" issue). This is the
# cleanest circular-attention configuration in the sweep.

module load gcc/11.4.0
module load cuda/12.8.0
module load anaconda3   # TODO: adjust if Lawrencium exposes conda differently
CONDA_ENV=phynteny_transformer   # TODO: set to your actual conda env name
source activate $CONDA_ENV

BASE_DIR=/global/scratch/users/$USER/phynteny_transformer   # TODO: confirm your Lawrencium scratch path
DATA_DIR=$BASE_DIR/phynteny_processed_training_files_and_splits_v0.1
OUT_DIR=$BASE_DIR/diagnostic_runs
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

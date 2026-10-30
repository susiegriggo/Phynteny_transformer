#!/bin/bash
#SBATCH --job-name=diag01_circular_l100_pe
#SBATCH --output=diag01_circular_l100_pe_%j.log
#SBATCH --error=diag01_circular_l100_pe_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=pc_microbiome      # TODO: confirm account/allocation on Lawrencium
#SBATCH --partition=es2              # TODO: confirm partition
#SBATCH --qos=es2_normal             # TODO: confirm qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7            # 7 CPUs per H100 (matches example_slurm/example_script.slurm ratio)
#SBATCH --gres=gpu:H100:1

# Diagnostic run 1/7 of 7: single-fold sweep to isolate two suspected
# contributors to the Reviewer 3 Figure 4D periodicity artifact:
#   (a) lambda_penalty=100 in every training script, vs the CLI's documented
#       default of 0.1 (the penalty term is bounded in [0,1]; cross-entropy is
#       bounded by ln(9)~=2.2, so 100 can dominate the loss)
#   (b) stacking sinusoidal absolute positional encoding with a learned
#       relative/circular attention bias at the same time (not standard -
#       literature picks one or the other)
#
# This run: circular attention, lambda_penalty=100, PE on.
# This is the CONTROL - it should reproduce current/original model behaviour,
# including the periodicity artifact. Compare against diag02 (lambda fixed)
# and diag03 (lambda fixed + PE off).

module load gcc/11.4.0
module load cuda/12.8.0
# TODO: adjust if Lawrencium exposes conda differently (e.g. `module load anaconda3`)
module load anaconda3
CONDA_ENV=phynteny_transformer   # TODO: set to your actual conda env name
source activate $CONDA_ENV

# TODO: point these at wherever you transfer data.X.pkl / data.y.pkl on Lawrencium
BASE_DIR=/global/scratch/users/$USER/phynteny_transformer   # TODO: confirm your Lawrencium scratch path
DATA_DIR=$BASE_DIR/phynteny_processed_training_files_and_splits_v0.1
OUT_DIR=$BASE_DIR/diagnostic_runs
mkdir -p $OUT_DIR

FOLD=1   # single-fold diagnostic - same 10-fold split (random_state=42) as every other run, so comparable fold-for-fold
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

OUTPUT_DIR="$OUT_DIR/diag01_circular_l100_pe_fold_${FOLD}"

echo "Diagnostic 1/7: attention=$ATTENTION lambda_penalty=$LAMBDA_PENALTY use_positional_encoding=$USE_PE fold=$FOLD"
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

echo "Completed diagnostic 1/7 (fold $FOLD)"

#!/bin/bash
#SBATCH --job-name=diag08_lstm
#SBATCH --output=diag08_lstm_%j.log
#SBATCH --error=diag08_lstm_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=pc_microbiome      # TODO: confirm account/allocation on Lawrencium
#SBATCH --partition=es2              # TODO: confirm partition
#SBATCH --qos=es2_normal             # TODO: confirm qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:H100:1

# Leave-one-out run 1/3: same backbone as diag02 (circular attention,
# lambda_penalty=0.1, PE on) but with the BiLSTM turned on.
#
# Depends on this session's fix to TransformerClassifierCircularRelativeAttention:
# the LSTM now runs BEFORE the transformer (embeddings -> BiLSTM -> transformer),
# matching the manuscript Methods description and the other two attention
# classes - previously this class ran transformer-then-LSTM, the opposite order,
# which would have made this comparison architecturally mismatched with what
# the manuscript claims.
#
# Compares against diag02 (same backbone, LSTM off) to isolate the BiLSTM's
# contribution to gene-order modelling on its own.

module load gcc/11.4.0
module load cuda/12.8.0

SCRATCH=/global/home/users/susiegrigson/scratch
source $SCRATCH/big_conda/etc/profile.d/conda.sh
conda activate $SCRATCH/big_conda/phynteny_transformer

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
USE_PE=True

OUTPUT_DIR="$OUT_DIR/diag08_lstm_fold_${FOLD}"

echo "Diagnostic 8 (leave-one-out 1/3): attention=$ATTENTION lambda_penalty=$LAMBDA_PENALTY use_positional_encoding=$USE_PE use_lstm=True fold=$FOLD"
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
  --use_lstm \
  --out $OUTPUT_DIR \
  --force

echo "Completed diagnostic 8 (fold $FOLD)"

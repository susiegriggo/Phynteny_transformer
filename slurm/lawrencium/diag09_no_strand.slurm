#!/bin/bash
#SBATCH --job-name=diag09_no_strand
#SBATCH --output=diag09_no_strand_%j.log
#SBATCH --error=diag09_no_strand_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=pc_microbiome      # TODO: confirm account/allocation on Lawrencium
#SBATCH --partition=es2              # TODO: confirm partition
#SBATCH --qos=es2_normal             # TODO: confirm qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:H100:1

# Leave-one-out run 2/3: same backbone as diag02 (circular attention,
# lambda_penalty=0.1, PE on, LSTM off), but every gene's strand feature is
# zeroed out before it reaches the model (train_no_strand.py) - the model
# never has access to strand/orientation at all. Everything else (protein
# embeddings, gene length, one-hot context channel, gene order/position) is
# untouched. Uses a dedicated training script (mirrors train_order_only.py's
# pattern) rather than a flag on train_transformer.py, since the existing
# --ignore_strand_gene_length flag on EmbeddingDataset has stale, incorrect
# column indices left over from before the one-hot-width change, and only
# zeroes masked tokens' own features rather than every gene's.

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
LAMBDA_PENALTY=0.1

OUTPUT_DIR="$OUT_DIR/diag09_no_strand_fold_${FOLD}"

echo "Diagnostic 9 (leave-one-out 2/3): no-strand ablation, lambda_penalty=$LAMBDA_PENALTY fold=$FOLD"
echo "Output directory: $OUTPUT_DIR"

python3 train_transformer/train_no_strand.py \
  --fold_index $FOLD \
  --x_path $DATA_DIR/data.X.pkl \
  --y_path $DATA_DIR/data.y.pkl \
  --mask_portion $MASK_PORTION \
  --epochs $EPOCHS \
  --lr $LR \
  --dropout $DROPOUT \
  --hidden_dim $HIDDEN_DIM \
  --num_heads $NUM_HEADS \
  --num_layers $NUM_LAYERS \
  --batch_size $BATCH_SIZE \
  --lambda_penalty $LAMBDA_PENALTY \
  --out $OUTPUT_DIR \
  --force

echo "Completed diagnostic 9 (fold $FOLD)"

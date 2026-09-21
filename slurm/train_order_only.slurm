#!/bin/bash
#SBATCH --job-name=train_order_only
#SBATCH --output=train_order_only_%A_%a.log
#SBATCH --error=train_order_only_%A_%a.err
#SBATCH --time=23:59:30
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:8
#SBATCH --array=1-10 # one task per fold

# Order-only ablation: same architecture/attention/PE/loss as "original", but
# every gene's protein embedding is zeroed out before it reaches the model - all
# that's left is gene order/position and neighbouring genes' known categories.
# Same 10-fold split as everything else, so directly comparable fold-for-fold.

module load singularity/4.1.0-slurm

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export containerImage=/home/grig0076/susie_scratch/docker/phynteny_transformer_v1_hugging_face_5.sif
export SINGULARITYENV_PYTHONPATH=/home/grig0076/susie_scratch/pyfix

FOLD=$SLURM_ARRAY_TASK_ID
BATCH_SIZE=64
MASK_PORTION=0.3
LR=1e-5
DROPOUT=0.05
HIDDEN_DIM=256
NUM_HEADS=4
NUM_LAYERS=2
LAMBDA_PENALTY=100
EPOCHS=50

OUTPUT_DIR="/home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/order_only_fold_${FOLD}"

echo "Training order-only ablation, fold $FOLD"
echo "Output directory: $OUTPUT_DIR"

singularity exec --rocm $containerImage python3 train_transformer/train_order_only.py \
  --fold_index $FOLD \
  --x_path /home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1/data.X.pkl \
  --y_path /home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1/data.y.pkl \
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

echo "Completed order-only training for fold $FOLD"

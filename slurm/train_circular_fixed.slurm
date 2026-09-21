#!/bin/bash
#SBATCH --job-name=train_circular_fixed
#SBATCH --output=train_circular_fixed_%A_%a.log
#SBATCH --error=train_circular_fixed_%A_%a.err
#SBATCH --time=23:59:30
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:8
#SBATCH --array=1-10 # one task per fold

# Same architecture as "original", except the circular relative attention's
# wraparound fold uses each genome's own true length instead of the padded
# batch length (see architecture review: the original ties the fold point to
# whichever other random genomes share a batch, not the genome's own biology).
# Compare against sinusoidal_comparison_original_fold_N fold-for-fold.
#
# Resource note: the per-genome fix materialises a (batch, seq_len, seq_len,
# head_dim) tensor instead of the original's cheaper broadcast - more GPU memory
# for long genomes/large batches. If this OOMs, first thing to try is reducing
# BATCH_SIZE below.

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

OUTPUT_DIR="/home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/circular_fixed_fold_${FOLD}"

echo "Training circular-fixed (per-genome wraparound) model, fold $FOLD"
echo "Output directory: $OUTPUT_DIR"

singularity exec --rocm $containerImage python3 train_transformer/train_circular_fixed.py \
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

echo "Completed circular-fixed training for fold $FOLD"

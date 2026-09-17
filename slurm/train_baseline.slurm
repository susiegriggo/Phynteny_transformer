#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH --output=train_baseline_%A_%a.log
#SBATCH --error=train_baseline_%A_%a.err
#SBATCH --time=23:59:30
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:8
#SBATCH --array=1-10 # one task per fold

# Context-free baseline (ESM protein embedding + strand + length -> MLP, no
# attention/BiLSTM/positional information, no access to any other gene) - for
# Reviewer 3's "simple baseline" request and to directly test whether genomic
# context/gene order contributes anything beyond each gene's own features.
# Uses the exact same 10-fold split (random_state=42) as the attention models,
# so --fold_index N here is directly comparable to fold N of
# sinusoidal_comparison_original/no_sinusoidal.

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
EPOCHS=50

OUTPUT_DIR="/home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/baseline_mlp_fold_${FOLD}"

echo "Training context-free baseline, fold $FOLD"
echo "Output directory: $OUTPUT_DIR"

singularity exec --rocm $containerImage python3 train_transformer/train_baseline.py \
  --fold_index $FOLD \
  --x_path /home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1/data.X.pkl \
  --y_path /home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1/data.y.pkl \
  --mask_portion $MASK_PORTION \
  --epochs $EPOCHS \
  --lr $LR \
  --dropout $DROPOUT \
  --hidden_dim $HIDDEN_DIM \
  --batch_size $BATCH_SIZE \
  --out $OUTPUT_DIR \
  --force

echo "Completed baseline training for fold $FOLD"

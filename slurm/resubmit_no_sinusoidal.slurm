#!/bin/bash
#SBATCH --job-name=resubmit_no_sinusoidal
#SBATCH --output=resubmit_no_sinusoidal_%j.log
#SBATCH --error=resubmit_no_sinusoidal_%j.err
#SBATCH --time=23:59:30
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1
# Only fold 1 needs re-running - folds 2-10 for no_sinusoidal already completed
# successfully (all 50 epochs, final_validation_*.pkl present). Fold 1 got partway
# (to epoch 16) then a retry hit the "directory already exists" guard because
# --force wasn't set - that's what --force below fixes. No array needed for a
# single fold, and gpu:1 (not gpu:8) since only one fold is training here.

module load singularity/4.1.0-slurm

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export containerImage=/home/grig0076/susie_scratch/docker/phynteny_transformer_v1_hugging_face_5.sif
export SINGULARITYENV_PYTHONPATH=/home/grig0076/susie_scratch/pyfix

FOLD=1
BATCH_SIZE=64
MASK_PORTION=0.3
ATTENTION="circular"
LR=1e-5
DROPOUT=0.05
HIDDEN_DIM=256
NUM_HEADS=4
NUM_LAYERS=2
LAMBDA_PENALTY=100
EPOCHS=50

OUTPUT_DIR="/home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/sinusoidal_comparison_no_sinusoidal_fold_${FOLD}"

echo "Re-running no_sinusoidal, fold $FOLD"
echo "Output directory: $OUTPUT_DIR (will overwrite the incomplete previous attempt)"

singularity exec --rocm $containerImage python3 train_transformer.py \
  --fold_index $FOLD \
  --x_path /home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1/data.X.pkl \
  --y_path /home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1/data.y.pkl \
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
  --out $OUTPUT_DIR \
  --force \
  --use_positional_encoding False

echo "Completed re-run for no_sinusoidal, fold $FOLD"

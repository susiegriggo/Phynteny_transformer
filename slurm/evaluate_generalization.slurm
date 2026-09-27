#!/bin/bash
#SBATCH --job-name=evaluate_generalization_circular_fixed
#SBATCH --output=evaluate_generalization_circular_fixed_%A_%a.log
#SBATCH --error=evaluate_generalization_circular_fixed_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-10 # one task per fold - evaluates BOTH circular_fixed and baseline
# for that fold in a single job (same reasoning as evaluate_phold_recovery.slurm).
# NOTE: circular_fixed still has folds mid-training - restrict this to the
# folds that are actually finished, e.g. sbatch --array=1,2,3,4,5,6,7,8
# slurm/evaluate_generalization.slurm (skip whichever 2 folds are still running).

module load singularity/4.1.0-slurm

export containerImage=/home/grig0076/susie_scratch/docker/phynteny_transformer_v1_hugging_face_5.sif
export SINGULARITYENV_PYTHONPATH=/home/grig0076/susie_scratch/pyfix

FOLD=$SLURM_ARRAY_TASK_ID
DATA_DIR=/home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1
PHOLD_DIR=/home/grig0076/scratch/Phynteny_transformer/phynteny_intermediate_files_v0.1/Phagescope_curated_data
MODELS_DIR=/home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope
OUT_DIR=/home/grig0076/susie_scratch/phynteny_transformer/generalization_results
mkdir -p $OUT_DIR

echo "Evaluating generalization diagnostics (circular_fixed vs baseline) for fold $FOLD"

singularity exec --rocm $containerImage python3 scripts/evaluate_generalization.py \
  --x_path $DATA_DIR/data.X.pkl \
  --y_path $DATA_DIR/data.y.pkl \
  --phold_y_path $PHOLD_DIR/phold_data.y.pkl \
  --circular_fixed_checkpoint $MODELS_DIR/circular_fixed_fold_${FOLD}/fold_${FOLD}/transformer_state_dict.pth \
  --baseline_checkpoint $MODELS_DIR/baseline_mlp_fold_${FOLD}/fold_${FOLD}/transformer_state_dict.pth \
  --fold_index $FOLD \
  --n_passes 5 \
  --out $OUT_DIR/generalization_circular_fixed_fold_${FOLD}.json

echo "Completed fold $FOLD"

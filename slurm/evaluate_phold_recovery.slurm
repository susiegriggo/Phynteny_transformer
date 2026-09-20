#!/bin/bash
#SBATCH --job-name=evaluate_phold_recovery
#SBATCH --output=evaluate_phold_recovery_%A_%a.log
#SBATCH --error=evaluate_phold_recovery_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-10 # one task per fold - evaluates BOTH original and baseline
# for that fold in a single job, to only pay the ~6 minute data-load cost once
# per fold rather than once per (model type, fold) combination.

module load singularity/4.1.0-slurm

export containerImage=/home/grig0076/susie_scratch/docker/phynteny_transformer_v1_hugging_face_5.sif
export SINGULARITYENV_PYTHONPATH=/home/grig0076/susie_scratch/pyfix

FOLD=$SLURM_ARRAY_TASK_ID
DATA_DIR=/home/grig0076/susie_scratch/phynteny_transformer/phynteny_processed_training_files_and_splits_v0.1
PHOLD_DIR=/home/grig0076/scratch/Phynteny_transformer/phynteny_intermediate_files_v0.1/Phagescope_curated_data
MODELS_DIR=/home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope
OUT_DIR=/home/grig0076/susie_scratch/phynteny_transformer/phold_recovery_results
mkdir -p $OUT_DIR

echo "Evaluating phold-recovery agreement for fold $FOLD"

singularity exec --rocm $containerImage python3 scripts/evaluate_phold_recovery.py \
  --x_path $DATA_DIR/data.X.pkl \
  --y_path $DATA_DIR/data.y.pkl \
  --phold_y_path $PHOLD_DIR/phold_data.y.pkl \
  --original_checkpoint $MODELS_DIR/sinusoidal_comparison_original_fold_${FOLD}/fold_${FOLD}/transformer_state_dict.pth \
  --baseline_checkpoint $MODELS_DIR/baseline_mlp_fold_${FOLD}/fold_${FOLD}/transformer_state_dict.pth \
  --fold_index $FOLD \
  --out $OUT_DIR/phold_recovery_fold_${FOLD}.json

echo "Completed fold $FOLD"

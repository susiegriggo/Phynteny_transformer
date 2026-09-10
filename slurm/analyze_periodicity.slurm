#!/bin/bash
#SBATCH --job-name=analyze_periodicity
#SBATCH --output=analyze_periodicity_%A_%a.log
#SBATCH --error=analyze_periodicity_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-dev
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2 # one task per model type: original, no_sinusoidal, untrained
# NOTE: no GPU is actually needed for this - it's pure numpy aggregation over the
# already-saved final_validation_*.pkl files. --gres=gpu:1/gpu-dev is only here to
# reuse a queue known to schedule fast; switch to a CPU-only partition if you have one.

module load singularity/4.1.0-slurm

export containerImage=/home/grig0076/susie_scratch/docker/phynteny_transformer_v1_hugging_face_5.sif
export SINGULARITYENV_PYTHONPATH=/home/grig0076/susie_scratch/pyfix

MODEL_TYPES=("original" "no_sinusoidal" "untrained")
MODEL_TYPE=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}

# Update this to wherever the trained_models directory actually lives
BASE_DIR=/home/grig0076/susie_scratch/phynteny_transformer/trained_models
OUT_DIR=/home/grig0076/susie_scratch/phynteny_transformer/periodicity_results
mkdir -p $OUT_DIR

echo "Aggregating OOF validation attention for model_type=$MODEL_TYPE"

singularity exec --rocm $containerImage python3 scripts/analyze_periodicity.py \
  --base_dir $BASE_DIR \
  --model_type $MODEL_TYPE \
  --folds 1,2,3,4,5,6,7,8,9,10 \
  --max_lag 40 \
  --out $OUT_DIR/periodicity_${MODEL_TYPE}.npz

echo "Done: $OUT_DIR/periodicity_${MODEL_TYPE}.npz"

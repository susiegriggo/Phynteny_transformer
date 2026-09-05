#!/bin/bash
#SBATCH --job-name=smoke_test_comparison
#SBATCH --output=smoke_test_%A_%a.log
#SBATCH --error=smoke_test_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=dev
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2 # 1 fold x 3 model types = 3 total combinations
# NOTE: adjust --partition/--account above to match your actual dev queue name -
# "dev" is a placeholder, check with `sinfo` if you're not sure what it's called.

module load singularity/4.1.0-slurm

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export containerImage=/home/grig0076/susie_scratch/docker/phynteny_transformer_v1_alive_progress_2.sif

# Work around the numpy/scipy ABI mismatch baked into the container (numpy linked
# against openblas64/ILP64 while scipy expects the standard LP64 build). Must already
# exist - build it once from the login node:
#   singularity exec --rocm $containerImage python3 -m pip install \
#     --target=/home/grig0076/susie_scratch/pyfix --no-deps \
#     --index-url https://pypi.org/simple numpy==1.26.4
export SINGULARITYENV_PYTHONPATH=/home/grig0076/susie_scratch/pyfix

# Small subset data produced by slurm/make_test_subset.slurm - fast to load, so this
# should fit comfortably within the dev queue's memory limit.
DATA_DIR=/home/grig0076/susie_scratch/phynteny_transformer/dev_test_data

# Single fold only, tiny model, few epochs - the goal here is just to confirm the
# whole pipeline runs end to end (imports, data loading, all three model types),
# not to produce meaningful results.
FOLD=1
MODEL_TYPES=("original" "no_sinusoidal" "untrained")
BATCH_SIZE=4
MASK_PORTION=0.3
ATTENTION="circular"
LR=1e-5
DROPOUT=0.05
HIDDEN_DIM=64
NUM_HEADS=4
NUM_LAYERS=2
LAMBDA_PENALTY=100
EPOCHS=2

MODEL_TYPE=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}
echo "Debug: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID, MODEL_TYPE=$MODEL_TYPE"

POSITIONAL_ENCODING_FLAGS=""
UNTRAINED_FLAG=""

if [ "$MODEL_TYPE" == "no_sinusoidal" ]; then
    POSITIONAL_ENCODING_FLAGS="--use_positional_encoding False"
    echo "Smoke-testing model WITHOUT sinusoidal positional encoding"
elif [ "$MODEL_TYPE" == "untrained" ]; then
    UNTRAINED_FLAG="--untrained"
    echo "Smoke-testing UNTRAINED/randomly initialised model"
else
    echo "Smoke-testing ORIGINAL model with sinusoidal positional encoding"
fi

OUTPUT_DIR="/home/grig0076/susie_scratch/phynteny_transformer/trained_models/smoke_test_${MODEL_TYPE}"

echo "Starting run for:"
echo "  Model type: $MODEL_TYPE"
echo "  Output directory: $OUTPUT_DIR"

singularity exec --rocm $containerImage python3 train_transformer.py \
  --fold_index $FOLD \
  --x_path $DATA_DIR/data.X.subset.pkl \
  --y_path $DATA_DIR/data.y.subset.pkl \
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
  $POSITIONAL_ENCODING_FLAGS \
  $UNTRAINED_FLAG

echo "Completed smoke test for model type: $MODEL_TYPE"

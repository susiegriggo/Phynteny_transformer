#!/bin/bash
#SBATCH --job-name=train_transformer_comparison
#SBATCH --output=train_transformer_%A_%a.log
#SBATCH --error=train_transformer_%A_%a.err
#SBATCH --time=23:59:30
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:8
#SBATCH --array=0-19 # 10 folds x 2 model types = 20 total combinations

# Load necessary modules
module load singularity/4.1.0-slurm

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
#export containerImage=/scratch/pawsey1018/grig0076/docker/phynteny_transformer_v1_12102024_2.sif
export containerImage=/home/grig0076/susie_scratch/docker/phynteny_transformer_v1_hugging_face_5.sif

# Define parameter values - using your original parameters
FOLDS=(1 2 3 4 5 6 7 8 9 10)
MODEL_TYPES=("original" "no_sinusoidal")
BATCH_SIZE=64
MASK_PORTION=0.3
ATTENTION="circular"
LR=1e-5
DROPOUT=0.05
HIDDEN_DIM=256
NUM_HEADS=4
NUM_LAYERS=2
LAMBDA_PENALTY=100

# Calculate the total number of combinations using proper arithmetic
FOLD_COUNT=${#FOLDS[@]}
MODEL_TYPE_COUNT=${#MODEL_TYPES[@]}
TOTAL_COMBINATIONS=$((FOLD_COUNT * MODEL_TYPE_COUNT))

echo "Debug: FOLD_COUNT=$FOLD_COUNT, MODEL_TYPE_COUNT=$MODEL_TYPE_COUNT, TOTAL_COMBINATIONS=$TOTAL_COMBINATIONS"
echo "Debug: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# Ensure the array index is within the range of total combinations
if [ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]; then
  echo "Array index $SLURM_ARRAY_TASK_ID out of range (max: $((TOTAL_COMBINATIONS - 1)))"
  exit 1
fi

# Calculate the indices for each parameter using proper arithmetic
FOLD_IDX=$((SLURM_ARRAY_TASK_ID % FOLD_COUNT))
MODEL_TYPE_IDX=$((SLURM_ARRAY_TASK_ID / FOLD_COUNT))

echo "Debug: FOLD_IDX=$FOLD_IDX, MODEL_TYPE_IDX=$MODEL_TYPE_IDX"

# Get the parameter values for this task
FOLD=${FOLDS[$FOLD_IDX]}
MODEL_TYPE=${MODEL_TYPES[$MODEL_TYPE_IDX]}

echo "Debug: Selected FOLD=$FOLD, MODEL_TYPE=$MODEL_TYPE"

# Set model-specific parameters
if [ "$MODEL_TYPE" == "no_sinusoidal" ]; then
    DISABLE_SINUSOIDAL_PE="--disable_sinusoidal_pe"
    OUTPUT_SUFFIX="_no_sinusoidal"
    echo "Training model WITHOUT sinusoidal positional encoding for fold $FOLD"
else
    DISABLE_SINUSOIDAL_PE=""
    OUTPUT_SUFFIX="_original"
    echo "Training ORIGINAL model with sinusoidal positional encoding for fold $FOLD"
fi

# Create output directory path
OUTPUT_DIR="/home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/sinusoidal_comparison_${MODEL_TYPE}_fold_${FOLD}"

echo "Starting training for:"
echo "  Model type: $MODEL_TYPE"
echo "  Fold: $FOLD"
echo "  Output directory: $OUTPUT_DIR"
echo "  Parameters: batch_size=$BATCH_SIZE, mask_portion=$MASK_PORTION, attention=$ATTENTION"
echo "  Learning rate: $LR, dropout=$DROPOUT, hidden_dim=$HIDDEN_DIM"
echo "  Num heads: $NUM_HEADS, num_layers=$NUM_LAYERS, lambda_penalty=$LAMBDA_PENALTY"

# Run the script with the selected parameters
singularity exec --rocm $containerImage python3 train_transformer.py \
  --fold_index $FOLD \
  --x_path /home/grig0076/susie_scratch/phynteny_transformer/data/phynteny_processed_training_files_and_splits_v0.1/data.X.pkl \
  --y_path /home/grig0076/susie_scratch/phynteny_transformer/data/phynteny_processed_training_files_and_splits_v0.1/data.y.pkl \
  --mask_portion $MASK_PORTION \
  --epochs 50 \
  --attention $ATTENTION \
  --lr $LR \
  --dropout $DROPOUT \
  --hidden_dim $HIDDEN_DIM \
  --num_heads $NUM_HEADS \
  --num_layers $NUM_LAYERS \
  --batch_size $BATCH_SIZE \
  --lambda_penalty $LAMBDA_PENALTY \
  --out $OUTPUT_DIR \
  $DISABLE_SINUSOIDAL_PE

echo "Completed training for model type: $MODEL_TYPE, fold: $FOLD"

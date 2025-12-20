#!/bin/bash
#SBATCH --job-name=train_transformer_grid
#SBATCH --output=train_transformer_%A_%a.log
#SBATCH --error=train_transformer_%A_%a.err
#SBATCH --time=23:59:30
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:8
#SBATCH --array=0-19 # set this to the total num of combinations (updated for positional encoding types)

# Load necessary modules
module load singularity/4.1.0-slurm

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
export containerImage=/scratch/pawsey1018/grig0076/docker/phynteny_transformer_v1_hugging_face_5.sif

# Define parameter values - using published Phynteny parameters
FOLDS=(1 2 3 4 5 6 7 8 9 10)
BATCH_SIZES=(16)  # Published parameter
MASK_PORTIONS=(0.15)  # Published parameter  
ATTENTIONS=("circular")
LRS=(1e-4)  # Published parameter: learning_rate=10^-4
DROPOUTS=(0.1)  # Standard dropout (not specified in paper)
HIDDEN_DIMS=(512)  # Published parameter: hidden_dimension=512
NUM_HEADS=(8)  # Not specified, using reasonable default for 512 hidden_dim
NUM_LAYERS=(2)  # Published parameter: num_transformer_layers=2
LAMBDA_PENALTIES=(0.1)  # Standard value
PROTEIN_DROPOUT_RATES=(0.7)  # Published parameter: protein_dropout_rate=0.7
POSITIONAL_ENCODING_TYPES=("sinusoidal" "none")  # Compare sinusoidal (reviewer's concern) vs none

# Calculate the total number of combinations
TOTAL_COMBINATIONS=${#FOLDS[@]}*${#BATCH_SIZES[@]}*${#MASK_PORTIONS[@]}*${#ATTENTIONS[@]}*${#LRS[@]}*${#DROPOUTS[@]}*${#HIDDEN_DIMS[@]}*${#NUM_HEADS[@]}*${#NUM_LAYERS[@]}*${#LAMBDA_PENALTIES[@]}*${#PROTEIN_DROPOUT_RATES[@]}*${#POSITIONAL_ENCODING_TYPES[@]}

# Ensure the array index is within the range of total combinations
if [ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]; then
  echo "Array index out of range"
  exit 1
fi

# Calculate the indices for each parameter
FOLD_IDX=$((SLURM_ARRAY_TASK_ID % ${#FOLDS[@]}))
BATCH_SIZE_IDX=$(((SLURM_ARRAY_TASK_ID / ${#FOLDS[@]}) % ${#BATCH_SIZES[@]}))
MASK_PORTION_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]})) % ${#MASK_PORTIONS[@]}))
ATTENTION_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]})) % ${#ATTENTIONS[@]}))
LR_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]} * ${#ATTENTIONS[@]})) % ${#LRS[@]}))
DROPOUT_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]} * ${#ATTENTIONS[@]} * ${#LRS[@]})) % ${#DROPOUTS[@]}))
HIDDEN_DIM_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]} * ${#ATTENTIONS[@]} * ${#LRS[@]} * ${#DROPOUTS[@]})) % ${#HIDDEN_DIMS[@]}))
NUM_HEADS_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]} * ${#ATTENTIONS[@]} * ${#LRS[@]} * ${#DROPOUTS[@]} * ${#HIDDEN_DIMS[@]})) % ${#NUM_HEADS[@]}))
NUM_LAYERS_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]} * ${#ATTENTIONS[@]} * ${#LRS[@]} * ${#DROPOUTS[@]} * ${#HIDDEN_DIMS[@]} * ${#NUM_HEADS[@]})) % ${#NUM_LAYERS[@]}))
LAMBDA_PENALTY_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]} * ${#ATTENTIONS[@]} * ${#LRS[@]} * ${#DROPOUTS[@]} * ${#HIDDEN_DIMS[@]} * ${#NUM_HEADS[@]} * ${#NUM_LAYERS[@]})) % ${#LAMBDA_PENALTIES[@]}))
PROTEIN_DROPOUT_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]} * ${#ATTENTIONS[@]} * ${#LRS[@]} * ${#DROPOUTS[@]} * ${#HIDDEN_DIMS[@]} * ${#NUM_HEADS[@]} * ${#NUM_LAYERS[@]} * ${#LAMBDA_PENALTIES[@]})) % ${#PROTEIN_DROPOUT_RATES[@]}))
POSITIONAL_ENCODING_IDX=$(((SLURM_ARRAY_TASK_ID / (${#FOLDS[@]} * ${#BATCH_SIZES[@]} * ${#MASK_PORTIONS[@]} * ${#ATTENTIONS[@]} * ${#LRS[@]} * ${#DROPOUTS[@]} * ${#HIDDEN_DIMS[@]} * ${#NUM_HEADS[@]} * ${#NUM_LAYERS[@]} * ${#LAMBDA_PENALTIES[@]} * ${#PROTEIN_DROPOUT_RATES[@]})) % ${#POSITIONAL_ENCODING_TYPES[@]}))

# Get the parameter values for this task
FOLD=${FOLDS[$FOLD_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BATCH_SIZE_IDX]}
MASK_PORTION=${MASK_PORTIONS[$MASK_PORTION_IDX]}
ATTENTION=${ATTENTIONS[$ATTENTION_IDX]}
LR=${LRS[$LR_IDX]}
DROPOUT=${DROPOUTS[$DROPOUT_IDX]}
HIDDEN_DIM=${HIDDEN_DIMS[$HIDDEN_DIM_IDX]}
NUM_HEADS=${NUM_HEADS[$NUM_HEADS_IDX]}
NUM_LAYERS=${NUM_LAYERS[$NUM_LAYERS_IDX]}
LAMBDA_PENALTY=${LAMBDA_PENALTIES[$LAMBDA_PENALTY_IDX]}
PROTEIN_DROPOUT=${PROTEIN_DROPOUT_RATES[$PROTEIN_DROPOUT_IDX]}
POSITIONAL_ENCODING=${POSITIONAL_ENCODING_TYPES[$POSITIONAL_ENCODING_IDX]}

# Run the script with the selected parameters
singularity exec --rocm $containerImage python3 /home/grig0076/susie_scratch/GitHubs/Phynteny_transformer/train_transformer.py \
  --fold_index $FOLD \
  --x_path /home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/merged_data.X.pkl \
  --y_path /home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/merged_data.y.pkl \
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
  --protein_dropout_rate $PROTEIN_DROPOUT \
  --positional_encoding_type $POSITIONAL_ENCODING \
  --out /home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/grid_search_phynteny_params_sinusoidal_vs_none/${BATCH_SIZE}_${MASK_PORTION}_${ATTENTION}_${LR}_${DROPOUT}_${HIDDEN_DIM}_${NUM_HEADS}_${NUM_LAYERS}_${LAMBDA_PENALTY}_${PROTEIN_DROPOUT}_${POSITIONAL_ENCODING}_${FOLD}

#!/bin/bash
#SBATCH --job-name=train_transformer_grid
#SBATCH --output=train_transformer_%A_%a.log
#SBATCH --error=train_transformer_%A_%a.err
#SBATCH --time=23:59:30
#SBATCH --partition=gpu-highmem
#SBATCH --account=pawsey1018-gpu
#SBATCH --gres=gpu:8
#SBATCH --array=0-360

# Load necessary modules
module load singularity/4.1.0-slurm

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
export containerImage=/scratch/pawsey1018/grig0076/docker/phynteny_transformer_v1_12102024_2.sif

# Define parameter values
FOLDS=(1 2 3 4 5 6 7 8 9 10)
BATCH_SIZES=(64)
MASK_PORTIONS=(0.1 0.2 0.3)
ATTENTIONS=( "circular")
LRS=(1e-4 1e-5)
DROPOUTS=(0.05 0.1 0.2)
HIDDEN_DIMS=(256) #also do 512 
NUM_HEADS=(4)
NUM_LAYERS=(2)
LAMBDA_PENALTIES=(100 10)

# Calculate the total number of combinations
TOTAL_COMBINATIONS=${#FOLDS[@]}*${#BATCH_SIZES[@]}*${#MASK_PORTIONS[@]}*${#ATTENTIONS[@]}*${#LRS[@]}*${#DROPOUTS[@]}*${#HIDDEN_DIMS[@]}*${#NUM_HEADS[@]}*${#NUM_LAYERS[@]}*${#LAMBDA_PENALTIES[@]}

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

# Run the script with the selected parameters
singularity exec --rocm $containerImage python3 /home/grig0076/GitHubs/Phynteny_transformer/train_transformer.py \
  --fold_index $FOLD \
  --x_path /home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/merged_data.X.pkl \
  --y_path /home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/merged_data.y.pkl \
  --mask_portion $MASK_PORTION \
  --epochs 100 \
  --attention $ATTENTION \
  --lr $LR \
  --dropout $DROPOUT \
  --hidden_dim $HIDDEN_DIM \
  --num_heads $NUM_HEADS \
  --num_layers $NUM_LAYERS \
  --batch_size $BATCH_SIZE \
  --lambda_penalty $LAMBDA_PENALTY \
  --out /home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope/grid_search_19012025_origoutputdim/${BATCH_SIZE}_${MASK_PORTION}_${ATTENTION}_${LR}_${DROPOUT}_${HIDDEN_DIM}_${NUM_HEADS}_${NUM_LAYERS}_${LAMBDA_PENALTY}_${FOLD}

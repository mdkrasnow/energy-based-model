#!/bin/bash

# Training launch script for GPU VM
# Run this ON the VM after setup

set -e

# Configuration
BUCKET_NAME="${GCS_BUCKET:-}"
USE_GCS="${USE_GCS:-false}"

# Training hyperparameters (adjust as needed)
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=100
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="./logs"

# If using GCS
if [ "$USE_GCS" == "true" ]; then
    if [ -z "$BUCKET_NAME" ]; then
        echo "Enter your GCS bucket name (without gs://):"
        read BUCKET_NAME
    fi
    
    # Mount GCS bucket if not already mounted
    if [ ! -d "/gcs/data" ]; then
        echo "Mounting GCS bucket..."
        sudo gcsfuse $BUCKET_NAME /gcs
    fi
    
    DATA_DIR="/gcs/data"
    CHECKPOINT_DIR="/gcs/checkpoints"
    LOG_DIR="/gcs/logs"
else
    DATA_DIR="./data"
fi

echo "=== Starting Training ==="
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Log directory: $LOG_DIR"
echo

# Ensure conda environment is activated
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ebm

# Create directories if they don't exist
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo

# Set environment variables for better GPU performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.5"  # For T4 GPU

# Launch training with accelerate
echo "Starting training..."
echo

# Basic training command (adjust based on your train.py arguments)
python train.py \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --device cuda \
    --num_workers 4

# Alternative: Use accelerate for distributed training or mixed precision
# accelerate launch train.py \
#     --data_dir $DATA_DIR \
#     --checkpoint_dir $CHECKPOINT_DIR \
#     --log_dir $LOG_DIR \
#     --batch_size $BATCH_SIZE \
#     --learning_rate $LEARNING_RATE \
#     --num_epochs $NUM_EPOCHS \
#     --mixed_precision fp16

echo
echo "Training completed!"
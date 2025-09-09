#!/bin/bash

# Sync local code to GPU VM

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
ZONE="us-east1-b"
INSTANCE_NAME="ebm-training-vm"
LOCAL_PROJECT_DIR="../"  # Parent directory (energy-based-model)
REMOTE_PROJECT_DIR="~/projects/energy-based-model"

if [ -z "$PROJECT_ID" ]; then
    echo "Enter your GCP Project ID:"
    read PROJECT_ID
fi

echo "Syncing code to VM..."
echo "From: $LOCAL_PROJECT_DIR"
echo "To: $INSTANCE_NAME:$REMOTE_PROJECT_DIR"
echo

# Create remote directory if it doesn't exist
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="mkdir -p $REMOTE_PROJECT_DIR"

# Sync code using gcloud compute scp
# Excluding unnecessary files
gcloud compute scp --recurse \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --exclude=".git/*,__pycache__/*,*.pyc,.DS_Store,checkpoints/*,logs/*,data/*,*.pth,*.pt,wandb/*,.ipynb_checkpoints/*" \
    $LOCAL_PROJECT_DIR/* \
    $INSTANCE_NAME:$REMOTE_PROJECT_DIR/

echo
echo "Code synced successfully!"
echo
echo "To connect to VM and start training:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo
echo "Then on the VM:"
echo "cd ~/projects/energy-based-model"
echo "conda activate ebm"
echo "./run-training.sh"
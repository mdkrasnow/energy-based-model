#!/bin/bash

# Create GPU VM Instance for Training
# Uses NVIDIA T4 (similar to RTX 2080 but with 16GB VRAM)

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
ZONE="us-east1-b"
INSTANCE_NAME="ebm-training-vm"
MACHINE_TYPE="n1-standard-8"  # 8 vCPUs, 30GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="200GB"

# Deep Learning VM Image (includes PyTorch, CUDA, Python)
# Using common-cu113 which is more widely available and doesn't require specific image
IMAGE_FAMILY="common-cu113"
IMAGE_PROJECT="deeplearning-platform-release"

if [ -z "$PROJECT_ID" ]; then
    echo "Enter your GCP Project ID:"
    read PROJECT_ID
fi

echo "Creating GPU VM Instance..."
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo "GPU: $GPU_COUNT x $GPU_TYPE"
echo

# Create the instance
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo
echo "VM created successfully!"
echo
echo "Waiting for instance to be ready..."
sleep 30

# Get the external IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE --project=$PROJECT_ID \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo
echo "=== VM Instance Ready ==="
echo "External IP: $EXTERNAL_IP"
echo
echo "Connect with: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo
echo "Or directly with SSH:"
echo "ssh -i ~/.ssh/google_compute_engine $USER@$EXTERNAL_IP"
echo
echo "Next steps:"
echo "1. Run: ./setup-vm-environment.sh"
echo "2. Then: ./sync-code-to-vm.sh"
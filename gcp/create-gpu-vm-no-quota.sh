#!/bin/bash

# Alternative: Create CPU-only VM and use Colab-style approach
# This avoids GPU quota requirements entirely

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
ZONE="us-east1-b"
INSTANCE_NAME="ebm-training-cpu"
MACHINE_TYPE="e2-highmem-8"  # 8 vCPUs, 64GB RAM, cost-effective
BOOT_DISK_SIZE="200GB"

# Using standard Ubuntu with CUDA-ready environment
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

if [ -z "$PROJECT_ID" ]; then
    echo "Enter your GCP Project ID:"
    read PROJECT_ID
fi

echo "Creating CPU VM Instance (No GPU quota required)..."
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo

# Create the instance without GPU
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-balanced \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo
echo "VM created successfully!"
echo
echo "=== IMPORTANT: NO GPU ALTERNATIVES ==="
echo
echo "Since GPU quotas are restricted, you have these options:"
echo
echo "1. Use this CPU instance for development/testing with small batches"
echo "2. Use Google Colab (FREE GPU): https://colab.research.google.com"
echo "   - Mount your Google Drive"
echo "   - Upload your code and run with free T4 GPU"
echo
echo "3. Use Kaggle Notebooks (FREE GPU): https://www.kaggle.com"
echo "   - 30 hours/week of free GPU (P100)"
echo
echo "4. Request GPU quota increase:"
echo "   - Upgrade to paid account first"
echo "   - Go to: https://console.cloud.google.com/iam-admin/quotas"
echo "   - Request T4 or L4 GPUs in us-east1"
echo
echo "To connect to CPU instance:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
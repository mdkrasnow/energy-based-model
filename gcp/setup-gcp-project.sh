#!/bin/bash

# GCP Setup Script for Energy-Based Model Training
# RTX 2080 equivalent: NVIDIA T4 (16GB) or L4 (24GB)

set -e

echo "=== GCP Project Setup for GPU Training ==="
echo

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="us-east1"
ZONE="us-east1-b"  # us-east1-b has good GPU availability

if [ -z "$PROJECT_ID" ]; then
    echo "Enter your GCP Project ID:"
    read PROJECT_ID
fi

echo "Setting up project: $PROJECT_ID"
echo "Region: $REGION"
echo "Zone: $ZONE"
echo

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com

# Create a Cloud Storage bucket for data and checkpoints
BUCKET_NAME="${PROJECT_ID}-ebm-training"
echo
echo "Creating Cloud Storage bucket: $BUCKET_NAME"
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME/ || echo "Bucket may already exist"

# Check current GPU quota
echo
echo "Checking GPU quota in $REGION..."
gcloud compute project-info describe --project=$PROJECT_ID

echo
echo "=== IMPORTANT: GPU Quota Request ==="
echo "You need to request GPU quota for us-east1. Go to:"
echo "https://console.cloud.google.com/iam-admin/quotas"
echo
echo "Request the following quotas:"
echo "1. Search for 'GPUs (all regions)' and request at least 1"
echo "2. Search for 'NVIDIA T4 GPUs' in us-east1 and request at least 1"
echo "   OR search for 'NVIDIA L4 GPUs' in us-east1 and request at least 1"
echo
echo "Quota requests are usually approved within 15 minutes for small amounts."
echo
echo "Press Enter when quota is approved to continue..."
read

echo
echo "Setup complete! Next run: ./create-gpu-vm.sh"
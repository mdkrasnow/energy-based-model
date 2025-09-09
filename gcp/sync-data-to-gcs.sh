#!/bin/bash

# Upload data to Google Cloud Storage

set -e

PROJECT_ID="${GCP_PROJECT_ID:-}"
BUCKET_NAME="${PROJECT_ID}-ebm-training"
LOCAL_DATA_DIR="../data"

if [ -z "$PROJECT_ID" ]; then
    echo "Enter your GCP Project ID:"
    read PROJECT_ID
    BUCKET_NAME="${PROJECT_ID}-ebm-training"
fi

echo "Uploading data to GCS..."
echo "Bucket: gs://$BUCKET_NAME"
echo

# Create data directories in bucket
gsutil -m mkdir -p gs://$BUCKET_NAME/data/
gsutil -m mkdir -p gs://$BUCKET_NAME/checkpoints/
gsutil -m mkdir -p gs://$BUCKET_NAME/logs/

# Upload data files if they exist
if [ -d "$LOCAL_DATA_DIR" ] && [ "$(ls -A $LOCAL_DATA_DIR)" ]; then
    echo "Uploading data files..."
    gsutil -m cp -r $LOCAL_DATA_DIR/* gs://$BUCKET_NAME/data/
else
    echo "No local data files found. You can upload data later with:"
    echo "gsutil -m cp -r /path/to/data/* gs://$BUCKET_NAME/data/"
fi

echo
echo "GCS bucket structure created:"
echo "  gs://$BUCKET_NAME/data/       - Training data"
echo "  gs://$BUCKET_NAME/checkpoints/ - Model checkpoints"
echo "  gs://$BUCKET_NAME/logs/        - Training logs"
echo
echo "To mount on VM, run this command on the VM:"
echo "gcsfuse $BUCKET_NAME /gcs"
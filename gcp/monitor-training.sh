#!/bin/bash

# Monitor training progress from local machine

set -e

PROJECT_ID="${GCP_PROJECT_ID:-}"
ZONE="us-east1-b"
INSTANCE_NAME="ebm-training-vm"

if [ -z "$PROJECT_ID" ]; then
    echo "Enter your GCP Project ID:"
    read PROJECT_ID
fi

echo "=== Training Monitor ==="
echo "1. GPU Usage"
echo "2. Training Logs"
echo "3. System Resources"
echo "4. Stop VM (to save costs)"
echo
echo "Select option (1-4):"
read OPTION

case $OPTION in
    1)
        echo "Monitoring GPU usage (refresh every 2 seconds, Ctrl+C to exit)..."
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID \
            --command="watch -n 2 nvidia-smi"
        ;;
    2)
        echo "Tailing training logs..."
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID \
            --command="tail -f ~/projects/energy-based-model/logs/train.log"
        ;;
    3)
        echo "System resource monitor..."
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID \
            --command="htop"
        ;;
    4)
        echo "Stopping VM to save costs..."
        echo "WARNING: This will stop your training. Continue? (y/n)"
        read CONFIRM
        if [ "$CONFIRM" == "y" ]; then
            gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
            echo "VM stopped. To restart: gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
        fi
        ;;
    *)
        echo "Invalid option"
        ;;
esac
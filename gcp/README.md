# GCP GPU Training Setup

This directory contains scripts to set up and run your training on Google Cloud Platform with GPU support.

## ⚠️ IMPORTANT: GPU Quota Issues

**Google Cloud requires quota approval for GPUs, which is NOT available on free tier accounts.**

### Recommended Alternative: Google Colab (FREE GPU)
Use `colab_training.ipynb` to train with a free T4 GPU (16GB VRAM) without any quota requirements:
1. Upload `colab_training.ipynb` to [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → GPU → T4
3. Follow the notebook instructions

This gives you the same T4 GPU for FREE with no setup required!

## Hardware Specs
- **GPU**: NVIDIA T4 (16GB VRAM) - comparable to RTX 2080 Ti
- **Region**: us-east1-b
- **Machine**: n1-standard-8 (8 vCPUs, 30GB RAM)
- **Storage**: 200GB SSD

## Quick Start

### 1. Initial Setup (One-time)
```bash
cd gcp
chmod +x *.sh

# Set up GCP project and request GPU quota
./setup-gcp-project.sh

# Wait for GPU quota approval (usually 15 minutes)
```

### 2. Create GPU VM
```bash
# Create the GPU instance
./create-gpu-vm.sh

# Copy and run setup script on VM
gcloud compute scp setup-vm-environment.sh ebm-training-vm:~/ --zone=us-east1-b
gcloud compute ssh ebm-training-vm --zone=us-east1-b --command="bash ~/setup-vm-environment.sh"
```

### 3. Transfer Code and Data
```bash
# Sync your code to the VM
./sync-code-to-vm.sh

# Upload data to Cloud Storage (optional, for large datasets)
./sync-data-to-gcs.sh
```

### 4. Start Training
```bash
# Connect to VM
gcloud compute ssh ebm-training-vm --zone=us-east1-b

# On the VM:
cd ~/projects/energy-based-model
conda activate ebm

# Copy training script
cp ~/setup-vm-environment.sh ./run-training.sh
chmod +x run-training.sh

# Start training
./run-training.sh
# OR with tmux to keep running after disconnect:
tmux new -s training
./run-training.sh
# Detach with Ctrl+B, D
```

### 5. Monitor Progress
From your local machine:
```bash
./monitor-training.sh
```

## Cost Management

### Estimated Costs (us-east1)
- **T4 GPU**: ~$0.35/hour
- **VM (n1-standard-8)**: ~$0.38/hour
- **Storage (200GB)**: ~$0.04/hour
- **Total**: ~$0.77/hour (~$18.48/day)

### Cost-Saving Tips

1. **Use Spot VMs** (70% cheaper):
   ```bash
   # Add to create-gpu-vm.sh:
   --provisioning-model=SPOT \
   --instance-termination-action=STOP
   ```

2. **Stop VM when not training**:
   ```bash
   gcloud compute instances stop ebm-training-vm --zone=us-east1-b
   ```

3. **Use scheduled start/stop**:
   ```bash
   # Set up in GCP Console > Compute Engine > VM instances > Schedule
   ```

## Training Script Arguments

Modify `run-training.sh` to match your `train.py` arguments. Common patterns:

```bash
# If your train.py uses argparse:
python train.py \
    --data_path ./data \
    --batch_size 32 \
    --epochs 100

# If using config files:
python train.py --config configs/train_config.yaml

# With wandb logging:
wandb login YOUR_API_KEY
python train.py --wandb_project ebm-training
```

## Data Management

### Option 1: Small datasets (<10GB)
- Include in code sync: `./sync-code-to-vm.sh`

### Option 2: Large datasets  
- Upload to GCS: `./sync-data-to-gcs.sh`
- Mount on VM: `gcsfuse BUCKET_NAME /gcs`

### Option 3: Download on VM
```bash
# On the VM
cd ~/data
wget https://example.com/dataset.tar.gz
tar -xzf dataset.tar.gz
```

## Troubleshooting

### GPU not detected
```bash
# Check driver installation
nvidia-smi

# Reinstall if needed
sudo /opt/deeplearning/install-driver.sh
```

### Out of memory
- Reduce batch size in `run-training.sh`
- Enable gradient checkpointing
- Use mixed precision training

### Quota errors
- Check quota: https://console.cloud.google.com/iam-admin/quotas
- Request increase for "NVIDIA T4 GPUs" in us-east1

### Connection issues
```bash
# Reset SSH keys
gcloud compute ssh ebm-training-vm --zone=us-east1-b --troubleshoot
```

## Clean Up

To avoid charges when done:
```bash
# Delete VM
gcloud compute instances delete ebm-training-vm --zone=us-east1-b

# Delete storage bucket
gsutil rm -r gs://YOUR_PROJECT_ID-ebm-training/
```

## Alternative GPU Options

If T4 unavailable or need more power:

| GPU | VRAM | Cost/hr | Use Case |
|-----|------|---------|----------|
| T4 | 16GB | $0.35 | Best value, good for most training |
| L4 | 24GB | $0.65 | Larger models, faster than T4 |
| V100 | 16GB | $2.48 | Legacy, avoid if possible |
| A100 | 40GB | $3.67 | Large models, fast training |

Change GPU in `create-gpu-vm.sh`:
```bash
# For L4:
MACHINE_TYPE="g2-standard-8"
GPU_TYPE="nvidia-l4"

# For A100:
MACHINE_TYPE="a2-highgpu-1g"
# (A100 comes attached to a2 machines)
```
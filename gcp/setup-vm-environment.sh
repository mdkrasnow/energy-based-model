#!/bin/bash

# Setup script to run ON the GPU VM after creation
# This installs dependencies and prepares the training environment

set -e

echo "=== Setting up training environment on VM ==="
echo

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y git tmux htop nvtop

# Verify GPU is available
echo
echo "Checking GPU availability..."
nvidia-smi

# Create working directory
echo
echo "Creating working directory..."
mkdir -p ~/projects
cd ~/projects

# Clone the repository (you'll need to update this with your repo URL)
echo
echo "Setting up project directory..."
if [ ! -d "energy-based-model" ]; then
    # If you have a git repo:
    # git clone https://github.com/YOUR_USERNAME/energy-based-model.git
    
    # Otherwise, we'll sync from local:
    echo "Project will be synced from local machine using sync-code-to-vm.sh"
    mkdir -p energy-based-model
fi

cd energy-based-model

# Create conda environment with Python 3.10
echo
echo "Creating conda environment..."
conda create -n ebm python=3.10 -y

# Activate and install requirements
echo
echo "Installing Python dependencies..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ebm

# Install PyTorch with CUDA support (should already be available in base, but ensuring correct version)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install accelerate==1.10.1
pip install einops==0.8.1
pip install ema_pytorch==0.7.7
pip install ipdb==0.13.13
pip install ipython==8.12.3
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install tabulate==0.9.0
pip install tqdm==4.67.1
pip install wandb  # For experiment tracking (optional but recommended)

# Create directories for data and checkpoints
echo
echo "Creating data and checkpoint directories..."
mkdir -p ~/data
mkdir -p ~/checkpoints
mkdir -p ~/logs

# Setup Cloud Storage FUSE for easy data access
echo
echo "Installing Cloud Storage FUSE..."
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y gcsfuse

# Create mount points for GCS buckets
echo
echo "Creating GCS mount points..."
sudo mkdir -p /gcs
sudo chmod 755 /gcs

# Test GPU with PyTorch
echo
echo "Testing PyTorch GPU access..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo
echo "=== Setup Complete! ==="
echo
echo "Environment is ready. To activate:"
echo "conda activate ebm"
echo
echo "To mount GCS bucket:"
echo "gcsfuse YOUR_BUCKET_NAME /gcs"
echo
echo "Next: Run sync-code-to-vm.sh from your local machine"
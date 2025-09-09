#!/usr/bin/env python
"""
Wrapper script for training with more intuitive command-line arguments.
This provides a simpler interface while calling the original train.py.
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Training wrapper with simplified arguments')
    
    # Essential arguments
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Directory containing training data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (currently fixed in train.py)')
    parser.add_argument('--num_iterations', type=int, default=100000,
                        help='Number of training iterations (paper uses 100k)')
    parser.add_argument('--diffusion_steps', type=int, default=None,
                        help='Number of diffusion time steps (omit for IRED)')
    
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='inverse',
                        choices=['addition', 'inverse', 'lowrank', 'parents', 
                                'uncle', 'connectivity', 'planning', 'sudoku'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'mlp-reverse', 'sudoku', 'sudoku-latent', 
                                'sudoku-transformer', 'sudoku-reverse', 'gnn', 
                                'gnn-reverse', 'gnn-conv', 'gnn-conv-1d', 
                                'gnn-conv-1d-v2', 'gnn-conv-1d-v2-reverse'],
                        help='Model architecture to use')
    parser.add_argument('--rank', type=int, default=20,
                        help='Rank of matrix for certain datasets')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # IRED-specific flags
    parser.add_argument('--use_innerloop_opt', action='store_true',
                        help='Enable inner loop optimization (IRED)')
    parser.add_argument('--supervise_energy_landscape', action='store_true',
                        help='Supervise energy landscape (IRED)')
    parser.add_argument('--cond_mask', action='store_true',
                        help='Use conditional masking')
    
    # Additional features
    parser.add_argument('--use_anm', action='store_true',
                        help='Enable Adversarial Negative Mining')
    parser.add_argument('--anm_steps', type=int, default=10,
                        help='Number of ANM optimization steps')
    parser.add_argument('--anm_step_mult', type=float, default=1.0,
                        help='ANM step size multiplier')
    parser.add_argument('--anm_loss_weight', type=float, default=0.5,
                        help='Weight for energy loss when ANM is active')
    parser.add_argument('--anm_adaptive', action='store_true',
                        help='Enable timestep-aware ANM step sizing')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (not yet supported)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation before training')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Set environment variable for results folder
    os.environ['RESULTS_FOLDER'] = args.checkpoint_dir
    
    # Build command for original train.py
    cmd = [
        sys.executable, 'train.py',
        '--dataset', args.dataset,
        '--model', args.model,
        '--batch_size', str(args.batch_size),
        '--rank', str(args.rank),
        '--data-workers', str(args.num_workers),
    ]
    
    # Add diffusion steps only if specified (omit for IRED)
    if args.diffusion_steps is not None:
        cmd.extend(['--diffusion_steps', str(args.diffusion_steps)])
    
    # Add IRED-specific flags
    if args.use_innerloop_opt:
        cmd.extend(['--use-innerloop-opt', 'True'])
    
    if args.supervise_energy_landscape:
        cmd.extend(['--supervise-energy-landscape', 'True'])
    
    if args.cond_mask:
        cmd.extend(['--cond_mask', 'True'])
    
    # Add optional arguments
    if args.use_anm:
        cmd.extend(['--use-anm', 'True'])
        cmd.extend(['--anm-steps', str(args.anm_steps)])
        cmd.extend(['--anm-step-mult', str(args.anm_step_mult)])
        cmd.extend(['--anm-loss-weight', str(args.anm_loss_weight)])
        if args.anm_adaptive:
            cmd.extend(['--anm-adaptive', 'True'])
    
    if args.evaluate:
        cmd.append('--evaluate')
    
    if args.load_checkpoint:
        cmd.extend(['--load-milestone', args.load_checkpoint])
    
    # Check if CUDA is available
    if args.device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                args.device = 'cpu'
        except ImportError:
            print("Warning: PyTorch not installed properly")
    
    print(f"Starting training with settings:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print()
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    # Run the original train.py
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)

if __name__ == '__main__':
    main()
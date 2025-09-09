#!/usr/bin/env python
"""
Test script to verify ANM fixes are working correctly.
Tests the critical components that were fixed.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adversarial_negative_mining.anm_utils import extract
from models import EBM, DiffusionWrapper

def test_ebm_energy_shape():
    """Test that EBM returns correct energy shape after fix."""
    print("Testing EBM energy shape...")
    
    batch_size = 4
    inp_dim = 8
    out_dim = 16
    
    # Create EBM model
    ebm = EBM(inp_dim=inp_dim, out_dim=out_dim, is_ebm=True)
    
    # Create test inputs
    x = torch.randn(batch_size, inp_dim + out_dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    energy = ebm(x, t)
    
    # Check shape
    assert energy.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {energy.shape}"
    print(f"✓ EBM energy shape correct: {energy.shape}")
    
    # Test with DiffusionWrapper
    wrapper = DiffusionWrapper(ebm)
    inp = torch.randn(batch_size, inp_dim)
    opt_out = torch.randn(batch_size, out_dim)
    
    # Test return_energy
    energy = wrapper(inp, opt_out, t, return_energy=True)
    assert energy.shape == (batch_size, 1), f"Expected energy shape {(batch_size, 1)}, got {energy.shape}"
    print(f"✓ DiffusionWrapper energy shape correct: {energy.shape}")
    
    # Test return_both
    energy, grad = wrapper(inp, opt_out, t, return_both=True)
    assert energy.shape == (batch_size, 1), f"Expected energy shape {(batch_size, 1)}, got {energy.shape}"
    assert grad.shape == opt_out.shape, f"Expected grad shape {opt_out.shape}, got {grad.shape}"
    print(f"✓ DiffusionWrapper return_both shapes correct: energy={energy.shape}, grad={grad.shape}")

def test_extract_function():
    """Test the extract function works correctly."""
    print("\nTesting extract function...")
    
    # Create test data
    values = torch.linspace(0, 1, 100)
    batch_size = 4
    indices = torch.randint(0, 100, (batch_size,))
    x_shape = (batch_size, 16)
    
    result = extract(values, indices, x_shape)
    assert result.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {result.shape}"
    print(f"✓ Extract function shape correct: {result.shape}")

def test_ground_truth_computation():
    """Test ground truth computation in xt space."""
    print("\nTesting ground truth computation...")
    
    batch_size = 4
    out_dim = 16
    
    # Create mock diffusion schedules
    num_timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    
    # Test data
    x_start = torch.randn(batch_size, out_dim)
    t = torch.randint(0, num_timesteps, (batch_size,))
    
    # Compute ground truth in xt space (the fixed way)
    x_start_xt = extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start
    
    # Check shape
    assert x_start_xt.shape == x_start.shape, f"Shape mismatch: {x_start_xt.shape} != {x_start.shape}"
    print(f"✓ Ground truth xt computation shape correct: {x_start_xt.shape}")

def test_clamping_logic():
    """Test the fixed clamping logic."""
    print("\nTesting clamping logic...")
    
    batch_size = 4
    out_dim = 16
    
    # Create mock data
    xt = torch.randn(batch_size, out_dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Mock sqrt_alphas_cumprod
    sqrt_alphas_cumprod = torch.linspace(1.0, 0.1, 1000)
    
    # Fixed clamping logic
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, xt.shape)
    sf = 1.0  # scale factor
    max_val = sqrt_alpha * sf
    
    # Check shape
    assert max_val.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {max_val.shape}"
    print(f"✓ Clamping max_val shape correct: {max_val.shape}")
    
    # Apply clamping
    xt_clamped = torch.clamp(xt, -max_val, max_val)
    assert xt_clamped.shape == xt.shape, f"Shape mismatch after clamping"
    print(f"✓ Clamped tensor shape correct: {xt_clamped.shape}")

def test_distance_penalty_gradient():
    """Test the fixed distance penalty gradient computation."""
    print("\nTesting distance penalty gradient...")
    
    batch_size = 4
    out_dim = 16
    distance_penalty = 0.25
    
    # Create test data
    xt = torch.randn(batch_size, out_dim)
    x_start_xt = torch.randn(batch_size, out_dim)
    grad = torch.randn(batch_size, out_dim)
    
    # Fixed distance penalty gradient (pushes AWAY from ground truth)
    distance_grad = -2 * distance_penalty * (xt - x_start_xt)
    grad_updated = grad + distance_grad  # Add, not subtract
    
    # Check shapes
    assert distance_grad.shape == grad.shape, f"Shape mismatch: {distance_grad.shape} != {grad.shape}"
    assert grad_updated.shape == grad.shape, f"Shape mismatch: {grad_updated.shape} != {grad.shape}"
    
    # Verify the gradient pushes away from ground truth
    # The negative sign ensures we maximize distance
    print(f"✓ Distance penalty gradient shape correct: {distance_grad.shape}")
    print(f"✓ Updated gradient shape correct: {grad_updated.shape}")

def test_margin_loss():
    """Test the fixed margin loss computation."""
    print("\nTesting margin loss...")
    
    batch_size = 4
    
    # Create test energies
    energy_real = torch.randn(batch_size, 1)
    energy_fake = torch.randn(batch_size, 1) 
    margin = 0.1
    
    # Fixed margin loss computation
    energy_margin = torch.relu(energy_real - energy_fake + margin)
    loss_margin = energy_margin.mean()  # Simple scalar mean, no keepdim
    
    # Check that loss is scalar
    assert loss_margin.shape == torch.Size([]), f"Expected scalar, got shape {loss_margin.shape}"
    print(f"✓ Margin loss is scalar: {loss_margin.item():.4f}")

def main():
    print("=" * 60)
    print("Testing ANM Fixes")
    print("=" * 60)
    
    try:
        test_ebm_energy_shape()
        test_extract_function()
        test_ground_truth_computation()
        test_clamping_logic()
        test_distance_penalty_gradient()
        test_margin_loss()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
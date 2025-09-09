"""
Unit tests for ANM (Adversarial Negative Mining) tensor shapes.

Tests ensure that tensor shapes are correctly maintained throughout
the ANM pipeline at each step.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adversarial_negative_mining.anm_utils import (
    extract,
    AdversarialNegativeMiner,
    AdaptiveANM
)


class MockModel(nn.Module):
    """Mock model for testing that returns predictable energy and gradients."""
    
    def __init__(self, inp_dim=8, out_dim=16):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
    def forward(self, inp, x, t, return_energy=False, return_both=False):
        """
        Mock forward pass that maintains proper tensor shapes.
        
        Args:
            inp: Input tensor of shape (batch_size, inp_dim)
            x: Data tensor of shape (batch_size, out_dim)
            t: Timestep tensor of shape (batch_size,)
            return_energy: If True, return energy scalar
            return_both: If True, return (energy, gradient)
        """
        batch_size = x.shape[0]
        
        # Compute mock energy (scalar per sample)
        energy = torch.sum(x ** 2, dim=-1) * 0.01  # Shape: (batch_size,)
        
        # Compute mock gradient (same shape as x)
        grad = x * 0.1  # Shape: (batch_size, out_dim)
        
        if return_both:
            return energy, grad
        elif return_energy:
            return energy
        else:
            return grad


class MockDiffusion:
    """Mock diffusion model for testing."""
    
    def __init__(self, model, num_timesteps=1000, continuous=False, shortest_path=False):
        self.model = model
        self.num_timesteps = num_timesteps
        self.continuous = continuous
        self.shortest_path = shortest_path
        
        # Create mock diffusion schedules
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.opt_step_size = torch.ones(num_timesteps) * 0.01
        
    def q_sample(self, x_start, t, noise):
        """Add noise to x_start at timestep t."""
        batch_size = x_start.shape[0]
        
        # Extract values for the batch
        sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Add noise
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise


class TestExtractFunction:
    """Test the extract utility function for proper tensor shapes."""
    
    def test_extract_basic_shapes(self):
        """Test extract function with various input shapes."""
        # Create a 1D tensor of values
        values = torch.linspace(0, 1, 100)
        
        # Test with batch of indices
        batch_size = 8
        indices = torch.randint(0, 100, (batch_size,))
        x_shape = (batch_size, 16)  # (batch_size, feature_dim)
        
        result = extract(values, indices, x_shape)
        
        # Check output shape
        assert result.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {result.shape}"
        
    def test_extract_multi_dimensional(self):
        """Test extract with multi-dimensional data shapes."""
        values = torch.linspace(0, 1, 100)
        batch_size = 4
        indices = torch.randint(0, 100, (batch_size,))
        
        # Test with 3D shape (batch, height, width)
        x_shape = (batch_size, 32, 32)
        result = extract(values, indices, x_shape)
        assert result.shape == (batch_size, 1, 1), f"Expected shape {(batch_size, 1, 1)}, got {result.shape}"
        
        # Test with 4D shape (batch, channels, height, width)
        x_shape = (batch_size, 3, 32, 32)
        result = extract(values, indices, x_shape)
        assert result.shape == (batch_size, 1, 1, 1), f"Expected shape {(batch_size, 1, 1, 1)}, got {result.shape}"


class TestANMInitialization:
    """Test ANM initialization and shape consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.inp_dim = 8
        self.out_dim = 16
        self.model = MockModel(self.inp_dim, self.out_dim)
        self.diffusion = MockDiffusion(self.model)
        
    def test_anm_initialization(self):
        """Test that ANM initializes correctly."""
        anm = AdversarialNegativeMiner(
            diffusion=self.diffusion,
            search_steps=5,
            step_size_multiplier=1.0,
            init_strategy="noisy",
            track_stats=True
        )
        
        assert anm.search_steps == 5
        assert anm.step_size_multiplier == 1.0
        assert anm.init_strategy == "noisy"
        assert anm.track_stats == True
        
    def test_different_init_strategies(self):
        """Test different initialization strategies maintain proper shapes."""
        strategies = ["noisy", "random", "perturbed"]
        
        for strategy in strategies:
            anm = AdversarialNegativeMiner(
                diffusion=self.diffusion,
                search_steps=3,
                init_strategy=strategy
            )
            
            # Create test data
            inp = torch.randn(self.batch_size, self.inp_dim)
            x_start = torch.randn(self.batch_size, self.out_dim)
            t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
            
            # Mine hard negatives
            xt_neg, stats = anm.mine_hard_negatives(inp, x_start, t)
            
            # Check output shape
            assert xt_neg.shape == x_start.shape, \
                f"Strategy {strategy}: Expected shape {x_start.shape}, got {xt_neg.shape}"


class TestMineHardNegatives:
    """Test the mine_hard_negatives method for shape consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.inp_dim = 8
        self.out_dim = 16
        self.model = MockModel(self.inp_dim, self.out_dim)
        self.diffusion = MockDiffusion(self.model)
        self.anm = AdversarialNegativeMiner(
            diffusion=self.diffusion,
            search_steps=5,
            step_size_multiplier=1.0,
            track_stats=True
        )
        
    def test_basic_mining(self):
        """Test basic hard negative mining maintains shapes."""
        inp = torch.randn(self.batch_size, self.inp_dim)
        x_start = torch.randn(self.batch_size, self.out_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
        
        xt_neg, stats = self.anm.mine_hard_negatives(inp, x_start, t)
        
        # Check shapes
        assert xt_neg.shape == x_start.shape, f"Expected shape {x_start.shape}, got {xt_neg.shape}"
        assert isinstance(stats, dict), "Stats should be a dictionary"
        
        # Check stats contain expected keys
        expected_keys = [
            "anm_initial_energy",
            "anm_final_energy",
            "anm_energy_reduction",
            "anm_optimization_movement",
            "anm_steps",
            "anm_step_mult"
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key {key} in stats"
            
    def test_mining_with_mask(self):
        """Test mining with mask maintains proper shapes."""
        inp = torch.randn(self.batch_size, self.inp_dim)
        x_start = torch.randn(self.batch_size, self.out_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
        
        # Create a binary mask
        mask = torch.bernoulli(torch.ones(self.batch_size, self.out_dim) * 0.3)
        
        xt_neg, stats = self.anm.mine_hard_negatives(inp, x_start, t, mask=mask)
        
        # Check shapes
        assert xt_neg.shape == x_start.shape, f"Expected shape {x_start.shape}, got {xt_neg.shape}"
        
    def test_mining_with_provided_noise(self):
        """Test mining with provided noise maintains shapes."""
        inp = torch.randn(self.batch_size, self.inp_dim)
        x_start = torch.randn(self.batch_size, self.out_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
        noise = torch.randn_like(x_start)
        
        xt_neg, stats = self.anm.mine_hard_negatives(inp, x_start, t, noise=noise)
        
        # Check shapes
        assert xt_neg.shape == x_start.shape, f"Expected shape {x_start.shape}, got {xt_neg.shape}"
        
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        batch_sizes = [1, 2, 8, 16]
        
        for bs in batch_sizes:
            inp = torch.randn(bs, self.inp_dim)
            x_start = torch.randn(bs, self.out_dim)
            t = torch.randint(0, self.diffusion.num_timesteps, (bs,))
            
            xt_neg, stats = self.anm.mine_hard_negatives(inp, x_start, t)
            
            assert xt_neg.shape == (bs, self.out_dim), \
                f"Batch size {bs}: Expected shape {(bs, self.out_dim)}, got {xt_neg.shape}"


class TestOptimizeAdversarial:
    """Test the _optimize_adversarial method for shape consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.inp_dim = 8
        self.out_dim = 16
        self.model = MockModel(self.inp_dim, self.out_dim)
        self.diffusion = MockDiffusion(self.model)
        self.anm = AdversarialNegativeMiner(
            diffusion=self.diffusion,
            search_steps=5,
            step_size_multiplier=1.0
        )
        
    def test_optimize_shapes(self):
        """Test that optimization maintains tensor shapes."""
        inp = torch.randn(self.batch_size, self.inp_dim)
        xt = torch.randn(self.batch_size, self.out_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
        x_start_xt = torch.randn(self.batch_size, self.out_dim)  # Ground truth in xt space
        
        xt_optimized = self.anm._optimize_adversarial(inp, xt, x_start_xt, t, mask=None, data_cond=None)
        
        # Check shape preservation
        assert xt_optimized.shape == xt.shape, f"Expected shape {xt.shape}, got {xt_optimized.shape}"
        
        # Check that output is detached
        assert not xt_optimized.requires_grad, "Output should be detached"
        
    def test_optimize_with_mask(self):
        """Test optimization with mask constraint."""
        inp = torch.randn(self.batch_size, self.inp_dim)
        xt = torch.randn(self.batch_size, self.out_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
        x_start_xt = torch.randn(self.batch_size, self.out_dim)  # Ground truth in xt space
        mask = torch.bernoulli(torch.ones(self.batch_size, self.out_dim) * 0.3)
        data_cond = torch.randn(self.batch_size, self.out_dim)
        
        xt_optimized = self.anm._optimize_adversarial(inp, xt, x_start_xt, t, mask=mask, data_cond=data_cond)
        
        # Check shape preservation
        assert xt_optimized.shape == xt.shape, f"Expected shape {xt.shape}, got {xt_optimized.shape}"
        
    def test_gradient_computation_shapes(self):
        """Test that gradient computation maintains proper shapes."""
        inp = torch.randn(self.batch_size, self.inp_dim)
        xt = torch.randn(self.batch_size, self.out_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
        
        # Manually test one step of optimization
        xt_test = xt.detach().requires_grad_(True)
        
        with torch.enable_grad():
            energy, grad = self.diffusion.model(inp, xt_test, t, return_both=True)
            
            # Check energy shape
            assert energy.shape == (self.batch_size,) or energy.shape == (self.batch_size, 1), \
                f"Energy shape {energy.shape} unexpected"
            
            # Check gradient shape
            assert grad.shape == xt.shape, f"Gradient shape {grad.shape} != input shape {xt.shape}"


class TestMultipleInitializations:
    """Test the mine_with_multiple_inits method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.inp_dim = 8
        self.out_dim = 16
        self.model = MockModel(self.inp_dim, self.out_dim)
        self.diffusion = MockDiffusion(self.model)
        self.anm = AdversarialNegativeMiner(
            diffusion=self.diffusion,
            search_steps=3,
            track_stats=True
        )
        
    def test_multiple_inits_shapes(self):
        """Test multiple initializations maintain shapes."""
        inp = torch.randn(self.batch_size, self.inp_dim)
        x_start = torch.randn(self.batch_size, self.out_dim)
        t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
        
        xt_neg, stats = self.anm.mine_with_multiple_inits(
            inp, x_start, t, n_inits=3
        )
        
        # Check shapes
        assert xt_neg.shape == x_start.shape, f"Expected shape {x_start.shape}, got {xt_neg.shape}"
        assert "anm_n_inits" in stats, "Stats should contain anm_n_inits"
        assert stats["anm_n_inits"] <= 3, "Number of inits should be <= 3"


class TestAdaptiveANM:
    """Test the AdaptiveANM class for shape consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.inp_dim = 8
        self.out_dim = 16
        self.model = MockModel(self.inp_dim, self.out_dim)
        self.diffusion = MockDiffusion(self.model)
        
    def test_adaptive_anm_shapes(self):
        """Test AdaptiveANM maintains proper shapes."""
        schedules = ["linear", "cosine", "sqrt"]
        
        for schedule in schedules:
            anm = AdaptiveANM(
                diffusion=self.diffusion,
                search_steps=3,
                adaptive_schedule=schedule,
                track_stats=True
            )
            
            inp = torch.randn(self.batch_size, self.inp_dim)
            x_start = torch.randn(self.batch_size, self.out_dim)
            t = torch.randint(0, self.diffusion.num_timesteps, (self.batch_size,))
            
            xt_neg, stats = anm.mine_hard_negatives(inp, x_start, t)
            
            # Check shapes
            assert xt_neg.shape == x_start.shape, \
                f"Schedule {schedule}: Expected shape {x_start.shape}, got {xt_neg.shape}"
            
    def test_adaptive_multiplier_computation(self):
        """Test adaptive multiplier computation."""
        anm = AdaptiveANM(
            diffusion=self.diffusion,
            search_steps=3,
            step_size_multiplier=2.0,
            adaptive_schedule="linear"
        )
        
        # Test with different timesteps
        timesteps = [0, 250, 500, 750, 999]
        
        for t_val in timesteps:
            t = torch.tensor([t_val])
            multiplier = anm._get_adaptive_multiplier(t)
            
            # Check that multiplier is a scalar
            assert isinstance(multiplier, float), f"Multiplier should be float, got {type(multiplier)}"
            
            # Check that multiplier is positive
            assert multiplier > 0, f"Multiplier should be positive, got {multiplier}"


class TestEdgeCases:
    """Test edge cases and potential shape mismatches."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = MockModel(8, 16)
        self.diffusion = MockDiffusion(self.model)
        self.anm = AdversarialNegativeMiner(
            diffusion=self.diffusion,
            search_steps=2
        )
        
    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        inp = torch.randn(1, 8)
        x_start = torch.randn(1, 16)
        t = torch.tensor([500])
        
        xt_neg, stats = self.anm.mine_hard_negatives(inp, x_start, t)
        
        assert xt_neg.shape == (1, 16), f"Expected shape (1, 16), got {xt_neg.shape}"
        
    def test_varying_timesteps(self):
        """Test with different timesteps in the same batch."""
        batch_size = 4
        inp = torch.randn(batch_size, 8)
        x_start = torch.randn(batch_size, 16)
        # Different timesteps for each sample
        t = torch.tensor([100, 300, 500, 700])
        
        xt_neg, stats = self.anm.mine_hard_negatives(inp, x_start, t)
        
        assert xt_neg.shape == x_start.shape, f"Expected shape {x_start.shape}, got {xt_neg.shape}"
        
    def test_zero_search_steps(self):
        """Test with zero search steps."""
        anm_zero = AdversarialNegativeMiner(
            diffusion=self.diffusion,
            search_steps=0
        )
        
        inp = torch.randn(2, 8)
        x_start = torch.randn(2, 16)
        t = torch.tensor([500, 600])
        
        xt_neg, stats = anm_zero.mine_hard_negatives(inp, x_start, t)
        
        # Should return initialized xt without optimization
        assert xt_neg.shape == x_start.shape, f"Expected shape {x_start.shape}, got {xt_neg.shape}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
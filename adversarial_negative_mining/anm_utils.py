"""
Adversarial Negative Mining (ANM) utilities for Energy-Based Diffusion Models.

This module provides adversarial negative mining to generate hard negative samples
that improve the energy landscape supervision during diffusion model training.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from contextlib import nullcontext


def extract(a, t, x_shape):
    """Extract values from a 1-D tensor for a batch of indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class AdversarialNegativeMiner:
    """
    Mines hard negative samples by optimizing in the noisy latent space (xt).
    
    Key improvements over naive negative sampling:
    - Operates directly in xt space without re-noising
    - Maintains mask consistency throughout optimization
    - Uses proper energy/gradient computation
    - Returns detailed optimization statistics
    - Implements proper projection to feasible set
    - Includes distance penalty to maintain separation from ground truth
    """
    
    def __init__(
        self,
        diffusion,
        search_steps: int = 10,
        step_size_multiplier: float = 1.0,
        init_strategy: str = "noisy",
        track_stats: bool = True,
        clamp_range: Optional[Tuple[float, float]] = None,
        projection_radius: float = 0.2,
        projection_norm: str = "l2",
        distance_penalty: float = 0.25,
        use_ema_model: bool = False,
        ema_model = None,
    ):
        """
        Initialize the Adversarial Negative Miner.
        
        Args:
            diffusion: GaussianDiffusion1D instance
            search_steps: Number of optimization steps for mining
            step_size_multiplier: Multiplier for the step size
            init_strategy: How to initialize adversarial samples ("noisy", "random", "perturbed")
            track_stats: Whether to track optimization statistics
            clamp_range: Optional range for clamping adversarial samples
            projection_radius: Radius of feasible set around ground truth
            projection_norm: Norm for projection ("l2" or "linf")
            distance_penalty: Weight for distance penalty term
            use_ema_model: Whether to use EMA model for mining
            ema_model: EMA model instance if available
        """
        self.diffusion = diffusion
        self.search_steps = search_steps
        self.step_size_multiplier = step_size_multiplier
        self.init_strategy = init_strategy
        self.track_stats = track_stats
        self.clamp_range = clamp_range or (-2.0, 2.0)
        self.projection_radius = projection_radius
        self.projection_norm = projection_norm
        self.distance_penalty = distance_penalty
        self.use_ema_model = use_ema_model
        self.ema_model = ema_model
        
    def mine_hard_negatives(
        self,
        inp: torch.Tensor,
        x_start: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Mine hard negative samples through adversarial optimization.
        
        This method:
        1. Initializes xt from the data distribution
        2. Optimizes xt to minimize energy (making it a harder negative)
        3. Returns the optimized xt directly (no re-noising)
        
        Args:
            inp: Input conditioning information
            x_start: Clean data samples (x0)
            t: Timestep tensor
            mask: Optional mask for conditional generation
            noise: Optional noise for initialization
            
        Returns:
            Tuple of:
            - xt_neg: Adversarial negative samples in xt space
            - stats: Dictionary of optimization statistics
        """
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Store ground truth in xt space for distance penalty
        x_start_xt = self.diffusion.q_sample(x_start=x_start, t=t, noise=torch.zeros_like(noise))
            
        # Initialize xt based on strategy
        if self.init_strategy == "noisy":
            # Standard noisy initialization from x_start
            xt_init = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
        elif self.init_strategy == "random":
            # Random initialization in xt space
            xt_init = torch.randn_like(x_start) * extract(
                self.diffusion.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
        elif self.init_strategy == "perturbed":
            # Perturbed initialization: mix clean signal with stronger noise
            xt_clean = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
            perturbation = torch.randn_like(x_start) * 0.5
            xt_init = xt_clean + perturbation * extract(
                self.diffusion.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")
            
        # Handle mask for conditional generation
        data_cond = None
        if mask is not None:
            # Compute the masked condition at timestep t
            data_cond = self.diffusion.q_sample(
                x_start=x_start, t=t, noise=torch.zeros_like(noise)
            )
            # Apply mask to initialization
            xt_init = xt_init * (1 - mask) + data_cond * mask
            
        # Track initial energy if requested
        initial_energy = None
        if self.track_stats:
            with torch.no_grad():
                # Use the model's forward method with return_energy=True
                initial_energy = self.diffusion.model(
                    inp, xt_init, t, return_energy=True
                )
                if initial_energy.dim() > 1:
                    initial_energy = initial_energy.squeeze(-1)
                initial_energy = initial_energy.mean().item()
                
        # Optimize in xt space to find hard negatives
        xt_neg = self._optimize_adversarial(
            inp=inp,
            xt=xt_init,
            x_start_xt=x_start_xt,
            t=t,
            mask=mask,
            data_cond=data_cond,
        )
        
        # Compute statistics
        stats = {}
        if self.track_stats:
            with torch.no_grad():
                # Ensure mask consistency for final energy evaluation
                xt_eval = xt_neg
                if mask is not None:
                    xt_eval = xt_neg * (1 - mask) + data_cond * mask
                    
                final_energy = self.diffusion.model(
                    inp, xt_eval, t, return_energy=True
                )
                if final_energy.dim() > 1:
                    final_energy = final_energy.squeeze(-1)
                final_energy = final_energy.mean().item()
                
                stats = {
                    "anm_initial_energy": initial_energy,
                    "anm_final_energy": final_energy,
                    "anm_energy_reduction": initial_energy - final_energy,
                    "anm_optimization_movement": (xt_neg - xt_init).pow(2).mean().item(),
                    "anm_steps": self.search_steps,
                    "anm_step_mult": self.step_size_multiplier,
                }
                
        return xt_neg.detach(), stats
    
    def _project_to_feasible_set(
        self,
        xt: torch.Tensor,
        x_anchor: torch.Tensor,
        radius: float,
        norm: str = "l2",
    ) -> torch.Tensor:
        """Project xt to feasible set (ball around x_anchor)."""
        if norm == "l2":
            # L2 projection
            diff = xt - x_anchor
            norm_val = torch.norm(diff, p=2, dim=-1, keepdim=True)
            scale = torch.clamp(radius / (norm_val + 1e-12), max=1.0)
            return x_anchor + diff * scale
        elif norm == "linf":
            # L-infinity projection
            return torch.clamp(xt, min=x_anchor - radius, max=x_anchor + radius)
        else:
            raise ValueError(f"Unknown norm: {norm}")
    
    def _optimize_adversarial(
        self,
        inp: torch.Tensor,
        xt: torch.Tensor,
        x_start_xt: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor],
        data_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Perform the adversarial optimization loop.
        
        This is a cleaner reimplementation of opt_step that:
        - Properly uses step_size_multiplier
        - Maintains mask consistency
        - Works correctly with energy/gradient computation
        - Includes projection to feasible set
        - Adds distance penalty
        """
        xt = xt.detach()
        
        # Get the base step size from the diffusion schedule
        base_step_size = extract(self.diffusion.opt_step_size, t, xt.shape)
        
        # Apply our multiplier
        step_size = base_step_size * self.step_size_multiplier
        
        # Use EMA model if available and requested
        model_to_use = self.ema_model if (self.use_ema_model and self.ema_model is not None) else self.diffusion.model
        
        with torch.enable_grad():
            for step_idx in range(self.search_steps):
                xt = xt.detach().requires_grad_(True)
                
                # Get energy and gradient using the model's return_both interface
                # The DiffusionWrapper returns (energy, grad) when return_both=True
                energy, grad = model_to_use(inp, xt, t, return_both=True)
                
                # Include distance penalty to keep negatives away from ground truth
                if self.distance_penalty > 0:
                    with torch.no_grad():
                        # Add penalty gradient to push away from ground truth
                        distance_grad = 2 * self.distance_penalty * (xt - x_start_xt)
                        grad = grad - distance_grad
                
                # Take a gradient step to minimize energy (find harder negatives)
                with torch.no_grad():
                    xt_new = xt - step_size * grad
                
                # Project to feasible set (ball around ground truth in xt space)
                xt_new = self._project_to_feasible_set(
                    xt_new, x_start_xt, self.projection_radius, self.projection_norm
                )
                
                # Enforce mask constraint if applicable
                if mask is not None and data_cond is not None:
                    xt_new = xt_new * (1 - mask) + data_cond * mask
                    
                # Clamp to valid range based on noise level
                # Use the noise schedule to determine valid range
                if self.diffusion.continuous:
                    sf = 2.0
                elif hasattr(self.diffusion, 'shortest_path') and self.diffusion.shortest_path:
                    sf = 0.1
                else:
                    sf = 1.0
                    
                # Get max value for clamping (use first element of batch)
                sqrt_alpha = extract(self.diffusion.sqrt_alphas_cumprod, t, xt.shape)
                max_val = sqrt_alpha.view(sqrt_alpha.shape[0], -1).max(dim=1, keepdim=True)[0] * sf
                max_val = max_val.view(-1, *([1] * (len(xt.shape) - 1)))
                xt_new = torch.clamp(xt_new, -max_val, max_val)
                
                # Check if energy decreased (for adaptive steps)
                if step_idx < self.search_steps - 1:  # Skip on last iteration
                    with torch.no_grad():
                        energy_new = model_to_use(inp, xt_new, t, return_energy=True)
                        
                        # Handle energy tensor shape
                        if energy_new.dim() > 1:
                            energy_new = energy_new.squeeze(-1)
                        if energy.dim() > 1:
                            energy = energy.squeeze(-1)
                        
                        # Only accept the step if energy decreased
                        bad_step = (energy_new > energy)
                        
                        # Properly broadcast bad_step to match xt dimensions
                        bad_step = bad_step.view(-1, *([1] * (len(xt.shape) - 1)))
                        xt_new = torch.where(bad_step, xt, xt_new)
                        
                # Update xt for next iteration
                xt = xt_new.detach()
                
        return xt.detach()
    
    def mine_with_multiple_inits(
        self,
        inp: torch.Tensor,
        x_start: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_inits: int = 3,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Mine hard negatives with multiple random initializations.
        
        This can help find better adversarial samples by exploring
        different regions of the energy landscape.
        
        Args:
            inp: Input conditioning
            x_start: Clean samples
            t: Timestep
            mask: Optional mask
            n_inits: Number of random initializations to try
            
        Returns:
            The best adversarial samples (lowest energy) and stats
        """
        best_xt = None
        best_energy = float('inf')
        best_stats = {}
        
        for i in range(n_inits):
            # Use different noise for each initialization
            noise = torch.randn_like(x_start)
            
            xt_neg, stats = self.mine_hard_negatives(
                inp=inp,
                x_start=x_start,
                t=t,
                mask=mask,
                noise=noise,
            )
            
            # Evaluate final energy
            with torch.no_grad():
                if mask is not None:
                    data_cond = self.diffusion.q_sample(
                        x_start=x_start, t=t, noise=torch.zeros_like(noise)
                    )
                    xt_eval = xt_neg * (1 - mask) + data_cond * mask
                else:
                    xt_eval = xt_neg
                    
                energy = self.diffusion.model(
                    inp, xt_eval, t, return_energy=True
                )
                if energy.dim() > 1:
                    energy = energy.squeeze(-1)
                energy = energy.mean().item()
                
            # Keep the best (lowest energy) adversarial samples
            if energy < best_energy:
                best_energy = energy
                best_xt = xt_neg
                best_stats = stats
                best_stats['anm_n_inits'] = i + 1
                
        return best_xt, best_stats


class AdaptiveANM(AdversarialNegativeMiner):
    """
    Adaptive variant that adjusts step size based on timestep.
    
    Uses larger steps at early (noisy) timesteps and smaller steps
    at later (clean) timesteps for more stable optimization.
    """
    
    def __init__(
        self,
        diffusion,
        search_steps: int = 10,
        step_size_multiplier: float = 1.0,
        adaptive_schedule: str = "linear",
        **kwargs
    ):
        super().__init__(diffusion, search_steps, step_size_multiplier, **kwargs)
        self.adaptive_schedule = adaptive_schedule
        
    def _get_adaptive_multiplier(self, t: torch.Tensor) -> float:
        """
        Compute an adaptive multiplier based on the timestep.
        
        Returns larger multipliers for earlier timesteps (more noise)
        and smaller multipliers for later timesteps (less noise).
        """
        # Normalize t to [0, 1] range
        t_normalized = t.float() / self.diffusion.num_timesteps
        
        if self.adaptive_schedule == "linear":
            # Linear decay from 2x to 0.5x
            multiplier = 2.0 - 1.5 * (1 - t_normalized.mean().item())
        elif self.adaptive_schedule == "cosine":
            # Cosine schedule
            import math
            multiplier = 1.0 + math.cos(math.pi * (1 - t_normalized.mean().item()))
        elif self.adaptive_schedule == "sqrt":
            # Square root schedule (less aggressive)
            multiplier = 1.0 + torch.sqrt(t_normalized).mean().item()
        else:
            multiplier = 1.0
            
        return multiplier * self.step_size_multiplier
    
    def _optimize_adversarial(
        self,
        inp: torch.Tensor,
        xt: torch.Tensor,
        x_start_xt: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor],
        data_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Override to use adaptive step sizing."""
        # Get adaptive multiplier based on timestep
        adaptive_mult = self._get_adaptive_multiplier(t)
        
        # Temporarily update multiplier
        original_mult = self.step_size_multiplier
        self.step_size_multiplier = adaptive_mult
        
        # Run optimization with adaptive step size
        result = super()._optimize_adversarial(inp, xt, x_start_xt, t, mask, data_cond)
        
        # Restore original multiplier
        self.step_size_multiplier = original_mult
        
        return result
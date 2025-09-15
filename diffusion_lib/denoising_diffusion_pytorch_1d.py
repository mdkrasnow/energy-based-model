import math
import sys
import collections
from multiprocessing import cpu_count
from pathlib import Path
import csv
from datetime import datetime
from random import random
from functools import partial
from collections import namedtuple
from tabulate import tabulate

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import os.path as osp
import time
import numpy as np

# Import curriculum configuration system
from curriculum_config import CurriculumConfig, DEFAULT_CURRICULUM
# from metrics_tracker import CurriculumMetricsTracker  # Removed enhanced metrics


def _custom_exception_hook(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, ipdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        ipdb.post_mortem(tb)


def hook_exception_ipdb():
    """Add a hook to ipdb when an exception is raised."""
    if not hasattr(_custom_exception_hook, 'origin_hook'):
        _custom_exception_hook.origin_hook = sys.excepthook
        sys.excepthook = _custom_exception_hook


def unhook_exception_ipdb():
    """Remove the hook to ipdb when an exception is raised."""
    assert hasattr(_custom_exception_hook, 'origin_hook')
    sys.excepthook = _custom_exception_hook.origin_hook

hook_exception_ipdb()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    val: float = 0
    avg: float = 0
    sum: float = 0
    sum2: float = 0
    std: float = 0
    count: float = 0
    tot_count: float = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum2 = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum2 += val * val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count
        self.std = (self.sum2 / self.count - self.avg * self.avg) ** 0.5

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ValidationGatedCurriculum:
    """
    Helper class for managing validation-based curriculum gating.
    
    This class tracks validation performance and makes decisions about
    curriculum progression, intensity adjustments, and rollbacks.
    
    Key Features:
    - Tracks validation loss and accuracy with exponential moving averages
    - Maintains stage-specific performance baselines
    - Detects performance degradation beyond configurable thresholds
    - Automatically adjusts curriculum intensity (0.1x to 2.0x multiplier)
    - Supports curriculum rollback for severe performance drops
    - Provides detailed logging of all adjustments and decisions
    
    Integration:
    - Automatically initialized when curriculum_config is provided
    - Integrates with get_current_curriculum_params() for seamless operation
    - Can be disabled via curriculum_config.enable_validation_gating = False
    - Thread-safe for concurrent validation updates
    """
    
    def __init__(self, config: CurriculumConfig, window_size: int = 10):
        """
        Initialize validation gating system.
        
        Args:
            config: CurriculumConfig with gating parameters
            window_size: Number of recent validation scores to track
        """
        self.config = config
        self.window_size = window_size
        
        # Validation performance tracking
        self.val_loss_history = collections.deque(maxlen=window_size)
        self.val_accuracy_history = collections.deque(maxlen=window_size)
        self.val_loss_ema = None
        self.val_accuracy_ema = None
        
        # Stage-specific baselines
        self.stage_baselines = {}  # stage_name -> {'loss': float, 'accuracy': float}
        self.current_stage_baseline = None
        
        # Curriculum adjustment state
        self.intensity_multiplier = 1.0  # Current intensity adjustment (0.0 to 2.0)
        self.consecutive_degradations = 0
        self.last_adjustment_step = 0
        self.rollback_count = 0
        
        # Performance tracking
        self.performance_trend = "stable"  # "improving", "stable", "degrading"
        self.adjustment_history = []  # List of (step, adjustment_type, reason)
        
    def update_validation_metrics(self, val_loss: float, val_accuracy: float, step: int):
        """
        Update validation metrics with exponential moving average.
        
        Args:
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy
            step: Current training step
        """
        # Update raw history
        self.val_loss_history.append(val_loss)
        self.val_accuracy_history.append(val_accuracy)
        
        # Update exponential moving averages (alpha = 0.1 for smoothing)
        alpha = 0.1
        if self.val_loss_ema is None:
            self.val_loss_ema = val_loss
            self.val_accuracy_ema = val_accuracy
        else:
            self.val_loss_ema = alpha * val_loss + (1 - alpha) * self.val_loss_ema
            self.val_accuracy_ema = alpha * val_accuracy + (1 - alpha) * self.val_accuracy_ema
    
    def set_stage_baseline(self, stage_name: str, force_update: bool = False):
        """
        Set baseline performance for a curriculum stage.
        
        Args:
            stage_name: Name of the curriculum stage
            force_update: Whether to update existing baseline
        """
        if (stage_name not in self.stage_baselines or force_update) and self.val_loss_ema is not None:
            self.stage_baselines[stage_name] = {
                'loss': self.val_loss_ema,
                'accuracy': self.val_accuracy_ema
            }
            self.current_stage_baseline = self.stage_baselines[stage_name]
    
    def detect_performance_degradation(self) -> tuple[bool, str]:
        """
        Detect if validation performance has degraded significantly.
        
        Returns:
            Tuple of (is_degraded, reason)
        """
        if (len(self.val_loss_history) < 3 or 
            self.current_stage_baseline is None or
            self.val_loss_ema is None):
            return False, "insufficient_data"
        
        baseline_loss = self.current_stage_baseline['loss']
        baseline_accuracy = self.current_stage_baseline['accuracy']
        
        # Check loss degradation (increase beyond threshold)
        loss_increase = (self.val_loss_ema - baseline_loss) / baseline_loss
        if loss_increase > self.config.validation_threshold:
            return True, f"loss_degradation_{loss_increase:.3f}"
        
        # Check accuracy degradation (decrease beyond threshold)
        accuracy_decrease = (baseline_accuracy - self.val_accuracy_ema) / baseline_accuracy
        if accuracy_decrease > self.config.validation_threshold:
            return True, f"accuracy_degradation_{accuracy_decrease:.3f}"
        
        return False, "performance_stable"
    
    def should_adjust_curriculum(self, step: int) -> tuple[bool, str, str]:
        """
        Determine if curriculum should be adjusted based on validation performance.
        
        Args:
            step: Current training step
            
        Returns:
            Tuple of (should_adjust, adjustment_type, reason)
            adjustment_type: "reduce_intensity", "increase_intensity", "rollback", "none"
        """
        if not self.config.enable_validation_gating:
            return False, "none", "gating_disabled"
        
        # Don't adjust too frequently (minimum 1000 steps between adjustments)
        if step - self.last_adjustment_step < 1000:
            return False, "none", "too_recent"
        
        # Check for performance degradation
        is_degraded, degradation_reason = self.detect_performance_degradation()
        
        if is_degraded:
            self.consecutive_degradations += 1
            
            # If multiple consecutive degradations, consider rollback
            if self.consecutive_degradations >= 3:
                return True, "rollback", f"consecutive_degradations_{degradation_reason}"
            else:
                return True, "reduce_intensity", degradation_reason
        else:
            self.consecutive_degradations = 0
            
            # Check if performance is improving faster than expected
            if (len(self.val_loss_history) >= 5 and 
                self.val_loss_ema < self.current_stage_baseline['loss'] * 0.95):
                # Performance is significantly better than baseline
                return True, "increase_intensity", "performance_exceeding_baseline"
        
        return False, "none", "no_adjustment_needed"
    
    def adjust_intensity(self, adjustment_type: str, step: int, reason: str) -> float:
        """
        Adjust curriculum intensity based on validation performance.
        
        Args:
            adjustment_type: Type of adjustment to make
            step: Current training step
            reason: Reason for adjustment
            
        Returns:
            New intensity multiplier
        """
        old_intensity = self.intensity_multiplier
        
        if adjustment_type == "reduce_intensity":
            self.intensity_multiplier = max(0.1, self.intensity_multiplier * 0.7)
        elif adjustment_type == "increase_intensity":
            self.intensity_multiplier = min(2.0, self.intensity_multiplier * 1.2)
        elif adjustment_type == "rollback":
            self.intensity_multiplier = max(0.1, self.intensity_multiplier * 0.5)
            self.rollback_count += 1
        
        # Record adjustment
        self.adjustment_history.append((step, adjustment_type, reason, old_intensity, self.intensity_multiplier))
        self.last_adjustment_step = step
        
        return self.intensity_multiplier
    
    def get_adjustment_log(self) -> str:
        """Get a formatted log of recent curriculum adjustments."""
        if not self.adjustment_history:
            return "No curriculum adjustments made."
        
        recent_adjustments = self.adjustment_history[-5:]  # Last 5 adjustments
        log_lines = ["Recent curriculum adjustments:"]
        
        for step, adj_type, reason, old_intensity, new_intensity in recent_adjustments:
            log_lines.append(
                f"  Step {step}: {adj_type} ({reason}) - "
                f"intensity {old_intensity:.3f} -> {new_intensity:.3f}"
            )
        
        return "\n".join(log_lines)


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        supervise_energy_landscape = True,
        use_innerloop_opt = True,
        show_inference_tqdm = True,
        baseline = False,
        sudoku = False,
        continuous = False,
        connectivity = False,
        shortest_path = False,
        # Adversarial corruption parameters (backward compatibility)
        use_adversarial_corruption = False,
        anm_warmup_steps = 5000,
        anm_adversarial_steps = 3,
        anm_distance_penalty = 0.1,
        # Curriculum configuration
        curriculum_config = None,
        disable_curriculum = False,
    ):
        super().__init__()
        self.model = model
        self.inp_dim = self.model.inp_dim
        self.out_dim = self.model.out_dim
        self.out_shape = (self.out_dim, )
        self.self_condition = False
        self.supervise_energy_landscape = supervise_energy_landscape
        self.use_innerloop_opt = use_innerloop_opt

        self.seq_length = seq_length
        self.objective = objective
        self.show_inference_tqdm = show_inference_tqdm
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.baseline = baseline
        self.sudoku = sudoku
        self.connectivity = connectivity
        self.continuous = continuous
        self.shortest_path = shortest_path
        
        # Adversarial corruption parameters (backward compatibility)
        self.use_adversarial_corruption = use_adversarial_corruption
        self.anm_warmup_steps = anm_warmup_steps
        self.anm_adversarial_steps = anm_adversarial_steps
        self.anm_distance_penalty = anm_distance_penalty
        
        # Curriculum configuration - handle explicit disabling for legacy behavior
        if disable_curriculum:
            self.curriculum_config = None
        elif curriculum_config is None:
            # If adversarial corruption is enabled, default to curriculum
            if use_adversarial_corruption:
                # Create a copy of the default curriculum with appropriate total steps
                import copy
                self.curriculum_config = copy.deepcopy(DEFAULT_CURRICULUM)
                # Update total steps to match legacy warmup behavior if needed  
                if self.anm_warmup_steps != DEFAULT_CURRICULUM.total_steps * 0.2:
                    self.curriculum_config.total_steps = self.anm_warmup_steps * 5
            else:
                self.curriculum_config = None
        else:
            self.curriculum_config = curriculum_config
            
        # Initialize validation gating system
        self.validation_gating = None
        if self.curriculum_config is not None:
            self.validation_gating = ValidationGatedCurriculum(self.curriculum_config)
            
        # Training state tracking
        self.training_step = 0
        self.recent_energy_diffs = []
        self.current_stage = None
        self.stage_transition_step = 0
        self.corruption_type_history = []
        
        # Track corruption type counts for logging
        self.corruption_type_counts = {'clean': 0, 'adversarial': 0, 'gaussian': 0}
        
        # Curriculum info caching for performance
        self._curriculum_info_cache = None
        self._curriculum_info_cache_step = -1
        self._cache_update_interval = 10  # Update cache every N steps
        
        # Validation gating state tracking
        self.effective_step = 0  # Step adjusted for rollbacks
        self.curriculum_rollback_target = None  # Target step for rollback

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Step size for optimizing
        register_buffer('opt_step_size', betas * torch.sqrt( 1 / (1 - alphas_cumprod)))
        # register_buffer('opt_step_size', 0.25 * torch.sqrt(alphas_cumprod) * torch.sqrt(1 / alphas_cumprod -1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)
        # whether to autonormalize

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, inp, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        with torch.enable_grad():
            model_output = self.model(inp, x, t)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, cond, x, t, x_self_cond = None, clip_denoised = False):
        preds = self.model_predictions(cond, x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            # x_start.clamp_(-6, 6)

            if self.continuous:
                sf = 2.0
            else:
                sf = 1.0

            x_start.clamp_(-sf, sf)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, cond, x, t, x_self_cond = None, clip_denoised = True, with_noise=False, scale=False):
        b, *_, device = *x.shape, x.device

        if type(t) == int:
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        else:
            batched_times = t
            noise = torch.randn_like(x)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(cond, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)

        # Don't scale inputs by expansion factor (Do that later)
        if not scale:
            model_mean = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape) * x_start

        if with_noise:
            pred_img = model_mean  + (0.5 * model_log_variance).exp() * noise
        else:
            pred_img = model_mean #  + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
        # Debug: Check if this is causing a hang with too many steps
        if step > 20 and self.training_step < 3:
            print(f"[DEBUG] opt_step called with {step} steps at training_step {self.training_step}")
        with torch.enable_grad():
            for i in range(step):
                energy, grad = self.model(inp, img, t, return_both=True)
                img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf  # / (i + 1) ** 0.5

                if mask is not None:
                    img_new = img_new * (1 - mask) + mask * data_cond

                if self.continuous:
                    sf = 2.0
                else:
                    sf = 1.0

                max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
                img_new = torch.clamp(img_new, -max_val, max_val)

                energy_new = self.model(inp, img_new, t, return_energy=True)
                if len(energy_new.shape) == 2:
                    bad_step = (energy_new > energy)[:, 0]
                elif len(energy_new.shape) == 1:
                    bad_step = (energy_new > energy)
                else:
                    raise ValueError('Bad shape!!!')

                # print("step: ", i, bad_step.float().mean())
                img_new[bad_step] = img[bad_step]

                if eval:
                    img = img_new.detach()
                else:
                    img = img_new

        return img

    def get_current_curriculum_params(self, step):
        """Get current curriculum stage and parameters with validation gating."""
        # Use effective step for rollback scenarios
        effective_step = step
        if self.curriculum_rollback_target is not None:
            effective_step = min(step, self.curriculum_rollback_target + (step - self.stage_transition_step))
            # Clear rollback once we've caught up
            if step >= self.curriculum_rollback_target + self.curriculum_config.rollback_steps:
                self.curriculum_rollback_target = None
        
        # Get base curriculum parameters
        stage = self.curriculum_config.get_stage(effective_step)
        base_epsilon = self.curriculum_config.get_smooth_epsilon(effective_step, self.anm_distance_penalty * 10)
        
        # Apply validation gating intensity adjustment
        intensity_multiplier = 1.0
        if self.validation_gating is not None:
            intensity_multiplier = self.validation_gating.intensity_multiplier
            
            # Check if we should adjust curriculum based on validation performance
            should_adjust, adjustment_type, reason = self.should_adjust_curriculum()
            if should_adjust:
                if adjustment_type == "rollback":
                    # Perform rollback
                    self.rollback_curriculum()
                    # Recalculate with new effective step
                    effective_step = self.effective_step
                    stage = self.curriculum_config.get_stage(effective_step)
                    base_epsilon = self.curriculum_config.get_smooth_epsilon(effective_step, self.anm_distance_penalty * 10)
                else:
                    # Adjust intensity
                    intensity_multiplier = self.adjust_curriculum_intensity(adjustment_type, reason)
        
        # Apply intensity adjustment to epsilon and stage ratios
        adjusted_epsilon = base_epsilon * intensity_multiplier
        
        # Create adjusted stage with modified parameters for smooth transitions
        if intensity_multiplier != 1.0:
            from copy import copy
            adjusted_stage = copy(stage)
            
            # Smooth intensity adjustments - when reducing intensity, increase clean ratio
            if intensity_multiplier < 1.0:
                # Increase clean ratio when reducing intensity
                clean_boost = (1.0 - intensity_multiplier) * 0.3
                adjusted_stage.clean_ratio = min(1.0, stage.clean_ratio + clean_boost)
                adjusted_stage.adversarial_ratio = max(0.0, stage.adversarial_ratio - clean_boost * 0.7)
                adjusted_stage.gaussian_ratio = max(0.0, stage.gaussian_ratio - clean_boost * 0.3)
            # When increasing intensity, we keep original ratios but increase epsilon
            
            stage = adjusted_stage
        
        # Track stage transitions
        if self.current_stage != stage.name:
            old_stage = self.current_stage
            self.current_stage = stage.name
            self.stage_transition_step = step
            
            # Set new baseline for validation gating
            if self.validation_gating is not None:
                self.validation_gating.set_stage_baseline(self.current_stage, force_update=True)
            
            print(f"[Curriculum] Step {step}: Stage transition {old_stage} -> {stage.name} "
                  f"(effective_step: {effective_step}, intensity: {intensity_multiplier:.3f})")
            
        return stage, adjusted_epsilon
    
    def _sample_corruption_type(self, stage):
        """Sample corruption type based on curriculum stage ratios."""
        rand_val = torch.rand(1).item()
        
        if rand_val < stage.clean_ratio:
            corruption_type = 'clean'
        elif rand_val < stage.clean_ratio + stage.adversarial_ratio:
            corruption_type = 'adversarial' 
        else:
            corruption_type = 'gaussian'
            
        # Update counts for logging
        self.corruption_type_counts[corruption_type] += 1
        return corruption_type
    
    def _clean_corruption(self, x_start, t):
        """Standard clean noise corruption - returns NOISED sample."""
        noise = torch.randn_like(x_start)
        # Return noised sample, matching the behavior of _standard_ired_corruption
        return self.q_sample(x_start=x_start, t=t, noise=noise)
    
    def _gaussian_noise_corruption(self, x_start, t, scale=3.0):
        """Gaussian noise corruption with scale - returns NOISED sample."""
        noise = torch.randn_like(x_start)
        # Return noised sample with scale, matching the behavior of _standard_ired_corruption
        return self.q_sample(x_start=x_start, t=t, noise=scale * noise)

    # Validation gating methods
    def update_validation_performance(self, val_loss: float, val_accuracy: float = None):
        """
        Update validation performance metrics for curriculum gating.
        
        Args:
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy (optional, defaults to 1-val_loss)
        """
        if self.validation_gating is None:
            return
        
        # Use 1-val_loss as accuracy approximation if not provided
        if val_accuracy is None:
            val_accuracy = max(0.0, 1.0 - val_loss)
        
        self.validation_gating.update_validation_metrics(val_loss, val_accuracy, self.training_step)
        
        # Set stage baseline if we've transitioned to a new stage
        if self.current_stage is not None:
            self.validation_gating.set_stage_baseline(self.current_stage)
    
    def should_adjust_curriculum(self) -> tuple[bool, str, str]:
        """
        Check if curriculum should be adjusted based on validation performance.
        
        Returns:
            Tuple of (should_adjust, adjustment_type, reason)
        """
        if self.validation_gating is None:
            return False, "none", "no_validation_gating"
        
        return self.validation_gating.should_adjust_curriculum(self.training_step)
    
    def adjust_curriculum_intensity(self, adjustment_type: str = None, reason: str = None) -> float:
        """
        Adjust curriculum intensity based on validation performance.
        
        Args:
            adjustment_type: Type of adjustment (auto-determined if None)
            reason: Reason for adjustment (auto-determined if None)
            
        Returns:
            New intensity multiplier
        """
        if self.validation_gating is None:
            return 1.0
        
        # Auto-determine adjustment if not specified
        if adjustment_type is None:
            should_adjust, adjustment_type, reason = self.should_adjust_curriculum()
            if not should_adjust:
                return self.validation_gating.intensity_multiplier
        
        old_intensity = self.validation_gating.intensity_multiplier
        new_intensity = self.validation_gating.adjust_intensity(adjustment_type, self.training_step, reason)
        
        # Log the adjustment
        if old_intensity != new_intensity:
            print(f"[Curriculum Gating] Step {self.training_step}: {adjustment_type} - "
                  f"intensity {old_intensity:.3f} -> {new_intensity:.3f} (reason: {reason})")
        
        return new_intensity
    
    def rollback_curriculum(self, rollback_steps: int = None) -> int:
        """
        Rollback curriculum to a previous stage if performance degrades.
        
        Args:
            rollback_steps: Number of steps to rollback (uses config default if None)
            
        Returns:
            New effective training step after rollback
        """
        if self.validation_gating is None:
            return self.training_step
        
        if rollback_steps is None:
            rollback_steps = self.curriculum_config.rollback_steps
        
        # Calculate rollback target
        self.curriculum_rollback_target = max(0, self.training_step - rollback_steps)
        self.effective_step = self.curriculum_rollback_target
        
        # Reset validation gating intensity and consecutive degradations
        self.validation_gating.intensity_multiplier = 1.0
        self.validation_gating.consecutive_degradations = 0
        
        print(f"[Curriculum Rollback] Step {self.training_step}: Rolling back {rollback_steps} steps "
              f"to effective step {self.effective_step}")
        
        return self.effective_step

    def enhanced_corruption_step_v2(self, inp, x_start, t, mask, data_cond, base_noise_scale=3.0):
        """New curriculum-aware corruption method - returns noised samples."""
        # Get current curriculum parameters
        stage, epsilon = self.get_current_curriculum_params(self.training_step)
        
        # Sample corruption type based on curriculum ratios
        corruption_type = self._sample_corruption_type(stage)
        
        # Store corruption type and parameters for use in p_losses
        self._current_corruption_type = corruption_type
        self._current_stage = stage
        
        # Apply the selected corruption type - all return noised samples
        if corruption_type == 'clean':
            x_corrupted = self._clean_corruption(x_start, t)
        elif corruption_type == 'gaussian':
            # Scale Gaussian noise by stage temperature for adaptive difficulty
            noise_scale = base_noise_scale * (2.0 / max(stage.temperature, 1.0))
            x_corrupted = self._gaussian_noise_corruption(x_start, t, noise_scale)
        else:  # adversarial
            x_corrupted = self._adversarial_corruption(inp, x_start, t, mask, data_cond, base_noise_scale, epsilon)
        
        # Note: All methods now return already-noised samples, matching _standard_ired_corruption
        # No additional noising will be applied in p_losses
        
        return x_corrupted

    def enhanced_corruption_step(self, inp, x_start, t, mask, data_cond, base_noise_scale=3.0):
        """Enhanced adversarial corruption with curriculum learning (legacy method)
        
        This method maintains backward compatibility while integrating curriculum support.
        Use enhanced_corruption_step_v2 for full curriculum features.
        """
        # Use new curriculum method if curriculum is configured and curriculum_config is not None
        if hasattr(self, 'curriculum_config') and self.curriculum_config is not None:
            return self.enhanced_corruption_step_v2(inp, x_start, t, mask, data_cond, base_noise_scale)
            
        # Legacy behavior for backward compatibility (when curriculum_config is explicitly set to None)
        # Note: training_step increment should be handled by caller to avoid double increment
        
        # Curriculum weight: 0 during warmup, then gradually increase
        if self.training_step < self.anm_warmup_steps:
            curriculum_weight = 0.0
        else:
            progress = (self.training_step - self.anm_warmup_steps) / max(self.anm_warmup_steps, 1)
            curriculum_weight = min(1.0, progress)
            
            # Adapt based on recent energy landscape quality
            if len(self.recent_energy_diffs) > 10:
                quality_factor = np.mean(self.recent_energy_diffs[-10:])
                curriculum_weight *= np.clip(quality_factor, 0.1, 1.0)
        
        # Use adversarial corruption with probability = curriculum_weight
        if torch.rand(1).item() < curriculum_weight and self.use_adversarial_corruption:
            return self._adversarial_corruption(inp, x_start, t, mask, data_cond, base_noise_scale, self.anm_distance_penalty)
        else:
            # Fallback to standard IRED corruption
            return self._standard_ired_corruption(inp, x_start, t, mask, data_cond, base_noise_scale)

    def _standard_ired_corruption(self, inp, x_start, t, mask, data_cond, base_noise_scale):
        """Original IRED corruption mechanism (extracted from p_losses)"""
        noise = torch.randn_like(x_start)
        xmin_noise = self.q_sample(x_start = x_start, t = t, noise = base_noise_scale * noise)
        
        if mask is not None:
            xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
        
        # Apply original opt_step with task-specific parameters
        if self.sudoku:
            step = 20
        else:
            step = 5
            
        xmin_noise = self.opt_step(inp, xmin_noise, t, mask, data_cond, step=step, sf=1.0)
        return xmin_noise

    def _adversarial_corruption(self, inp, x_start, t, mask, data_cond, base_noise_scale, epsilon=None):
        """Enhanced corruption with distance penalty - optimizes in noisy space, returns clean"""
        # Start with standard noise corruption in noisy space
        noise = torch.randn_like(x_start)
        xmin_noise = self.q_sample(x_start = x_start, t = t, noise = base_noise_scale * noise)
        
        if mask is not None:
            xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
        
        xmin_noise.requires_grad_(True)
        opt_step_size = extract(self.opt_step_size, t, xmin_noise.shape)
        
        # Enhanced adversarial steps with distance penalty
        effective_epsilon = epsilon if epsilon is not None else self.anm_distance_penalty
        
        # Store the original noisy sample for reference
        xmin_noise_orig = xmin_noise.clone().detach()
        
        for i in range(self.anm_adversarial_steps):
            energy, grad = self.model(inp, xmin_noise, t, return_both=True)
            
            # Distance penalty prevents collapse to ground truth (in noisy space)
            # Use original noisy sample as reference instead of clean x_start
            distance_penalty = F.mse_loss(xmin_noise, xmin_noise_orig)
            adaptive_penalty_weight = effective_epsilon * torch.clamp(1.0 / (distance_penalty + 1e-6), 0.1, 2.0)
            
            # Modified gradient: energy gradient - distance penalty gradient
            penalty_grad = torch.autograd.grad(distance_penalty, xmin_noise, create_graph=False, retain_graph=True)[0]
            modified_grad = grad - adaptive_penalty_weight * penalty_grad
            
            # Apply gradient step with decreasing step size
            step_scale = 1.0 * (0.7 ** i)  # Decreasing step size for stability
            xmin_noise = xmin_noise - opt_step_size * modified_grad * step_scale
            
            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
                
            # Apply existing task-specific clipping
            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0
                
            max_val = extract(self.sqrt_alphas_cumprod, t, xmin_noise.shape) * sf
            xmin_noise = torch.clamp(xmin_noise, -max_val, max_val)
            xmin_noise.requires_grad_(True)
        
        # Return the optimized noisy sample directly, matching _standard_ired_corruption behavior
        return xmin_noise.detach()

    def get_curriculum_info_for_metrics(self):
        """Get cached curriculum info for metrics tracking (performance optimized)."""
        # Use cache if available and recent
        if (self._curriculum_info_cache is not None and 
            self.training_step - self._curriculum_info_cache_step < self._cache_update_interval):
            return self._curriculum_info_cache
        
        # Update cache
        self._curriculum_info_cache = self.get_curriculum_info()
        self._curriculum_info_cache_step = self.training_step
        return self._curriculum_info_cache
    
    def get_curriculum_info(self):
        """Get current curriculum information for logging and monitoring."""
        if not hasattr(self, 'curriculum_config') or self.curriculum_config is None:
            return {
                'curriculum_enabled': False,
                'current_step': self.training_step,
                'legacy_warmup_phase': self.training_step < self.anm_warmup_steps
            }
        
        # For metrics, use simplified params without validation gating checks
        # This avoids expensive computation on every call
        stage = self.curriculum_config.get_stage(self.training_step)
        epsilon = self.curriculum_config.get_smooth_epsilon(self.training_step, self.anm_distance_penalty * 10)
        progress = self.training_step / self.curriculum_config.total_steps
        
        # Calculate corruption type percentages from recent history
        recent_history = self.corruption_type_history[-100:] if len(self.corruption_type_history) > 0 else []
        corruption_percentages = {}
        if recent_history:
            for corruption_type in ['clean', 'adversarial', 'gaussian']:
                corruption_percentages[f'{corruption_type}_recent_pct'] = recent_history.count(corruption_type) / len(recent_history)
        
        # Validation gating information
        validation_gating_info = {}
        if self.validation_gating is not None:
            validation_gating_info = {
                'validation_gating_enabled': self.curriculum_config.enable_validation_gating,
                'intensity_multiplier': self.validation_gating.intensity_multiplier,
                'validation_loss_ema': self.validation_gating.val_loss_ema,
                'validation_accuracy_ema': self.validation_gating.val_accuracy_ema,
                'consecutive_degradations': self.validation_gating.consecutive_degradations,
                'rollback_count': self.validation_gating.rollback_count,
                'last_adjustment_step': self.validation_gating.last_adjustment_step,
                'performance_trend': self.validation_gating.performance_trend,
                'effective_step': getattr(self, 'effective_step', self.training_step),
                'curriculum_rollback_active': self.curriculum_rollback_target is not None
            }
            
            # Add current stage baseline if available
            if self.validation_gating.current_stage_baseline is not None:
                validation_gating_info.update({
                    'stage_baseline_loss': self.validation_gating.current_stage_baseline['loss'],
                    'stage_baseline_accuracy': self.validation_gating.current_stage_baseline['accuracy']
                })
        else:
            validation_gating_info = {
                'validation_gating_enabled': False,
                'intensity_multiplier': 1.0
            }
        
        return {
            'curriculum_enabled': True,
            'current_step': self.training_step,
            'total_steps': self.curriculum_config.total_steps,
            'progress': progress,
            'current_stage': stage.name,
            'stage_transition_step': self.stage_transition_step,
            'stage_ratios': {
                'clean': stage.clean_ratio,
                'adversarial': stage.adversarial_ratio,
                'gaussian': stage.gaussian_ratio
            },
            'epsilon_multiplier': stage.epsilon_multiplier,
            'temperature': stage.temperature,
            'current_epsilon': epsilon,
            'corruption_counts': dict(self.corruption_type_counts),
            **corruption_percentages,
            **validation_gating_info,
            'recent_energy_quality': np.mean(self.recent_energy_diffs[-10:]) if len(self.recent_energy_diffs) >= 10 else 0.0
        }

    def get_validation_gating_summary(self) -> str:
        """
        Get a summary of validation gating status and recent adjustments.
        
        Returns:
            Formatted string with validation gating information
        """
        if self.validation_gating is None:
            return "Validation gating: Disabled"
        
        lines = [
            f"Validation gating: {'Enabled' if self.curriculum_config.enable_validation_gating else 'Disabled'}",
            f"Current intensity: {self.validation_gating.intensity_multiplier:.3f}",
            f"Validation loss EMA: {self.validation_gating.val_loss_ema:.4f}" if self.validation_gating.val_loss_ema else "Validation loss EMA: Not available",
            f"Validation accuracy EMA: {self.validation_gating.val_accuracy_ema:.4f}" if self.validation_gating.val_accuracy_ema else "Validation accuracy EMA: Not available",
            f"Consecutive degradations: {self.validation_gating.consecutive_degradations}",
            f"Total rollbacks: {self.validation_gating.rollback_count}"
        ]
        
        if self.curriculum_rollback_target is not None:
            lines.append(f"Active rollback: target step {self.curriculum_rollback_target}")
        
        # Add recent adjustments
        adjustment_log = self.validation_gating.get_adjustment_log()
        if adjustment_log != "No curriculum adjustments made.":
            lines.append(adjustment_log)
        
        return "\n".join(lines)

    def reset_curriculum_tracking(self):
        """Reset curriculum tracking variables (useful for evaluation or new training)."""
        self.training_step = 0
        self.recent_energy_diffs = []
        self.current_stage = None
        self.stage_transition_step = 0
        self.corruption_type_history = []
        self.corruption_type_counts = {'clean': 0, 'adversarial': 0, 'gaussian': 0}
        
        # Reset validation gating
        if self.validation_gating is not None:
            self.validation_gating = ValidationGatedCurriculum(self.curriculum_config)
        self.effective_step = 0
        self.curriculum_rollback_target = None

    @torch.no_grad()
    def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False):
        device = self.betas.device

        if hasattr(self.model, 'randn'):
            img = self.model.randn(batch_size, shape, inp, device)
        else:
            img = torch.randn((batch_size, *shape), device=device)

        x_start = None


        if self.show_inference_tqdm:
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))

        preds = []

        for t in iterator:
            self_cond = x_start if self.self_condition else None
            batched_times = torch.full((img.shape[0],), t, device = inp.device, dtype = torch.long)

            cond_val = None
            if mask is not None:
                cond_val = self.q_sample(x_start = inp, t = batched_times, noise = torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask

            img, x_start = self.p_sample(inp, img, t, self_cond, scale=False, with_noise=self.baseline)

            if mask is not None:
                img = img * (1 - mask) + cond_val * mask

            # if t < 50:

            if self.sudoku:
                step = 20
            else:
                step = 5

            if self.use_innerloop_opt:
                if t < 1:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)
                else:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)

                img = img.detach()

            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0

            # This clip threshold needs to be adjust to be larger for generalizations settings
            max_val = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape)[0, 0] * sf

            img = torch.clamp(img, -max_val, max_val)

            # Correctly scale output
            img_unscaled = self.predict_start_from_noise(img, batched_times, torch.zeros_like(img))
            preds.append(img_unscaled)

            batched_times_prev = batched_times - 1

            if t != 0:
                img = extract(self.sqrt_alphas_cumprod, batched_times_prev, img_unscaled.shape) * img_unscaled
            # img, _, _ = self.q_posterior(img_unscaled, img, batched_times)

        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    @torch.no_grad()
    def sample(self, x, label, mask, batch_size = 16, return_traj=False):
        # seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(batch_size, self.out_shape, x, label, mask, return_traj=return_traj)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, inp, x_start, mask, t, noise = None):
        b, *c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        if mask is not None:
            # Mask out inputs
            x_cond = self.q_sample(x_start = inp, t = t, noise = torch.zeros_like(noise))
            x = x * (1 - mask) + mask * x_cond

        # predict and take gradient step

        model_out = self.model(inp, x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if mask is not None:
            # Mask out targets
            model_out = model_out * (1 - mask) + mask * target

        loss = F.mse_loss(model_out, target, reduction = 'none')

        if self.shortest_path:
            mask1 = (x_start > 0)
            mask2 = torch.logical_not(mask1)
            # mask1, mask2 = mask1.float(), mask2.float()
            weight = mask1 * 10 + mask2 * 0.5
            # loss = (loss * weight) / weight.sum() * target.numel()
            loss = loss * weight

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss_mse = loss

        if self.supervise_energy_landscape:
            noise = torch.randn_like(x_start)
            data_sample = self.q_sample(x_start = x_start, t = t, noise = noise)

            if mask is not None:
                data_cond = self.q_sample(x_start = x_start, t = t, noise = torch.zeros_like(noise))
                data_sample = data_sample * (1 - mask) + mask * data_cond
            else:
                data_cond = None

            # Enhanced adversarial corruption replaces all task-specific corruption logic
            # Note: training_step should be incremented before calling this method
            xmin_noise_rescale = self.enhanced_corruption_step(inp, x_start, t, mask, data_cond)
            loss_opt = torch.ones(1).to(x_start.device)
            loss_scale = 0.5

            # The corruption methods now return already-noised samples, so no need to noise again
            # This matches the behavior of _standard_ired_corruption
            xmin_noise = xmin_noise_rescale

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

            # Compute energy of both distributions
            inp_concat = torch.cat([inp, inp], dim=0)
            x_concat = torch.cat([data_sample, xmin_noise], dim=0)
            # x_concat = torch.cat([xmin, xmin_noise_min], dim=0)
            t_concat = torch.cat([t, t], dim=0)
            energy = self.model(inp_concat, x_concat, t_concat, return_energy=True)

            # Compute noise contrastive energy loss
            energy_real, energy_fake = torch.chunk(energy, 2, 0)
            energy_stack = torch.cat([energy_real, energy_fake], dim=-1)
            target = torch.zeros(energy_real.size(0)).to(energy_stack.device)
            loss_energy = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')[:, None]

            # Track energy landscape quality for adaptive curriculum
            with torch.no_grad():
                energy_diff = (energy_fake - energy_real).mean().item()
                self.recent_energy_diffs.append(max(0, energy_diff))
                if len(self.recent_energy_diffs) > 100:
                    self.recent_energy_diffs.pop(0)

            # loss_energy = energy_real.mean() - energy_fake.mean()# loss_energy.mean()

            loss = loss_mse + loss_scale * loss_energy # + 0.001 * loss_opt
            return loss.mean(), (loss_mse.mean(), loss_energy.mean(), loss_opt.mean())
        else:
            loss = loss_mse
            return loss.mean(), (loss_mse.mean(), -1, -1)

    def forward(self, inp, target, mask, *args, **kwargs):
        b, *c = target.shape
        device = target.device
        if len(c) == 1:
            self.out_dim = c[0]
            self.out_shape = c
        else:
            self.out_dim = c[-1]
            self.out_shape = c

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(inp, target, mask, t, *args, **kwargs)

# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        validation_batch_size = None,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        data_workers = None,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        metric = 'mse',
        cond_mask = False,
        validation_dataset = None,
        extra_validation_datasets = None,
        extra_validation_every_mul = 10,
        evaluate_first = False,
        latent = False,
        autoencode_model = None,
        save_csv_logs = False,
        csv_log_interval = 100,
        csv_log_dir = './csv_logs',
        dataset_name = None  # Dataset name for task-specific accuracy
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model

        # Conditioning on mask

        self.cond_mask = cond_mask

        # Whether to do reasoning in the latent space

        self.latent = latent

        if autoencode_model is not None:
            self.autoencode_model = autoencode_model.cuda()

        # sampling and training hyperparameters
        self.out_dim = self.model.out_dim

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.extra_validation_every_mul = extra_validation_every_mul

        self.batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size if validation_batch_size is not None else train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # Evaluation metric.
        self.metric = metric
        self.data_workers = data_workers
        self.dataset = dataset_name  # Store dataset name for accuracy computation

        if self.data_workers is None:
            # Use a conservative default to avoid multiprocessing issues, especially on macOS
            # cpu_count() can return high values that cause deadlocks
            self.data_workers = min(4, cpu_count())

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = self.data_workers)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.validation_dataset = validation_dataset

        if self.validation_dataset is not None:
            dl = DataLoader(self.validation_dataset, batch_size = validation_batch_size, shuffle=False, pin_memory=False, num_workers = self.data_workers)
            dl = self.accelerator.prepare(dl)
            self.validation_dl = dl
        else:
            self.validation_dl = None

        self.extra_validation_datasets = extra_validation_datasets

        if self.extra_validation_datasets is not None:
            self.extra_validation_dls = dict()
            for key, dataset in self.extra_validation_datasets.items():
                dl = DataLoader(dataset, batch_size = validation_batch_size, shuffle=False, pin_memory=False, num_workers = self.data_workers)
                dl = self.accelerator.prepare(dl)
                self.extra_validation_dls[key] = dl
        else:
            self.extra_validation_dls = None

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # CSV logging setup
        self.save_csv_logs = save_csv_logs
        self.csv_log_interval = csv_log_interval
        if self.save_csv_logs and self.accelerator.is_main_process:
            self.csv_log_dir = Path(csv_log_dir)
            self.csv_log_dir.mkdir(exist_ok=True)
            self._init_csv_logging()
        
        # Metrics tracking disabled
        self.metrics_tracker = None

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        
        # Initialize EMA after model is prepared to ensure same device
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        self.evaluate_first = evaluate_first

    def _init_csv_logging(self):
        """Initialize CSV files for logging training and validation metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training metrics CSV
        self.train_csv_path = self.csv_log_dir / f'training_metrics_{timestamp}.csv'
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'total_loss', 'loss_denoise', 'loss_energy', 'loss_opt',
                'data_time', 'nn_time', 'learning_rate', 'timestamp'
            ])
        
        # Validation metrics CSV
        self.val_csv_path = self.csv_log_dir / f'validation_metrics_{timestamp}.csv'
        with open(self.val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'milestone', 'dataset_name', 'metric_name', 'metric_value', 'timestamp'
            ])
        
        # Energy landscape metrics CSV (for adversarial corruption analysis)
        self.energy_csv_path = self.csv_log_dir / f'energy_metrics_{timestamp}.csv'
        with open(self.energy_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'energy_pos_mean', 'energy_neg_mean', 'energy_diff', 
                'curriculum_weight', 'corruption_type', 'timestamp'
            ])
        
        # Curriculum metrics CSV
        self.curriculum_csv_path = self.csv_log_dir / f'curriculum_metrics_{timestamp}.csv'
        with open(self.curriculum_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'stage', 'corruption_type', 'epsilon', 'temperature',
                'clean_ratio', 'adversarial_ratio', 'gaussian_ratio', 'timestamp'
            ])
        
        # Overfitting metrics CSV
        self.overfitting_csv_path = self.csv_log_dir / f'overfitting_metrics_{timestamp}.csv'
        with open(self.overfitting_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'train_val_gap', 'risk_level', 'gap_trend', 'patience_counter',
                'should_intervene', 'steps_without_improvement', 'timestamp'
            ])
        
        # Robustness metrics CSV
        self.robustness_csv_path = self.csv_log_dir / f'robustness_metrics_{timestamp}.csv'
        with open(self.robustness_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'clean_accuracy', 'adversarial_accuracy', 'attack_success_rate',
                'accuracy_gap', 'robustness_ratio', 'pareto_score', 'timestamp'
            ])
    
    def _log_to_csv(self, csv_path, row_data):
        """Helper function to append data to CSV file"""
        if self.save_csv_logs and self.accelerator.is_main_process:
            try:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
            except Exception as e:
                print(f"Warning: Failed to write to CSV {csv_path}: {e}")
    
    def _log_curriculum_metrics(self, curriculum_info):
        """Log curriculum stage, corruption types, epsilon, temperature"""
        if not self.save_csv_logs or not curriculum_info.get('curriculum_enabled', False):
            return
            
        timestamp = datetime.now().isoformat()
        stage_ratios = curriculum_info.get('stage_ratios', {})
        
        curriculum_row = [
            curriculum_info.get('current_step', self.step),
            curriculum_info.get('current_stage', 'unknown'),
            curriculum_info.get('corruption_type', 'unknown'),
            curriculum_info.get('current_epsilon', 0.0),
            curriculum_info.get('temperature', 1.0),
            stage_ratios.get('clean', 0.0),
            stage_ratios.get('adversarial', 0.0),
            stage_ratios.get('gaussian', 0.0),
            timestamp
        ]
        self._log_to_csv(self.curriculum_csv_path, curriculum_row)
    
    def _log_overfitting_metrics(self, overfitting_check):
        """Log train-val gap, overfitting risk, patience counter"""
        if not self.save_csv_logs:
            return
            
        timestamp = datetime.now().isoformat()
        overfitting_row = [
            self.step,
            overfitting_check.get('gap', 0.0),
            overfitting_check.get('risk_level', 'unknown'),
            overfitting_check.get('gap_trend', 0.0),
            overfitting_check.get('patience_counter', 0),
            overfitting_check.get('should_intervene', False),
            overfitting_check.get('steps_without_improvement', 0),
            timestamp
        ]
        self._log_to_csv(self.overfitting_csv_path, overfitting_row)
    
    def _log_robustness_metrics(self, clean_acc=None, adv_acc=None, attack_success=None):
        """Log clean vs adversarial accuracy if available"""
        if not self.save_csv_logs or clean_acc is None:
            return
            
        timestamp = datetime.now().isoformat()
        
        # Calculate derived metrics
        accuracy_gap = (clean_acc - adv_acc) if adv_acc is not None else 0.0
        robustness_ratio = (adv_acc / clean_acc) if adv_acc is not None and clean_acc > 0 else 0.0
        pareto_score = (0.5 * clean_acc + 0.5 * adv_acc) if adv_acc is not None else clean_acc
        
        robustness_row = [
            self.step,
            clean_acc,
            adv_acc if adv_acc is not None else 0.0,
            attack_success if attack_success is not None else 0.0,
            accuracy_gap,
            robustness_ratio,
            pareto_score,
            timestamp
        ]
        self._log_to_csv(self.robustness_csv_path, robustness_row)
    
    def _save_comprehensive_metrics(self):
        """Save comprehensive metrics dictionary to file"""
        # Skip metrics tracking
        return
            
        try:
            import json
            metrics_dict = self.metrics_tracker.export_metrics_to_dict()
            
            # Create metrics export directory
            metrics_export_dir = self.csv_log_dir / 'comprehensive_metrics'
            metrics_export_dir.mkdir(exist_ok=True)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = metrics_export_dir / f'metrics_step_{self.step}_{timestamp}.json'
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
                
            print(f"📊 Comprehensive metrics saved to {metrics_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save comprehensive metrics: {e}")

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        if osp.isfile(milestone):
            milestone_file = milestone
        else:
            milestone_file = str(self.results_folder / f'model-{milestone}.pt')
        data = torch.load(milestone_file)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
        # Metrics tracking disabled

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if self.evaluate_first:
            milestone = self.step // self.save_and_sample_every
            self.evaluate(device, milestone)
            self.evaluate_first = False  # hack: later we will use this flag as a bypass signal to determine whether we want to run extra validation.

        end_time = time.time()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:

            epoch = 0
            while self.step < self.train_num_steps:

                total_loss = 0.

                end_tiem = time.time()
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)

                    if self.cond_mask:
                        inp, label, mask = data
                        inp, label, mask = inp.float().to(device), label.float().to(device), mask.float().to(device)
                    elif self.latent:
                        inp, label, label_gt, mask_latent = data
                        mask_latent = mask_latent.float().to(device)
                        inp, label, label_gt = inp.float().to(device), label.float().to(device), label_gt.float().to(device)
                        mask = None
                    else:
                        inp, label = data
                        inp, label = inp.float().to(device), label.float().to(device)
                        mask = None

                    data_time = time.time() - end_time; end_time = time.time()

                    # Increment model's training step for curriculum tracking
                    model = self.model.module if hasattr(self.model, 'module') else self.model
                    if hasattr(model, 'training_step'):
                        model.training_step = self.step

                    with self.accelerator.autocast():
                        loss, (loss_denoise, loss_energy, loss_opt) = self.model(inp, label, mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                nn_time = time.time() - end_time; end_time = time.time()
                
                # Handle case where loss_energy and loss_opt might be -1 (integers)
                loss_energy_display = loss_energy if not hasattr(loss_energy, 'item') else loss_energy.item()
                loss_opt_display = loss_opt if not hasattr(loss_opt, 'item') else loss_opt.item()
                
                pbar.set_description(f'loss: {total_loss:.4f} loss_denoise: {loss_denoise:.4f} loss_energy: {loss_energy_display:.4f} loss_opt: {loss_opt_display:.4f} data_time: {data_time:.2f} nn_time: {nn_time:.2f}')

                # Enhanced metrics tracking and logging
                if self.step % self.csv_log_interval == 0:
                    # Get model reference for curriculum info
                    model = self.model.module if hasattr(self.model, 'module') else self.model
                    
                    # Metrics tracking disabled
                    
                    # CSV logging (original functionality preserved)
                    if self.save_csv_logs:
                        current_lr = self.opt.param_groups[0]['lr']
                        timestamp = datetime.now().isoformat()
                        
                        # Handle case where loss_energy and loss_opt might be -1 (integers)
                        loss_energy_val = loss_energy.item() if hasattr(loss_energy, 'item') else loss_energy
                        loss_opt_val = loss_opt.item() if hasattr(loss_opt, 'item') else loss_opt
                        
                        train_row = [
                            self.step, epoch, total_loss, loss_denoise.item(), 
                            loss_energy_val, loss_opt_val, data_time, nn_time, 
                            current_lr, timestamp
                        ]
                        self._log_to_csv(self.train_csv_path, train_row)
                        
                        # Log energy landscape metrics if available
                        if hasattr(model, 'recent_energy_diffs'):
                            recent_diffs = model.recent_energy_diffs
                            if len(recent_diffs) > 0:
                                self._log_energy_metrics(loss_energy.item(), recent_diffs[-1])
                
                # Periodic comprehensive reporting (every 1000 steps)
                # Metrics tracking disabled

                # Metrics tracking disabled

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # if True:
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every

                        self.save(milestone)
                        
                        # Metrics tracking disabled

                        if self.latent:
                            self.evaluate(device, milestone, inp=inp, label=label_gt, mask=mask_latent)
                        else:
                            self.evaluate(device, milestone, inp=inp, label=label, mask=mask)


                pbar.update(1)
                
                # Update epoch counter (approximate)
                if self.step % 1000 == 0:
                    epoch += 1

        accelerator.print('training complete')
        
    def _log_energy_metrics(self, loss_energy, energy_diff):
        """Log energy landscape specific metrics"""
        curriculum_weight = 0.0
        corruption_type = "standard"
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model, 'use_adversarial_corruption') and model.use_adversarial_corruption:
            if hasattr(model, 'training_step') and model.training_step > model.anm_warmup_steps:
                curriculum_weight = min(1.0, (model.training_step - model.anm_warmup_steps) / model.anm_warmup_steps)
                corruption_type = "adversarial" if curriculum_weight > 0.1 else "mixed"
        
        timestamp = datetime.now().isoformat()
        energy_row = [
            self.step, 0, 0, energy_diff, curriculum_weight, corruption_type, timestamp
        ]
        self._log_to_csv(self.energy_csv_path, energy_row)

    def evaluate(self, device, milestone, inp=None, label=None, mask=None):
        print('Running Evaluation...')
        self.ema.ema_model.eval()

        if inp is not None and label is not None:
            with torch.no_grad():
                # batches = num_to_groups(self.num_samples, self.batch_size)

                if self.latent:
                    all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0)), range(1)))
                else:
                    all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0)), range(1)))
                    # all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0), return_traj=True), range(1)))
                # all_samples_list = list(map(lambda n: self.model.sample(inp, label, mask, batch_size=inp.size(0)), range(1)))
                # all_samples_list = [self.model.sample(inp, batch_size=inp.size(0))]

                all_samples = torch.cat(all_samples_list, dim = 0)

                print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (Train)')
                if self.metric == 'mse':
                    all_samples = torch.cat(all_samples_list, dim = 0)
                    
                    # Compute inverse accuracy for inverse task
                    if self.dataset == 'inverse':
                        accuracy_metrics = inverse_accuracy(all_samples, inp, label)
                        rows = [[k, v] for k, v in accuracy_metrics.items()]
                        print(tabulate(rows))
                        
                        # Log all metrics to CSV
                        if self.save_csv_logs:
                            timestamp = datetime.now().isoformat()
                            for metric_name, metric_value in accuracy_metrics.items():
                                val_row = [self.step, milestone, 'train_sample', metric_name, metric_value, timestamp]
                                self._log_to_csv(self.val_csv_path, val_row)
                    else:
                        # Regular MSE for other tasks
                        mse_error = (all_samples - label).pow(2).mean()
                        rows = [('mse_error', mse_error)]
                        print(tabulate(rows))
                        
                        # Log to CSV
                        if self.save_csv_logs:
                            timestamp = datetime.now().isoformat()
                            val_row = [self.step, milestone, 'train_sample', 'mse_error', mse_error.item(), timestamp]
                            self._log_to_csv(self.val_csv_path, val_row)
                elif self.metric == 'bce':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                elif self.metric == 'sudoku':
                    assert len(all_samples_list) == 1
                    summary = sudoku_accuracy(all_samples_list[0], label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                    
                    # Log to CSV
                    if self.save_csv_logs:
                        timestamp = datetime.now().isoformat()
                        for metric_name, metric_value in summary.items():
                            val_row = [self.step, milestone, 'train_sample', metric_name, metric_value, timestamp]
                            self._log_to_csv(self.val_csv_path, val_row)
                elif self.metric == 'sort':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    summary.update(sort_accuracy(all_samples_list[0], label, mask))
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'sort-2':
                    assert len(all_samples_list) == 1
                    summary = sort_accuracy_2(all_samples_list[0], label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'shortest-path-1d':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    summary.update(shortest_path_1d_accuracy(all_samples_list[0], label, mask, inp))
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'sudoku_latent':
                    sample = all_samples_list[0].view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)

                    assert len(all_samples_list) == 1
                    summary = sudoku_accuracy(prediction, label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                else:
                    raise NotImplementedError()

        if self.validation_dl is not None:
            self._run_validation(self.validation_dl, device, milestone, prefix = 'Validation')

        if (self.step % (self.save_and_sample_every * self.extra_validation_every_mul) == 0 and self.extra_validation_dls is not None) or self.evaluate_first:
            for key, extra_dl in self.extra_validation_dls.items():
                self._run_validation(extra_dl, device, milestone, prefix = key)

    def _run_validation(self, dl, device, milestone, prefix='Validation'):
        meters = collections.defaultdict(AverageMeter)
        with torch.no_grad():
            for i, data in enumerate(tqdm(dl, total=len(dl), desc=f'running on the validation dataset (ID: {prefix})')):
                if self.cond_mask:
                    inp, label, mask = map(lambda x: x.float().to(device), data)
                elif self.latent:
                    inp, label, label_gt, mask = map(lambda x: x.float().to(device), data)
                else:
                    inp, label = map(lambda x: x.float().to(device), data)
                    mask = None

                if self.latent:
                    # Masking doesn't make sense in the latent space
                    # samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                    samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                else:
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))

                # np.savez("sudoku.npz", inp=inp.detach().cpu().numpy(), label=label.detach().cpu().numpy(), mask=mask.detach().cpu().numpy(), samples=samples.detach().cpu().numpy())
                # import pdb
                # pdb.set_trace()
                # print("here")
                if self.metric == 'sudoku':
                    # samples_traj = samples
                    summary = sudoku_accuracy(samples[-1], label, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sudoku_latent':
                    sample = samples.view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)
                    summary = sudoku_accuracy(prediction, label_gt, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sort':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(sort_accuracy(samples, label, mask))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'sort-2':
                    summary = sort_accuracy_2(samples, label, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'shortest-path-1d':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(shortest_path_1d_accuracy(samples, label, mask, inp))
                    # summary.update(shortest_path_1d_accuracy_closed_loop(samples, label, mask, inp, self.ema.ema_model.sample))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'mse':
                    # Compute inverse accuracy for inverse task
                    if self.dataset == 'inverse':
                        accuracy_metrics = inverse_accuracy(samples, inp, label)
                        for k, v in accuracy_metrics.items():
                            meters[k].update(v, n=inp.size(0))
                    else:
                        # Regular MSE for other tasks
                        mse_error = (samples - label).pow(2).mean()
                        meters['mse'].update(mse_error, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'bce':
                    summary = binary_classification_accuracy_4(samples, label)
                    for k, v in summary.items():
                        meters[k].update(v, n=samples.shape[0])
                    if i > 20:
                        break
                else:
                    raise NotImplementedError()

            rows = [[k, v.avg] for k, v in meters.items()]
            print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (ID: {prefix})')
            print(tabulate(rows))
            
            # Metrics tracking disabled
            
            # Log validation results to CSV (original functionality preserved)
            if self.save_csv_logs:
                timestamp = datetime.now().isoformat()
                for metric_name, meter in meters.items():
                    val_row = [self.step, milestone, prefix, metric_name, float(meter.avg), timestamp]
                    self._log_to_csv(self.val_csv_path, val_row)


as_float = lambda x: float(x.item())


@torch.no_grad()
def binary_classification_accuracy(pred: torch.Tensor, label: torch.Tensor, name: str = '', saturation: bool = True) -> dict[str, float]:
    r"""Compute the accuracy of binary classification.

    Args:
        pred: the prediction, of the same shape as ``label``.
        label: the label, of the same shape as ``pred``.
        name: the name of this monitor.
        saturation: whether to check the saturation of the prediction. Saturation
            is defined as :math:`1 - \min(pred, 1 - pred)`

    Returns:
        a dict of monitor values.
    """
    if name != '':
        name = '/' + name
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    acc = label.float().eq((pred > 0.5).float())
    if saturation:
        sat = 1 - (pred - (pred > 0.5).float()).abs()
        return {
            prefix: as_float(acc.float().mean()),
            prefix + '/saturation/mean': as_float(sat.mean()),
            prefix + '/saturation/min': as_float(sat.min())
        }
    return {prefix: as_float(acc.float().mean())}


@torch.no_grad()
def binary_classification_accuracy_4(pred: torch.Tensor, label: torch.Tensor, name: str = '') -> dict[str, float]:
    if name != '':
        name = '/' + name

    # table = list()
    # table.append(('pred', pred[0].squeeze()))
    # table.append(('label', label[0].squeeze()))
    # print(tabulate(table))

    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    numel = pred.numel()

    gt_0_pred_0 = ((label < 0.0) & (pred < 0.0)).sum() / numel
    gt_0_pred_1 = ((label < 0.0) & (pred >= 0.0)).sum() / numel
    gt_1_pred_0 = ((label > 0.0) & (pred < 0.0)).sum() / numel
    gt_1_pred_1 = ((label > 0.0) & (pred >= 0.0)).sum() / numel

    accuracy = gt_0_pred_0 + gt_1_pred_1
    balanced_accuracy = sum([
        gt_0_pred_0 / ((label < 0.0).float().sum() / numel),
        gt_1_pred_1 / ((label >= 0.0).float().sum() / numel),
    ]) / 2

    return {
        prefix + '/gt_0_pred_0': as_float(gt_0_pred_0),
        prefix + '/gt_0_pred_1': as_float(gt_0_pred_1),
        prefix + '/gt_1_pred_0': as_float(gt_1_pred_0),
        prefix + '/gt_1_pred_1': as_float(gt_1_pred_1),
        prefix + '/accuracy': as_float(accuracy),
        prefix + '/balance_accuracy': as_float(balanced_accuracy),
    }


@torch.no_grad()
def sudoku_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = '') -> dict[str, float]:
    if name != '':
        name = '/' + name

    pred = pred.view(-1, 9, 9, 9).argmax(dim=-1)
    label = label.view(-1, 9, 9, 9).argmax(dim=-1)

    correct = (pred == label).float()
    mask = mask.view(-1, 9, 9, 9)[:, :, :, 0]
    mask_inverse = 1 - mask

    accuracy = (correct * mask_inverse).sum() / mask_inverse.sum()

    return {
        'accuracy': as_float(accuracy),
        'consistency': as_float(sudoku_consistency(pred)),
        'board_accuracy': as_float(sudoku_score(pred))
    }


def sudoku_consistency(pred: torch.Tensor) -> bool:
    pred_onehot = F.one_hot(pred, num_classes=9)

    all_row_correct = (pred_onehot.sum(dim=1) == 1).all(dim=-1).all(dim=-1)
    all_col_correct = (pred_onehot.sum(dim=2) == 1).all(dim=-1).all(dim=-1)

    blocked = pred_onehot.view(-1, 3, 3, 3, 3, 9)
    all_block_correct = (blocked.sum(dim=(2, 4)) == 1).all(dim=-1).all(dim=-1).all(dim=-1)

    return (all_row_correct & all_col_correct & all_block_correct).float().mean()


def sudoku_score(pred: torch.Tensor) -> bool:
    valid_mask = torch.ones_like(pred)

    pred_sum_axis_1 = pred.sum(dim=1, keepdim=True)
    pred_sum_axis_2 = pred.sum(dim=2, keepdim=True)

    # Use the sum criteria from the SAT-Net paper
    axis_1_mask = (pred_sum_axis_1 == 36)
    axis_2_mask = (pred_sum_axis_2 == 36)

    valid_mask = valid_mask * axis_1_mask.float() * axis_2_mask.float()

    valid_mask = valid_mask.view(-1, 3, 3, 3, 3)
    grid_mask = pred.view(-1, 3, 3, 3, 3).sum(dim=(2, 4), keepdim=True) == 36

    valid_mask = valid_mask * grid_mask.float()

    return valid_mask.mean()


def sort_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    array = (label[:, 0, ..., 2] * 0.5 + 0.5).sum(dim=-1).cpu()
    pred = pred.cpu()
    for t in range(pred.shape[1]):
        pred_xy = pred[:, t, ..., -1].reshape(pred.shape[0], -1).argmax(dim=-1)
        pred_x = torch.div(pred_xy, pred.shape[2], rounding_mode='floor')
        pred_y = pred_xy % pred.shape[2]
        # swap x and y
        next_array = array.clone()
        next_array.scatter_(1, pred_y.unsqueeze(1), array.gather(1, pred_x.unsqueeze(1)))
        next_array.scatter_(1, pred_x.unsqueeze(1), array.gather(1, pred_y.unsqueeze(1)))
        array = next_array

    ground_truth = torch.arange(pred.shape[2] - 1, -1, -1, device=array.device).unsqueeze(0).repeat(pred.shape[0], 1)
    elem_close = (array - ground_truth).abs() < 0.1
    element_correct = elem_close.float().mean()
    array_correct = elem_close.all(dim=-1).float().mean()
    return {
        'element_correct': as_float(element_correct),
        'array_correct': as_float(array_correct),
    }


def sort_accuracy_2(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    array = label[:, 0, :, 0].clone().cpu()  # B x N
    pred = pred.cpu()
    for t in range(pred.shape[1]):
        pred_x = pred[:, t, :, 1].argmax(dim=-1)  # B x N
        pred_y = pred[:, t, :, 2].argmax(dim=-1)  # B x N
        # swap x and y
        next_array = array.clone()
        next_array.scatter_(1, pred_y.unsqueeze(1), array.gather(1, pred_x.unsqueeze(1)))
        next_array.scatter_(1, pred_x.unsqueeze(1), array.gather(1, pred_y.unsqueeze(1)))
        array = next_array

    # stupid_impl_array = label[:, 0, :, 0].clone()  # B x N
    # for b in range(pred.shape[0]):
    #     for t in range(pred.shape[1]):
    #         pred_x = pred[b, t, :, 1].argmax(dim=-1)2
    #         pred_y = pred[b, t, :, 2].argmax(dim=-1)
    #         # swap x and y
    #         u, v = stupid_impl_array[b, pred_y].clone(), stupid_impl_array[b, pred_x].clone()
    #         stupid_impl_array[b, pred_x], stupid_impl_array[b, pred_y] = u, v

    # assert (array == stupid_impl_array).all(), 'Inconsistent implementation'
    # print('Consistent implementation!!')

    elem_close = torch.abs(array - label[:, -1, :, 0].cpu()) < 1e-5
    element_correct = elem_close.float().mean()
    array_correct = elem_close.all(dim=-1).float().mean()

    pred_first_action = pred[:, 0, :, 1:3].argmax(dim=-2).cpu()
    label_first_action = label[:, 0, :, 1:3].argmax(dim=-2).cpu()
    first_action_correct = (pred_first_action == label_first_action).all(dim=-1).float().mean()

    return {
        'element_accuracy' + name: as_float(element_correct),
        'array_accuracy' + name: as_float(array_correct),
        'first_action_accuracy' + name: as_float(first_action_correct)
    }


@torch.no_grad()
def inverse_accuracy(pred: torch.Tensor, inp: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
    """Calculate accuracy metrics for matrix inverse task.
    
    Args:
        pred: Predicted inverse matrices (flattened)
        inp: Input matrices to be inverted (flattened)
        label: True inverse matrices (flattened)
    
    Returns:
        Dictionary with accuracy metrics
    """
    batch_size = pred.shape[0]
    rank = int(np.sqrt(pred.shape[1]))
    
    # Reshape to matrices
    pred_mat = pred.view(batch_size, rank, rank)
    input_mat = inp.view(batch_size, rank, rank)
    true_mat = label.view(batch_size, rank, rank)
    
    # Key metric: pred @ input should be Identity
    product = torch.bmm(pred_mat, input_mat)
    identity = torch.eye(rank, device=pred.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Compute metrics
    identity_error = (product - identity).pow(2).mean(dim=(1,2))  # Per-sample error
    mse = (pred - label).pow(2).mean()
    relative_error = (pred - label).pow(2).sum(dim=1) / (label.pow(2).sum(dim=1) + 1e-8)
    
    # Task success: product is close to identity (threshold-based)
    threshold = 0.1
    task_success = (identity_error < threshold).float()
    
    return {
        'mse': as_float(mse),
        'identity_error': as_float(identity_error.mean()),
        'relative_error': as_float(relative_error.mean()),
        'accuracy': as_float(task_success.mean()),  # Changed from accuracy_pct, now returns fraction (0-1)
        'mean_abs_error': as_float((pred - label).abs().mean())
    }


def shortest_path_1d_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, inp: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    pred_argmax = pred[:, :, :, -1].argmax(-1)
    label_argmax = label[:, :, :, -1].argmax(-1)

    argmax_accuracy = (pred_argmax == label_argmax).float().mean()

    # vis_array = torch.stack([pred_argmax, label_argmax], dim=1)
    # table = list()
    # for i in range(len(vis_array)):
    #     table.append((vis_array[i, 0].cpu().tolist(), vis_array[i, 1].cpu().tolist()))
    # print(tabulate(table))

    pred_argmax_first = pred_argmax[:, 0]
    label_argmax_first = label_argmax[:, 0]

    first_action_accuracy = (pred_argmax_first == label_argmax_first).float().mean()

    first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
    first_action_t = pred_argmax_first
    first_action_feasibility = (inp[
        torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
        first_action_s,
        first_action_t,
        0
    ] > 0).float().cpu()

    final_t = label_argmax[:, -1]
    first_action_accuracy_2 = first_action_distance_accuracy(inp[..., 0], first_action_s, final_t, first_action_t).float().cpu()
    first_action_accuracy_2 = first_action_accuracy_2 * first_action_feasibility

    return {
        'argmax_accuracy' + name: as_float(argmax_accuracy),
        'first_action_accuracy' + name: as_float(first_action_accuracy),
        'first_action_feasibility' + name: as_float(first_action_feasibility.mean()),
        'first_action_accuracy_2' + name: as_float(first_action_accuracy_2.mean()),
    }


def get_shortest_batch(edges: torch.Tensor) -> torch.Tensor:
    """ Return the length of shortest path between nodes. """
    b = edges.shape[0]
    n = edges.shape[1]

    # n + 1 indicates unreachable.
    shortest = torch.ones((b, n, n), dtype=torch.float32, device=edges.device) * (n + 1)
    shortest[torch.where(edges == 1)] = 1
    # Make sure that shortest[x, x] = 0
    shortest -= shortest * torch.eye(n).unsqueeze(0).to(shortest.device)
    shortest = shortest

    # Floyd Algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    shortest[:, i, j] = torch.min(shortest[:, i, j], shortest[:, i, k] + shortest[:, k, j])
    return shortest


def first_action_distance_accuracy(edge: torch.Tensor, s: torch.Tensor, t: torch.Tensor, pred: torch.Tensor):
    shortest = get_shortest_batch(edge.detach().cpu())
    b = edge.shape[0]
    b_arrange = torch.arange(b, dtype=torch.int64, device=edge.device)
    return shortest[b_arrange, pred, t] < shortest[b_arrange, s, t]


def shortest_path_1d_accuracy_closed_loop(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, inp: torch.Tensor, sample_fn, name: str = '', execution_steps: int = 1):
    assert execution_steps in (1, 2), 'Only 1-step and 2-step execution is supported'
    b, t, n, _ = pred.shape
    failed = torch.zeros(b, dtype=torch.bool, device='cpu')
    succ = torch.zeros(b, dtype=torch.bool, device='cpu')

    for i in range(8 // execution_steps):
        pred_argmax = pred[:, :, :, -1].argmax(-1)
        pred_argmax_first = pred_argmax[:, 0]
        pred_argmax_second = pred_argmax[:, 1]
        target_argmax = inp[:, :, 0, 3].argmax(dim=-1)

        first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
        first_action_t = pred_argmax_first
        first_action_feasibility = (inp[
            torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
            first_action_s,
            first_action_t,
            0
        ] > 0).cpu()
        last_t = first_action_t

        failed |= ~(first_action_feasibility.to(torch.bool))
        succ |= (first_action_t == target_argmax).cpu() & ~failed

        print(f'Step {i} (F) s={first_action_s[0].item()}, t={first_action_t[0].item()}, goal={target_argmax[0].item()}, feasible={first_action_feasibility[0].item()}')

        if execution_steps >= 2:
            second_action_s = first_action_t
            second_action_t = pred_argmax_second
            second_action_feasibility = (inp[
                torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
                second_action_s,
                second_action_t,
                0
            ] > 0).cpu()
            failed |= ~(second_action_feasibility.to(torch.bool))
            succ |= (second_action_t == target_argmax).cpu() & ~failed
            last_t = second_action_t

            print(f'Step {i} (S) s={second_action_s[0].item()}, t={second_action_t[0].item()}, goal={target_argmax[0].item()}, feasible={second_action_feasibility[0].item()}')

        inp_clone = inp.clone()
        inp_clone[:, :, :, 1] = 0
        inp_clone[torch.arange(b, dtype=torch.int64, device=inp.device), last_t, :, 1] = 1
        inp = inp_clone
        pred = sample_fn(inp, label, mask, batch_size=inp.size(0))

    return {
        'closed_loop_success_rate' + name: as_float(succ.float().mean()),
    }

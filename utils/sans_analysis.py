"""
SANS (Self-Adversarial Negative Sampling) Analysis Utilities

This module provides functions for analyzing and visualizing SANS training metrics,
including correlation analysis, energy distributions, and hyperparameter optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import torch
from scipy import stats


def load_sans_metrics(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load standard metrics and SANS debugging metrics from a results directory.
    
    Args:
        results_dir: Path to the results directory
    
    Returns:
        Tuple of (metrics_df, sans_debug_df)
    """
    results_path = Path(results_dir)
    
    # Load standard metrics
    metrics_path = results_path / 'metrics.csv'
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    
    # Load SANS debugging metrics
    sans_path = results_path / 'sans_debug.csv'
    sans_df = pd.read_csv(sans_path) if sans_path.exists() else pd.DataFrame()
    
    return metrics_df, sans_df


def analyze_correlation_quality(sans_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze the quality of SANS correlation between weights and energy.
    
    Good SANS behavior should show:
    - Negative correlation (weights focus on harder negatives with lower energy)
    - Correlation strength increasing over training
    - Stable correlation values (not oscillating wildly)
    
    Args:
        sans_df: DataFrame with SANS debugging metrics
    
    Returns:
        Dictionary with correlation statistics
    """
    if 'weight_energy_corr' not in sans_df.columns:
        return {}
    
    corr_values = sans_df['weight_energy_corr'].dropna()
    
    return {
        'mean_correlation': corr_values.mean(),
        'std_correlation': corr_values.std(),
        'min_correlation': corr_values.min(),
        'max_correlation': corr_values.max(),
        'negative_ratio': (corr_values < 0).mean(),  # Should be close to 1.0
        'strong_negative_ratio': (corr_values < -0.3).mean(),  # Strong negative correlation
        'correlation_trend': np.polyfit(range(len(corr_values)), corr_values, 1)[0] if len(corr_values) > 1 else 0
    }


def analyze_entropy_dynamics(sans_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze the entropy of SANS weight distributions.
    
    Entropy indicates how focused the sampling is:
    - High entropy: More uniform sampling across negatives
    - Low entropy: Strong focus on specific negatives
    - Entropy ratio: Current entropy / max possible entropy
    
    Args:
        sans_df: DataFrame with SANS debugging metrics
    
    Returns:
        Dictionary with entropy statistics
    """
    if 'entropy_ratio' not in sans_df.columns:
        return {}
    
    entropy_ratios = sans_df['entropy_ratio'].dropna()
    
    return {
        'mean_entropy_ratio': entropy_ratios.mean(),
        'std_entropy_ratio': entropy_ratios.std(),
        'min_entropy_ratio': entropy_ratios.min(),
        'max_entropy_ratio': entropy_ratios.max(),
        'entropy_trend': np.polyfit(range(len(entropy_ratios)), entropy_ratios, 1)[0] if len(entropy_ratios) > 1 else 0,
        'low_entropy_ratio': (entropy_ratios < 0.5).mean(),  # Highly focused sampling
        'high_entropy_ratio': (entropy_ratios > 0.9).mean()  # Nearly uniform sampling
    }


def analyze_energy_separation(sans_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze the separation between positive and negative sample energies.
    
    Good training should show:
    - Clear separation between real and negative energies
    - Increasing separation over time
    - Stable energy values (not exploding or collapsing)
    
    Args:
        sans_df: DataFrame with SANS debugging metrics
    
    Returns:
        Dictionary with energy separation statistics
    """
    if 'real_energy_mean' not in sans_df.columns or 'neg_energy_mean' not in sans_df.columns:
        return {}
    
    real_energy = sans_df['real_energy_mean'].dropna()
    neg_energy = sans_df['neg_energy_mean'].dropna()
    
    if len(real_energy) == 0 or len(neg_energy) == 0:
        return {}
    
    # Energy separation (negative samples should have higher energy)
    separation = neg_energy - real_energy
    
    return {
        'mean_separation': separation.mean(),
        'std_separation': separation.std(),
        'min_separation': separation.min(),
        'max_separation': separation.max(),
        'positive_separation_ratio': (separation > 0).mean(),  # Should be close to 1.0
        'separation_trend': np.polyfit(range(len(separation)), separation, 1)[0] if len(separation) > 1 else 0,
        'real_energy_stability': real_energy.std() / (abs(real_energy.mean()) + 1e-8),
        'neg_energy_stability': neg_energy.std() / (abs(neg_energy.mean()) + 1e-8)
    }


def analyze_temperature_schedule(sans_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze the effectiveness of temperature scheduling.
    
    Args:
        sans_df: DataFrame with SANS debugging metrics
    
    Returns:
        Dictionary with temperature schedule statistics
    """
    if 'alpha_effective' not in sans_df.columns:
        return {}
    
    alpha_values = sans_df['alpha_effective'].dropna()
    
    if len(alpha_values) == 0:
        return {}
    
    return {
        'initial_alpha': alpha_values.iloc[0] if len(alpha_values) > 0 else None,
        'final_alpha': alpha_values.iloc[-1] if len(alpha_values) > 0 else None,
        'mean_alpha': alpha_values.mean(),
        'alpha_range': alpha_values.max() - alpha_values.min(),
        'alpha_trend': np.polyfit(range(len(alpha_values)), alpha_values, 1)[0] if len(alpha_values) > 1 else 0
    }


def compare_sans_configurations(experiments: Dict[str, str]) -> pd.DataFrame:
    """
    Compare multiple SANS configurations across experiments.
    
    Args:
        experiments: Dictionary mapping experiment names to result directories
    
    Returns:
        DataFrame with comparative statistics
    """
    results = []
    
    for exp_name, results_dir in experiments.items():
        metrics_df, sans_df = load_sans_metrics(results_dir)
        
        # Get final training metrics
        if len(metrics_df) > 0:
            final_loss = metrics_df['loss'].iloc[-1]
            best_loss = metrics_df['loss'].min()
            convergence_step = get_convergence_step(metrics_df)
        else:
            final_loss = best_loss = convergence_step = None
        
        # Get SANS-specific metrics
        corr_stats = analyze_correlation_quality(sans_df)
        entropy_stats = analyze_entropy_dynamics(sans_df)
        energy_stats = analyze_energy_separation(sans_df)
        temp_stats = analyze_temperature_schedule(sans_df)
        
        # Combine all statistics
        result = {
            'experiment': exp_name,
            'final_loss': final_loss,
            'best_loss': best_loss,
            'convergence_step': convergence_step,
            **{f'corr_{k}': v for k, v in corr_stats.items()},
            **{f'entropy_{k}': v for k, v in entropy_stats.items()},
            **{f'energy_{k}': v for k, v in energy_stats.items()},
            **{f'temp_{k}': v for k, v in temp_stats.items()}
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def get_convergence_step(df: pd.DataFrame, threshold: float = 0.1) -> int:
    """
    Find the step where the model has converged to within threshold of final loss.
    
    Args:
        df: DataFrame with loss values
        threshold: Fraction of initial loss reduction to consider as convergence
    
    Returns:
        Step number where convergence was achieved
    """
    if 'loss' not in df.columns or len(df) == 0:
        return -1
    
    initial_loss = df['loss'].iloc[0]
    final_loss = df['loss'].iloc[-1]
    target_loss = initial_loss - (1 - threshold) * (initial_loss - final_loss)
    
    conv_idx = df[df['loss'] <= target_loss].index
    if len(conv_idx) > 0:
        return df.loc[conv_idx[0], 'step']
    return df['step'].iloc[-1]


def plot_sans_diagnostics(sans_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create comprehensive diagnostic plots for SANS behavior.
    
    Args:
        sans_df: DataFrame with SANS debugging metrics
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('SANS Diagnostic Dashboard', fontsize=16, y=1.02)
    
    # 1. Correlation over time
    ax = axes[0, 0]
    if 'weight_energy_corr' in sans_df.columns:
        ax.plot(sans_df['step'], sans_df['weight_energy_corr'], 'b-', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=-0.3, color='g', linestyle='--', alpha=0.3, label='Good threshold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Weight-Energy Correlation')
        ax.set_title('Correlation Quality (should be negative)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Entropy ratio over time
    ax = axes[0, 1]
    if 'entropy_ratio' in sans_df.columns:
        ax.plot(sans_df['step'], sans_df['entropy_ratio'], 'r-', alpha=0.7)
        ax.fill_between(sans_df['step'], 0.3, 0.7, alpha=0.2, color='g', label='Optimal range')
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy Ratio')
        ax.set_title('Weight Distribution Entropy')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Energy separation
    ax = axes[0, 2]
    if 'real_energy_mean' in sans_df.columns and 'neg_energy_mean' in sans_df.columns:
        ax.plot(sans_df['step'], sans_df['real_energy_mean'], 'g-', label='Real', alpha=0.7)
        ax.plot(sans_df['step'], sans_df['neg_energy_mean'], 'r-', label='Negative', alpha=0.7)
        ax.fill_between(sans_df['step'], sans_df['real_energy_mean'], sans_df['neg_energy_mean'], 
                        alpha=0.2, color='b')
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Separation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Temperature schedule
    ax = axes[1, 0]
    if 'alpha_effective' in sans_df.columns:
        ax.plot(sans_df['step'], sans_df['alpha_effective'], 'purple', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('α (Temperature)')
        ax.set_title('Effective Temperature Schedule')
        ax.grid(True, alpha=0.3)
    
    # 5. Negative energy distribution
    ax = axes[1, 1]
    if 'neg_energy_std' in sans_df.columns:
        ax.plot(sans_df['step'], sans_df['neg_energy_std'], 'orange', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Std Dev')
        ax.set_title('Negative Energy Variance')
        ax.grid(True, alpha=0.3)
    
    # 6. Correlation vs Entropy scatter
    ax = axes[1, 2]
    if 'weight_energy_corr' in sans_df.columns and 'entropy_ratio' in sans_df.columns:
        scatter = ax.scatter(sans_df['entropy_ratio'], sans_df['weight_energy_corr'], 
                           c=sans_df['step'], cmap='viridis', alpha=0.6, s=10)
        ax.set_xlabel('Entropy Ratio')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlation vs Entropy Trade-off')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Step')
        ax.grid(True, alpha=0.3)
    
    # 7. Mean timestep evolution
    ax = axes[2, 0]
    if 't_mean' in sans_df.columns:
        ax.plot(sans_df['step'], sans_df['t_mean'], 'brown', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Timestep')
        ax.set_title('Diffusion Timestep Distribution')
        ax.grid(True, alpha=0.3)
    
    # 8. Energy range (min-max)
    ax = axes[2, 1]
    if 'neg_energy_min' in sans_df.columns and 'neg_energy_max' in sans_df.columns:
        ax.fill_between(sans_df['step'], sans_df['neg_energy_min'], sans_df['neg_energy_max'], 
                       alpha=0.3, color='purple', label='Negative range')
        if 'real_energy_min' in sans_df.columns and 'real_energy_max' in sans_df.columns:
            ax.fill_between(sans_df['step'], sans_df['real_energy_min'], sans_df['real_energy_max'], 
                          alpha=0.3, color='green', label='Real range')
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Ranges')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 9. Summary statistics box
    ax = axes[2, 2]
    ax.axis('off')
    
    # Calculate summary statistics
    summary_text = "Summary Statistics\n" + "="*25 + "\n"
    
    if 'weight_energy_corr' in sans_df.columns:
        mean_corr = sans_df['weight_energy_corr'].mean()
        summary_text += f"Mean Correlation: {mean_corr:.3f}\n"
        summary_text += f"Negative Corr %: {(sans_df['weight_energy_corr'] < 0).mean()*100:.1f}%\n"
    
    if 'entropy_ratio' in sans_df.columns:
        mean_entropy = sans_df['entropy_ratio'].mean()
        summary_text += f"Mean Entropy Ratio: {mean_entropy:.3f}\n"
    
    if 'real_energy_mean' in sans_df.columns and 'neg_energy_mean' in sans_df.columns:
        mean_sep = (sans_df['neg_energy_mean'] - sans_df['real_energy_mean']).mean()
        summary_text += f"Mean Energy Sep: {mean_sep:.3f}\n"
    
    if 'num_negatives' in sans_df.columns:
        summary_text += f"Num Negatives: {sans_df['num_negatives'].iloc[0]}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def find_optimal_hyperparameters(experiments_df: pd.DataFrame, 
                                metric: str = 'final_loss',
                                constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict:
    """
    Find optimal hyperparameters from experiment results.
    
    Args:
        experiments_df: DataFrame from compare_sans_configurations
        metric: Metric to optimize (minimize)
        constraints: Optional constraints on other metrics
    
    Returns:
        Dictionary with optimal configuration
    """
    df = experiments_df.copy()
    
    # Apply constraints if provided
    if constraints:
        for col, (min_val, max_val) in constraints.items():
            if col in df.columns:
                df = df[(df[col] >= min_val) & (df[col] <= max_val)]
    
    # Find best configuration
    if len(df) == 0:
        return {}
    
    best_idx = df[metric].idxmin() if metric in df.columns else 0
    best_config = df.iloc[best_idx].to_dict()
    
    return best_config


def generate_hyperparameter_report(experiments: Dict[str, str], save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive report comparing SANS hyperparameter configurations.
    
    Args:
        experiments: Dictionary mapping experiment names to result directories
        save_path: Optional path to save the report
    
    Returns:
        Report as string
    """
    comparison_df = compare_sans_configurations(experiments)
    
    report = []
    report.append("="*60)
    report.append("SANS HYPERPARAMETER TUNING REPORT")
    report.append("="*60)
    report.append("")
    
    # Best configuration by final loss
    best_config = find_optimal_hyperparameters(comparison_df, 'final_loss')
    if best_config:
        report.append("BEST CONFIGURATION (by final loss):")
        report.append("-"*40)
        report.append(f"Experiment: {best_config.get('experiment', 'N/A')}")
        report.append(f"Final Loss: {best_config.get('final_loss', 'N/A'):.6f}")
        report.append(f"Mean Correlation: {best_config.get('corr_mean_correlation', 'N/A'):.3f}")
        report.append(f"Mean Entropy Ratio: {best_config.get('entropy_mean_entropy_ratio', 'N/A'):.3f}")
        report.append("")
    
    # Configuration comparison table
    report.append("CONFIGURATION COMPARISON:")
    report.append("-"*40)
    
    if len(comparison_df) > 0:
        # Select key columns for display
        display_cols = ['experiment', 'final_loss', 'convergence_step', 
                       'corr_mean_correlation', 'entropy_mean_entropy_ratio', 
                       'energy_mean_separation']
        display_cols = [col for col in display_cols if col in comparison_df.columns]
        
        if display_cols:
            report.append(comparison_df[display_cols].to_string(index=False))
    
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text
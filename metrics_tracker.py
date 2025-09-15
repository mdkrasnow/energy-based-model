"""
Comprehensive metrics tracking system for energy-based model training.

This module provides a complete solution for monitoring training progress,
detecting overfitting, tracking curriculum effectiveness, and maintaining
energy landscape health during adversarial training.
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import threading
from dataclasses import dataclass
import warnings


@dataclass
class MetricWindow:
    """Container for a sliding window of metric values."""
    values: deque
    timestamps: deque
    
    def __post_init__(self):
        if len(self.values) != len(self.timestamps):
            raise ValueError("Values and timestamps must have same length")


class CurriculumMetricsTracker:
    """
    Comprehensive metrics tracking system for energy-based model training.
    
    Tracks training metrics, detects overfitting, monitors curriculum effectiveness,
    and maintains energy landscape health indicators.
    """
    
    def __init__(self, window_size: int = 1000, patience: int = 50):
        """
        Initialize the metrics tracker.
        
        Args:
            window_size: Size of sliding window for metric calculations
            patience: Number of steps without improvement for early stopping
        """
        self.window_size = window_size
        self.patience = patience
        self._lock = threading.Lock()
        
        # Core metric storage
        self.training_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.validation_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.energy_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.curriculum_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.robustness_metrics = defaultdict(lambda: deque(maxlen=window_size))
        
        # Timestamps for all metrics
        self.training_timestamps = deque(maxlen=window_size)
        self.validation_timestamps = deque(maxlen=window_size)
        self.energy_timestamps = deque(maxlen=window_size)
        self.curriculum_timestamps = deque(maxlen=window_size)
        self.robustness_timestamps = deque(maxlen=window_size)
        
        # Corruption tracking
        self.corruption_counts = defaultdict(int)
        
        # Stage tracking
        self.stage_history = deque(maxlen=window_size)
        self.stage_transitions = []
        self.stage_metrics = defaultdict(lambda: defaultdict(list))
        
        # Best metrics for early stopping
        self.best_val_loss = float('inf')
        self.best_val_step = 0
        self.steps_without_improvement = 0
        
        # Overfitting detection
        self.overfitting_alerts = []
        self.intervention_history = []
        
    def update_training_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Update training metrics.
        
        Args:
            step: Current training step
            metrics: Dictionary of metric name -> value
        """
        with self._lock:
            self.training_timestamps.append(step)
            for name, value in metrics.items():
                self.training_metrics[name].append(value)
    
    def update_validation_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Update validation metrics and check for improvements.
        
        Args:
            step: Current training step
            metrics: Dictionary of metric name -> value
        """
        with self._lock:
            self.validation_timestamps.append(step)
            for name, value in metrics.items():
                self.validation_metrics[name].append(value)
            
            # Update best validation loss for early stopping
            if 'loss' in metrics:
                val_loss = metrics['loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_step = step
                    self.steps_without_improvement = 0
                else:
                    self.steps_without_improvement += 1
    
    def update_energy_metrics(self, step: int, positive_energy: float, 
                            negative_energy: float, **kwargs) -> None:
        """
        Update energy landscape metrics.
        
        Args:
            step: Current training step
            positive_energy: Energy for positive samples
            negative_energy: Energy for negative samples
            **kwargs: Additional energy-related metrics
        """
        with self._lock:
            self.energy_timestamps.append(step)
            self.energy_metrics['positive_energy'].append(positive_energy)
            self.energy_metrics['negative_energy'].append(negative_energy)
            self.energy_metrics['energy_margin'].append(negative_energy - positive_energy)
            
            for name, value in kwargs.items():
                self.energy_metrics[name].append(value)
    
    def update_curriculum_metrics(self, step: int, stage: str, corruption_type: str,
                                epsilon: float, temperature: float = 1.0) -> None:
        """
        Update curriculum progression metrics.
        
        Args:
            step: Current training step
            stage: Current curriculum stage
            corruption_type: Type of corruption being used
            epsilon: Corruption strength parameter
            temperature: Sampling temperature
        """
        with self._lock:
            self.curriculum_timestamps.append(step)
            self.curriculum_metrics['stage'].append(stage)
            self.curriculum_metrics['corruption_type'].append(corruption_type)
            self.curriculum_metrics['epsilon'].append(epsilon)
            self.curriculum_metrics['temperature'].append(temperature)
            
            # Track stage transitions
            if len(self.stage_history) > 0 and self.stage_history[-1] != stage:
                self.stage_transitions.append((step, self.stage_history[-1], stage))
            
            self.stage_history.append(stage)
            self.update_corruption_counts(corruption_type)
    
    def update_corruption_counts(self, corruption_type: str) -> None:
        """Update counts of corruption type usage."""
        with self._lock:
            self.corruption_counts[corruption_type] += 1
    
    def update_robustness_metrics(self, step: int, clean_acc: float, 
                                adv_acc: float, attack_success_rate: float) -> None:
        """
        Update robustness metrics.
        
        Args:
            step: Current training step
            clean_acc: Clean accuracy
            adv_acc: Adversarial accuracy  
            attack_success_rate: Success rate of attacks
        """
        with self._lock:
            self.robustness_timestamps.append(step)
            self.robustness_metrics['clean_accuracy'].append(clean_acc)
            self.robustness_metrics['adversarial_accuracy'].append(adv_acc)
            self.robustness_metrics['attack_success_rate'].append(attack_success_rate)
    
    def calculate_train_val_gap(self) -> float:
        """
        Calculate current training-validation loss gap.
        
        Returns:
            Gap between training and validation loss
        """
        if (not self.training_metrics['loss'] or 
            not self.validation_metrics['loss']):
            return 0.0
        
        recent_train_loss = np.mean(list(self.training_metrics['loss'])[-10:])
        recent_val_loss = np.mean(list(self.validation_metrics['loss'])[-10:])
        
        return recent_val_loss - recent_train_loss
    
    def check_overfitting_signals(self) -> Dict[str, Any]:
        """
        Check for overfitting signals and return risk assessment.
        
        Returns:
            Dictionary with risk level, gap trend, and intervention recommendation
        """
        gap = self.calculate_train_val_gap()
        
        # Calculate trend in gap over recent history
        if len(self.training_metrics['loss']) < 20:
            gap_trend = 0.0
        else:
            recent_gaps = []
            train_losses = list(self.training_metrics['loss'])
            val_losses = list(self.validation_metrics['loss'])
            
            for i in range(-20, 0):
                if abs(i) <= len(val_losses):
                    recent_gaps.append(val_losses[i] - train_losses[i])
            
            if len(recent_gaps) > 1:
                gap_trend = self._calculate_trend(recent_gaps)
            else:
                gap_trend = 0.0
        
        # Determine risk level
        risk_level = "low"
        if gap > 0.1 and gap_trend > 0.01:
            risk_level = "high"
        elif gap > 0.05 or gap_trend > 0.005:
            risk_level = "medium"
        
        should_intervene = (risk_level == "high" and 
                          self.steps_without_improvement > self.patience // 2)
        
        return {
            'risk_level': risk_level,
            'gap': gap,
            'gap_trend': gap_trend,
            'should_intervene': should_intervene,
            'steps_without_improvement': self.steps_without_improvement
        }
    
    def get_patience_counter(self) -> int:
        """Get number of steps since best validation performance."""
        return self.steps_without_improvement
    
    def should_early_stop(self) -> bool:
        """Determine if training should be stopped early."""
        return self.steps_without_improvement >= self.patience
    
    def get_stage_metrics(self, stage_name: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific curriculum stage.
        
        Args:
            stage_name: Name of the curriculum stage
            
        Returns:
            Dictionary of metrics for the stage
        """
        stage_data = self.stage_metrics.get(stage_name, {})
        
        if not stage_data:
            return {}
        
        metrics = {}
        for metric_name, values in stage_data.items():
            if values:
                metrics[f"{metric_name}_mean"] = np.mean(values)
                metrics[f"{metric_name}_std"] = np.std(values)
                metrics[f"{metric_name}_final"] = values[-1]
        
        return metrics
    
    def calculate_stage_transition_impact(self) -> List[Dict[str, Any]]:
        """
        Calculate impact of stage transitions on loss.
        
        Returns:
            List of transition impacts
        """
        impacts = []
        
        for step, old_stage, new_stage in self.stage_transitions:
            # Find losses before and after transition
            before_losses = []
            after_losses = []
            
            train_losses = list(self.training_metrics['loss'])
            train_steps = list(self.training_timestamps)
            
            for i, ts in enumerate(train_steps):
                if ts <= step and ts > step - 50:  # 50 steps before
                    before_losses.append(train_losses[i])
                elif ts > step and ts <= step + 50:  # 50 steps after
                    after_losses.append(train_losses[i])
            
            if before_losses and after_losses:
                before_mean = np.mean(before_losses)
                after_mean = np.mean(after_losses)
                impact = after_mean - before_mean
                
                impacts.append({
                    'step': step,
                    'old_stage': old_stage,
                    'new_stage': new_stage,
                    'loss_impact': impact,
                    'relative_impact': impact / before_mean if before_mean > 0 else 0
                })
        
        return impacts
    
    def get_corruption_distribution(self) -> Dict[str, float]:
        """
        Get distribution of corruption types used.
        
        Returns:
            Dictionary mapping corruption type to usage percentage
        """
        if not self.corruption_counts:
            return {}
        
        total = sum(self.corruption_counts.values())
        return {k: v / total for k, v in self.corruption_counts.items()}
    
    def calculate_curriculum_effectiveness(self) -> float:
        """
        Calculate overall curriculum effectiveness score.
        
        Returns:
            Effectiveness score between 0 and 1
        """
        if len(self.validation_metrics['loss']) < 10:
            return 0.5  # Neutral score for insufficient data
        
        # Measure improvement over time
        val_losses = list(self.validation_metrics['loss'])
        initial_loss = np.mean(val_losses[:10])
        final_loss = np.mean(val_losses[-10:])
        
        if initial_loss <= 0:
            return 0.5
        
        improvement = (initial_loss - final_loss) / initial_loss
        
        # Factor in stage transition smoothness
        transition_impacts = self.calculate_stage_transition_impact()
        if transition_impacts:
            avg_impact = np.mean([abs(t['relative_impact']) for t in transition_impacts])
            smoothness_penalty = min(avg_impact, 0.3)  # Cap penalty at 0.3
        else:
            smoothness_penalty = 0.0
        
        effectiveness = max(0, min(1, improvement - smoothness_penalty))
        return effectiveness
    
    def calculate_energy_margin(self) -> float:
        """Calculate current energy margin between positive and negative samples."""
        if not self.energy_metrics['energy_margin']:
            return 0.0
        
        return np.mean(list(self.energy_metrics['energy_margin'])[-10:])
    
    def get_energy_variance(self) -> Dict[str, float]:
        """Get variance in energy outputs."""
        variances = {}
        
        for energy_type in ['positive_energy', 'negative_energy']:
            if self.energy_metrics[energy_type]:
                values = list(self.energy_metrics[energy_type])
                variances[energy_type] = np.var(values[-50:])  # Recent variance
        
        return variances
    
    def calculate_energy_stability(self) -> float:
        """
        Calculate energy function stability metric.
        
        Returns:
            Stability score between 0 and 1
        """
        if not self.energy_metrics['energy_margin']:
            return 0.5
        
        margins = list(self.energy_metrics['energy_margin'])[-50:]
        
        if len(margins) < 10:
            return 0.5
        
        # Stability based on margin consistency and trend
        margin_std = np.std(margins)
        margin_trend = self._calculate_trend(margins)
        
        # Good stability: low variance, positive or stable trend
        stability = max(0, min(1, 1 - margin_std/10 + margin_trend/10))
        return stability
    
    def get_gradient_health(self) -> Dict[str, Any]:
        """
        Check for vanishing/exploding gradients.
        
        Returns:
            Dictionary with gradient health indicators
        """
        if 'gradient_norm' not in self.training_metrics:
            return {'status': 'unknown', 'reason': 'no gradient data'}
        
        grad_norms = list(self.training_metrics['gradient_norm'])[-50:]
        
        if len(grad_norms) < 5:
            return {'status': 'insufficient_data'}
        
        mean_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms)
        trend = self._calculate_trend(grad_norms)
        
        status = 'healthy'
        issues = []
        
        if mean_norm < 1e-6:
            status = 'vanishing'
            issues.append('Very small gradient norms')
        elif mean_norm > 100:
            status = 'exploding'
            issues.append('Very large gradient norms')
        
        if std_norm / mean_norm > 2:
            status = 'unstable'
            issues.append('High gradient variance')
        
        return {
            'status': status,
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'trend': trend,
            'issues': issues
        }
    
    def calculate_robustness_tradeoff(self) -> Dict[str, float]:
        """
        Calculate trade-off between clean and robust accuracy.
        
        Returns:
            Dictionary with trade-off metrics
        """
        if (not self.robustness_metrics['clean_accuracy'] or 
            not self.robustness_metrics['adversarial_accuracy']):
            return {}
        
        clean_acc = list(self.robustness_metrics['clean_accuracy'])[-10:]
        adv_acc = list(self.robustness_metrics['adversarial_accuracy'])[-10:]
        
        clean_mean = np.mean(clean_acc)
        adv_mean = np.mean(adv_acc)
        
        return {
            'clean_accuracy': clean_mean,
            'adversarial_accuracy': adv_mean,
            'accuracy_gap': clean_mean - adv_mean,
            'robustness_ratio': adv_mean / clean_mean if clean_mean > 0 else 0
        }
    
    def get_pareto_score(self, clean_weight: float = 0.5) -> float:
        """
        Calculate combined clean and robust performance score.
        
        Args:
            clean_weight: Weight for clean accuracy (1-clean_weight for robust)
            
        Returns:
            Combined Pareto score
        """
        tradeoff = self.calculate_robustness_tradeoff()
        
        if not tradeoff:
            return 0.0
        
        clean_acc = tradeoff['clean_accuracy']
        adv_acc = tradeoff['adversarial_accuracy']
        
        return clean_weight * clean_acc + (1 - clean_weight) * adv_acc
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get comprehensive current status of all metrics.
        
        Returns:
            Dictionary with current status across all tracked metrics
        """
        status = {
            'training_step': max(self.training_timestamps) if self.training_timestamps else 0,
            'overfitting_check': self.check_overfitting_signals(),
            'early_stopping': {
                'should_stop': self.should_early_stop(),
                'patience_counter': self.get_patience_counter(),
                'best_val_loss': self.best_val_loss
            },
            'energy_health': {
                'margin': self.calculate_energy_margin(),
                'stability': self.calculate_energy_stability(),
                'variance': self.get_energy_variance()
            },
            'curriculum': {
                'effectiveness': self.calculate_curriculum_effectiveness(),
                'corruption_distribution': self.get_corruption_distribution()
            },
            'gradient_health': self.get_gradient_health()
        }
        
        # Add robustness if available
        if self.robustness_metrics['clean_accuracy']:
            status['robustness'] = {
                'tradeoff': self.calculate_robustness_tradeoff(),
                'pareto_score': self.get_pareto_score()
            }
        
        return status
    
    def get_summary_report(self) -> str:
        """
        Get formatted summary report of key metrics.
        
        Returns:
            Formatted string with summary report
        """
        status = self.get_current_status()
        
        report = []
        report.append("=== METRICS SUMMARY ===")
        report.append(f"Training Step: {status['training_step']}")
        report.append("")
        
        # Overfitting status
        overfitting = status['overfitting_check']
        report.append(f"Overfitting Risk: {overfitting['risk_level'].upper()}")
        report.append(f"Train-Val Gap: {overfitting['gap']:.4f}")
        report.append(f"Steps Without Improvement: {overfitting['steps_without_improvement']}")
        report.append("")
        
        # Energy health
        energy = status['energy_health']
        report.append(f"Energy Margin: {energy['margin']:.4f}")
        report.append(f"Energy Stability: {energy['stability']:.3f}")
        report.append("")
        
        # Curriculum effectiveness
        curriculum = status['curriculum']
        report.append(f"Curriculum Effectiveness: {curriculum['effectiveness']:.3f}")
        
        corruption_dist = curriculum['corruption_distribution']
        if corruption_dist:
            report.append("Corruption Distribution:")
            for corruption, pct in corruption_dist.items():
                report.append(f"  {corruption}: {pct:.1%}")
        report.append("")
        
        # Gradient health
        grad_health = status['gradient_health']
        if grad_health['status'] != 'unknown':
            report.append(f"Gradient Health: {grad_health['status'].upper()}")
            if 'mean_norm' in grad_health:
                report.append(f"Mean Gradient Norm: {grad_health['mean_norm']:.2e}")
        report.append("")
        
        # Robustness (if available)
        if 'robustness' in status:
            robustness = status['robustness']['tradeoff']
            report.append(f"Clean Accuracy: {robustness['clean_accuracy']:.3f}")
            report.append(f"Adversarial Accuracy: {robustness['adversarial_accuracy']:.3f}")
            report.append(f"Pareto Score: {status['robustness']['pareto_score']:.3f}")
        
        return "\n".join(report)
    
    def export_metrics_to_dict(self) -> Dict[str, Any]:
        """
        Export all metrics for saving/serialization.
        
        Returns:
            Dictionary with all tracked metrics and metadata
        """
        return {
            'config': {
                'window_size': self.window_size,
                'patience': self.patience
            },
            'training_metrics': {k: list(v) for k, v in self.training_metrics.items()},
            'validation_metrics': {k: list(v) for k, v in self.validation_metrics.items()},
            'energy_metrics': {k: list(v) for k, v in self.energy_metrics.items()},
            'curriculum_metrics': {k: list(v) for k, v in self.curriculum_metrics.items()},
            'robustness_metrics': {k: list(v) for k, v in self.robustness_metrics.items()},
            'timestamps': {
                'training': list(self.training_timestamps),
                'validation': list(self.validation_timestamps),
                'energy': list(self.energy_timestamps),
                'curriculum': list(self.curriculum_timestamps),
                'robustness': list(self.robustness_timestamps)
            },
            'corruption_counts': dict(self.corruption_counts),
            'stage_transitions': self.stage_transitions,
            'best_metrics': {
                'best_val_loss': self.best_val_loss,
                'best_val_step': self.best_val_step,
                'steps_without_improvement': self.steps_without_improvement
            }
        }
    
    def get_warning_messages(self) -> List[str]:
        """
        Get list of current warnings and alerts.
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check overfitting
        overfitting = self.check_overfitting_signals()
        if overfitting['risk_level'] == 'high':
            warnings.append("HIGH OVERFITTING RISK: Consider regularization or early stopping")
        elif overfitting['risk_level'] == 'medium':
            warnings.append("Medium overfitting risk detected")
        
        if overfitting['should_intervene']:
            warnings.append("INTERVENTION RECOMMENDED: Training may benefit from regularization")
        
        # Check early stopping
        if self.should_early_stop():
            warnings.append("EARLY STOPPING TRIGGERED: No improvement in validation loss")
        
        # Check gradient health
        grad_health = self.get_gradient_health()
        if grad_health['status'] == 'vanishing':
            warnings.append("VANISHING GRADIENTS: Consider adjusting learning rate or architecture")
        elif grad_health['status'] == 'exploding':
            warnings.append("EXPLODING GRADIENTS: Consider gradient clipping or lower learning rate")
        elif grad_health['status'] == 'unstable':
            warnings.append("UNSTABLE GRADIENTS: High variance in gradient norms")
        
        # Check energy margin
        if self.energy_metrics['energy_margin']:
            margin = self.calculate_energy_margin()
            if margin < 0.1:
                warnings.append("LOW ENERGY MARGIN: Energy function may not be well-separated")
            elif margin < 0:
                warnings.append("NEGATIVE ENERGY MARGIN: Energy function ordering is problematic")
        
        return warnings
    
    # Helper methods
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values using least squares."""
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            return coeffs[0]  # Return slope
        except np.linalg.LinAlgError:
            return 0.0
    
    def _moving_average(self, values: List[float], window: int = 10) -> List[float]:
        """Calculate moving average of values."""
        if len(values) < window:
            return values
        
        return [np.mean(values[i:i+window]) for i in range(len(values) - window + 1)]
    
    def _smooth_values(self, values: List[float], alpha: float = 0.1) -> List[float]:
        """Apply exponential smoothing to values."""
        if not values:
            return []
        
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed_val = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_val)
        
        return smoothed
    
    def _calculate_percentiles(self, values: List[float], 
                             percentiles: List[int] = [25, 50, 75]) -> Dict[int, float]:
        """Calculate percentiles of values."""
        if not values:
            return {p: 0.0 for p in percentiles}
        
        return {p: np.percentile(values, p) for p in percentiles}


# Utility functions for external use
def create_metrics_tracker(window_size: int = 1000, patience: int = 50) -> CurriculumMetricsTracker:
    """Create a configured metrics tracker instance."""
    return CurriculumMetricsTracker(window_size=window_size, patience=patience)


def load_metrics_from_dict(data: Dict[str, Any]) -> CurriculumMetricsTracker:
    """
    Load metrics tracker from exported dictionary.
    
    Args:
        data: Dictionary from export_metrics_to_dict()
        
    Returns:
        Configured CurriculumMetricsTracker instance
    """
    config = data.get('config', {})
    tracker = CurriculumMetricsTracker(
        window_size=config.get('window_size', 1000),
        patience=config.get('patience', 50)
    )
    
    # Restore metrics
    for name, values in data.get('training_metrics', {}).items():
        tracker.training_metrics[name].extend(values)
    
    for name, values in data.get('validation_metrics', {}).items():
        tracker.validation_metrics[name].extend(values)
    
    for name, values in data.get('energy_metrics', {}).items():
        tracker.energy_metrics[name].extend(values)
    
    for name, values in data.get('curriculum_metrics', {}).items():
        tracker.curriculum_metrics[name].extend(values)
    
    for name, values in data.get('robustness_metrics', {}).items():
        tracker.robustness_metrics[name].extend(values)
    
    # Restore timestamps
    timestamps = data.get('timestamps', {})
    if 'training' in timestamps:
        tracker.training_timestamps.extend(timestamps['training'])
    if 'validation' in timestamps:
        tracker.validation_timestamps.extend(timestamps['validation'])
    if 'energy' in timestamps:
        tracker.energy_timestamps.extend(timestamps['energy'])
    if 'curriculum' in timestamps:
        tracker.curriculum_timestamps.extend(timestamps['curriculum'])
    if 'robustness' in timestamps:
        tracker.robustness_timestamps.extend(timestamps['robustness'])
    
    # Restore other state
    tracker.corruption_counts.update(data.get('corruption_counts', {}))
    tracker.stage_transitions = data.get('stage_transitions', [])
    
    best_metrics = data.get('best_metrics', {})
    tracker.best_val_loss = best_metrics.get('best_val_loss', float('inf'))
    tracker.best_val_step = best_metrics.get('best_val_step', 0)
    tracker.steps_without_improvement = best_metrics.get('steps_without_improvement', 0)
    
    return tracker
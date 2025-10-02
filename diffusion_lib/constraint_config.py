"""
Constraint configuration for adversarial corruption in IRED.

This module provides configuration management for constraint-aware adversarial
corruption, ensuring that negative samples respect task-specific constraints
(e.g., low-rank structure for matrix completion).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ConstraintConfig:
    """Configuration for constraint-aware adversarial corruption.
    
    Attributes:
        enabled: Whether constraint-aware corruption is enabled
        constraint_type: Type of constraint ('none', 'low_rank', etc.)
        constraint_params: Parameters specific to the constraint type
    """
    enabled: bool = False
    constraint_type: str = 'none'  # 'none', 'low_rank', future: 'positive_semidefinite', 'orthogonal'
    constraint_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraint_params is None:
            self.constraint_params = {}
    
    @classmethod
    def from_flags(cls, use_constraint: bool, rank: int, dataset_name: str) -> 'ConstraintConfig':
        """Factory method to create config from command-line flags.
        
        Args:
            use_constraint: Whether to enable constraint-aware corruption
            rank: Rank parameter for low-rank tasks
            dataset_name: Name of the dataset to determine constraint type
            
        Returns:
            ConstraintConfig instance configured for the given task
        """
        if not use_constraint:
            return cls(enabled=False, constraint_type='none')
        
        # Determine constraint type from dataset
        if dataset_name in ['lowrank', 'matrix_completion']:
            return cls(
                enabled=True,
                constraint_type='low_rank',
                constraint_params={'rank': rank}
            )
        # For unconstrained tasks (addition, inverse), keep disabled even if flag is True
        # This provides safety - the flag only affects tasks with known constraints
        else:
            return cls(enabled=False, constraint_type='none')
    
    def __str__(self) -> str:
        if not self.enabled:
            return "ConstraintConfig(disabled)"
        return f"ConstraintConfig(type={self.constraint_type}, params={self.constraint_params})"
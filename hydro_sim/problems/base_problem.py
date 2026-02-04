# problems/base_problem.py
"""
Base class for all hydrodynamic problem cases.
Provides a common interface that specific problem types inherit from.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProblemCase(ABC):
    """
    Abstract base class for hydrodynamic problem configurations.
    
    All problem types (Riemann, DrivenShock, Sedov, etc.) inherit from this
    base class to ensure a consistent interface.
    
    Attributes:
        x_min: Left domain boundary
        x_max: Right domain boundary
        t_end: Simulation end time
        title: Descriptive name for the problem
    """
    x_min: float
    x_max: float
    t_end: float
    title: str = ""
    
    @property
    def domain_length(self) -> float:
        """Total length of the computational domain."""
        return self.x_max - self.x_min

# problems/base_problem.py
"""
Base class for all hydrodynamic problem cases.
Provides a common interface that specific problem types inherit from.

Physical parameters (what physics to simulate) belong here.
Numerical parameters (how to run the solver) belong in SimulationConfig.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Any


@dataclass(frozen=True)
class ProblemCase(ABC):
    """
    Abstract base class for hydrodynamic problem configurations.
    
    All problem types (Riemann, DrivenShock, Sedov, etc.) inherit from this
    base class to ensure a consistent interface.
    """
    # Common to all problems
    gamma: float
    x_min: float
    x_max: float
    t_end: float
    title: str = ""
    
    # Geometry (can be overridden by subclasses)
    geom: Any = None  # Will be set to planar() by default in subclasses
    
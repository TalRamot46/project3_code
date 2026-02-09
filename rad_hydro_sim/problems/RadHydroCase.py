# problems/base_problem.py
"""
Base class for all hydrodynamic problem cases.
Provides a common interface that specific problem types inherit from.

Physical parameters (what physics to simulate) belong here.
Numerical parameters (how to run the solver) belong in SimulationConfig.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Any, Tuple


@dataclass(frozen=True)
class RadHydroCase(ABC):
    """
    Abstract base class for radiation hydrodynamic problem configurations.
    
    All problem types (Riemann, DrivenShock, Sedov, etc.) inherit from this
    base class to ensure a consistent interface.
    """
    # Rosen's opacity parameters
    g: float
    alpha: float
    lambda_: float

    # Rosen's specific energy parameters
    f: float
    gamma: float
    mu: float

    # coupling factor
    chi: float = 1000

    # Boundar conditions
    T_0: float = 100.0
    tau: float = 1.0

    # Initial conditions0
    
    # grid parameters
    Ncells: int = 1000
    CFL: float = 1/3
    sigma_visc: float = 1.0
    x_min: float = 0.0
    x_max: float = 1.0e-3
    t_end: float = 1.0e-9
    title: str = ""
    
    # Geometry (can be overridden by subclasses)
    geom: Any = None  # Will be set to planar() by default in subclasses

def _get_params(self) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Returns the parameters as a tuple"""
    return self.alpha, self.gamma, self.mu, self.f, self.chi, self.lambda_, self.g, self.T_0, self.tau
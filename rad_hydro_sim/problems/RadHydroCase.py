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

from project_3.hydro_sim.core.geometry import Geometry, planar


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
    chi: float

    # Boundary conditions
    T0: float
    tau: float

    # initial conditions
    rho0: float 
    p0: float
    u0: float

    # adiabatic index
    r: float # r = \gamma_adiabatic - 1

    # Initial conditions0
    
    # grid parameters
    x_min: float
    x_max: float
    t_end: float 
    title: str = ""
    
    # Geometry
    geom: Geometry = planar()  # Default to planar geometry

    def _get_params(
        self,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Returns the parameters as a tuple for radiation step"""
        return self.alpha, self.gamma, self.mu, self.f, self.chi, self.lambda_, self.g
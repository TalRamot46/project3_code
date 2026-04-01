# problems/base_problem.py
"""
Base class for all hydrodynamic problem cases.
Provides a common interface that specific problem types inherit from.

Physical parameters (what physics to simulate) belong here.
Numerical parameters (how to run the solver) belong in SimulationConfig.
"""
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

import numpy as np

from project3_code.hydro_sim.core.geometry import Geometry, planar


@dataclass(frozen=True)
class RadHydroCase(ABC):
    """
    Abstract base class for radiation hydrodynamic problem configurations.
    
    All problem types (Riemann, DrivenShock, Sedov, etc.) inherit from this
    base class to ensure a consistent interface.
    """
    # Rosen's opacity parameters
    g_Kelvin: float # defined such that e = g_Kelvin * T_Kelvin^alpha * rho^lambda_
    alpha: float
    lambda_: float

    # Rosen's specific energy parameters
    f_Kelvin: float # defined such that e = f_Kelvin * T_Kelvin^gamma * mu^mu * rho^rho
    beta_Rosen: float
    mu: float

    # coupling factor
    chi: float

    # Boundary conditions
    T0_Kelvin: float | Optional[float]
    P0_Barye: float | Optional[float]
    tau: float

    # initial conditions
    rho0: float | Optional[float]
    p0: float | Optional[float]
    u0: float | Optional[float]

    T_initial_Kelvin: float | Optional[float]

    # adiabatic index
    r: float # r = \gamma_adiabatic - 1

    # Initial conditions0
    
    # grid parameters
    x_min: float
    x_max: float
    t_sec_end: float 

    # for flags
    initial_condition: str # e.g. "pressure, velocity, density", 
    scenario: str # "hydro_only", "radiation_only", "full_rad_hydro"
    
    # Optional fields with defaults
    title: str = ""
    # Right BC for radiation energy solve:
    # - "Dirichlet": hard wall with fixed right temperature (uses T_initial_Kelvin)
    # - "Neuman": trivial Neumann / "Vaccum" (zero gradient at right edge)
    # - "free": unconstrained right edge in the solve; boundary is extrapolated after solve
    right_BC: Literal["Dirichlet", "Neuman", "free"] = "Dirichlet"
    # Kept for diffusion/reference compatibility and optional right Dirichlet value.
    T_right_Kelvin: Optional[float] = 0.0

    # Geometry
    geom: Geometry = planar()  # Default to planar geometry
    force_black: Literal["gray corrected", "full black"] | None = None
    # Tridiagonal coefficient formulation for gray radiation solve:
    # - "legacy": existing center-weighted discretization
    # - "face_weighted": doc-aligned form using rho_face and center D_i terms
    radiation_coeff_scheme: Literal["legacy", "face_weighted"] = "face_weighted"
    times_for_png: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )

    def _get_params(
        self,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Returns the parameters as a tuple for radiation step"""
        return self.alpha, self.beta_Rosen, self.mu, self.f_Kelvin, self.chi, self.lambda_, self.g_Kelvin
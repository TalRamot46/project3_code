# problems/sedov_problem.py
"""
Sedov-Taylor point explosion problem setup.

The Sedov explosion is a classical self-similar solution for a strong
point explosion in a uniform medium. It's an excellent test for
spherical/cylindrical hydrodynamics codes.
"""
import numpy as np
from dataclasses import dataclass
from typing import Any

# Handle both package import and direct run cases
try:
    from .base_problem import ProblemCase
    from ..core.eos import internal_energy_from_prho
    from ..core.grid import cell_volumes, masses_from_initial_rho
    from ..core.state import HydroState
    from ..core.geometry import spherical, cylindrical, planar
except ImportError:
    from problems.base_problem import ProblemCase
    from core.eos import internal_energy_from_prho
    from core.grid import cell_volumes, masses_from_initial_rho
    from core.state import HydroState
    from core.geometry import spherical, cylindrical, planar


@dataclass(frozen=True)
class SedovExplosionCase(ProblemCase):
    """
    Configuration for Sedov-Taylor point explosion problem.
    
    The Sedov problem models a strong explosion from a point source
    releasing energy E0 into a uniform ambient medium with density rho0
    and negligible initial pressure p0.
    
    Unique Attributes:
        E0: Total explosion energy deposited at the origin
        rho0: Ambient (background) density
        p0: Ambient pressure (should be very small compared to explosion)
        r_init: Radius within which initial energy is deposited (0 = first cell)
    """
    # Unique to Sedov problems
    E0: float = 1.0
    rho0: float = 1.0
    p0: float = 1e-6
    r_init: float = 0.0  # Will be set based on grid if 0


# Pre-defined Sedov test cases
SEDOV_TEST_CASES = {
    "spherical": SedovExplosionCase(
        gamma=1.4,
        x_min=0.0, x_max=1.2, t_end=0.5,
        E0=0.851072, rho0=1.0, p0=1e-6,
        geom=spherical(),
        title="Standard Spherical Sedov"
    ),
    "cylindrical": SedovExplosionCase(
        gamma=1.4,
        x_min=0.0, x_max=1.2, t_end=0.5,
        E0=0.311357, rho0=1.0, p0=1e-6,
        geom=cylindrical(),
        title="Standard Cylindrical Sedov"
    ),
    "planar": SedovExplosionCase(
        gamma=1.4,
        x_min=0.0, x_max=1.2, t_end=0.5,
        E0=0.5, rho0=1.0, p0=1e-6,
        geom=planar(),
        title="Standard Planar Sedov"
    ),
    "strong_spherical": SedovExplosionCase(
        gamma=1.4,
        x_min=0.0, x_max=2.0, t_end=0.5,
        E0=10.0, rho0=1.0, p0=1e-8,
        geom=spherical(),
        title="Strong Point Explosion"
    ),
}

# Legacy aliases
SEDOV_TEST_CASES["standard_spherical"] = SEDOV_TEST_CASES["spherical"]
SEDOV_TEST_CASES["standard_cylindrical"] = SEDOV_TEST_CASES["cylindrical"]
SEDOV_TEST_CASES["standard_planar"] = SEDOV_TEST_CASES["planar"]
SEDOV_TEST_CASES["strong_explosion"] = SEDOV_TEST_CASES["strong_spherical"]


def init_sedov(x_nodes: np.ndarray, case: SedovExplosionCase) -> tuple:
    """
    Initialize HydroState and cell masses for a Sedov explosion problem.
    
    The initial condition places all explosion energy E0 in the innermost
    cell(s) as thermal energy, with the rest of the domain at ambient
    conditions (rho0, p0).
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    N = x_nodes.size - 1
    if N < 2:
        raise ValueError("Need at least 2 cells.")
    
    geom = case.geom if case.geom is not None else spherical()
    gamma = case.gamma
    
    # Cell centers
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    
    # Uniform ambient conditions
    rho = np.full(N, case.rho0)
    p = np.full(N, case.p0)
    
    # Compute cell volumes
    V_cells = cell_volumes(x_nodes, geom)
    
    # Determine initialization radius
    # If r_init is 0 or not set, use the first cell
    r_init = case.r_init if case.r_init > 0 else x_nodes[1]
    
    # Find cells within the initialization region
    init_mask = x_cells <= r_init
    
    if not np.any(init_mask):
        # At minimum, deposit energy in the first cell
        init_mask[0] = True
    
    # Total volume of initialization region
    V_init = np.sum(V_cells[init_mask])
    
    # Energy density in the initialization region
    # E0 = integral of rho*e over V_init
    # For uniform initial density: E0 = rho0 * e_init * V_init
    # => e_init = E0 / (rho0 * V_init)
    e_init = case.E0 / (case.rho0 * V_init)
    
    # Corresponding pressure from EOS: p = (gamma - 1) * rho * e
    p_init = (gamma - 1.0) * case.rho0 * e_init
    
    # Set initial pressure in the explosion region
    p[init_mask] = p_init
    
    # Compute internal energy from pressure and density
    e = internal_energy_from_prho(p, rho, gamma)
    
    # Artificial viscosity (initially zero)
    q = np.zeros_like(rho)
    
    # All nodes start at rest
    u_nodes = np.zeros(N + 1)
    a_nodes = np.zeros(N + 1)
    
    # Lagrangian masses (conserved)
    m_cells = masses_from_initial_rho(x_nodes, rho, geom)
    
    state = HydroState(
        t=0.0,
        x=x_nodes.copy(),
        u=u_nodes,
        a=a_nodes,
        V=V_cells,
        rho=rho,
        e=e,
        p=p,
        q=q
    )
    
    return state, m_cells


# Legacy alias
init_sedov_explosion = lambda x_nodes, geom, gamma, case: init_sedov(x_nodes, case)
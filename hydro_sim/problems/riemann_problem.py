# problems/riemann_problem.py
"""
Riemann shock tube problem setup.

Classic test problem for compressible gas dynamics: two uniform states
separated by a discontinuity at t=0, with exact solution available.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Handle both package import and direct run cases
try:
    from .Hydro_case import HydroCase
    from ..core.eos import internal_energy_from_prho
    from ..core.grid import cell_volumes, masses_from_initial_rho
    from ..core.state import HydroState
    from ..core.geometry import planar
except (ImportError, ValueError):
    # Fallback for when running as a standalone script
    import sys
    from pathlib import Path
    _parent_dir = str(Path(__file__).parent.parent.absolute())
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from problems.Hydro_case import HydroCase
    from core.eos import internal_energy_from_prho
    from core.grid import cell_volumes, masses_from_initial_rho
    from core.state import HydroState
    from core.geometry import planar


@dataclass(frozen=True)
class RiemannCase(HydroCase):
    """
    Configuration for Riemann (shock tube) problems.
    Defines left and right states separated by a discontinuity at x0.
    
    Unique Attributes:
        left: Tuple of (rho, u, p) for left state
        right: Tuple of (rho, u, p) for right state
        x0: Position of initial discontinuity
    """
    # Unique to Riemann problems
    left: Tuple[float, float, float] = (1.0, 0.0, 1.0)      # (rho, u, p)
    right: Tuple[float, float, float] = (0.125, 0.0, 0.1)   # (rho, u, p)
    x0: float = 0.0


# Pre-defined Riemann test cases
RIEMANN_TEST_CASES = {
    "sod": RiemannCase(
        gamma=1.4,
        left=(1.0, 0.0, 1.0), 
        right=(0.125, 0.0, 0.1),
        x_min=-1.0, x_max=1.0, t_end=0.25,
        geom=planar(),
        title="Sod shock tube"
    ),
    "strong_shock": RiemannCase(
        gamma=1.4,
        left=(1.0, 0.0, 1000.0), 
        right=(1.0, 0.0, 0.01),
        x_min=-1.0, x_max=1.0, t_end=0.012,
        geom=planar(),
        title="Strong pressure jump"
    ),
    "reverse_shock": RiemannCase(
        gamma=1.4,
        left=(1.0, 0.0, 0.01), 
        right=(1.0, 0.0, 100.0),
        x_min=-1.0, x_max=1.0, t_end=0.035,
        geom=planar(),
        title="Reverse pressure jump"
    ),
    "colliding": RiemannCase(
        gamma=1.4,
        left=(5.99924, 19.5975, 460.894), 
        right=(5.99242, -6.19633, 46.0950),
        x_min=-1.0, x_max=1.0, t_end=0.035,
        geom=planar(),
        title="Colliding streams"
    ),
}

# Legacy integer-keyed access for backward compatibility
RIEMANN_TEST_CASES[1] = RIEMANN_TEST_CASES["sod"]
RIEMANN_TEST_CASES[2] = RIEMANN_TEST_CASES["strong_shock"]
RIEMANN_TEST_CASES[3] = RIEMANN_TEST_CASES["reverse_shock"]
RIEMANN_TEST_CASES[4] = RIEMANN_TEST_CASES["colliding"]


def init_riemann(x_nodes: np.ndarray, case: RiemannCase) -> tuple:
    """Initialize HydroState and cell masses for a Riemann problem. """
    geom = case.geom if case.geom is not None else planar()
    gamma = case.gamma
    x0 = case.x0
    
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])

    rhoL, uL, pL = case.left
    rhoR, uR, pR = case.right

    left_mask = x_cells < x0

    rho = np.where(left_mask, rhoL, rhoR)
    p = np.where(left_mask, pL, pR)
    e = internal_energy_from_prho(p, rho, gamma)
    q = np.zeros_like(rho)

    # Node velocities: piecewise constant using node position
    u_nodes = np.where(x_nodes < x0, uL, uR)

    V = cell_volumes(x_nodes, geom)
    m = masses_from_initial_rho(x_nodes, rho, geom)

    a_nodes = np.zeros_like(x_nodes)

    state = HydroState(t=0.0, x=x_nodes, u=u_nodes, a=a_nodes, V=V, rho=rho, e=e, p=p, q=q)
    return state, m


# Legacy alias
init_planar_riemann_case = lambda x_nodes, geom, gamma, case, x0=0.0: init_riemann(x_nodes, case)

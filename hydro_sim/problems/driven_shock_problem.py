# problems/driven_shock_problem.py
"""
Driven shock problem setup.

Models a shock driven by a time-dependent pressure boundary condition
at the left boundary, propagating into a uniform medium.
"""
import numpy as np
from dataclasses import dataclass

# Handle both package import and direct run cases
try:
    from .Hydro_case import HydroCase
    from ..core.state import HydroState
    from ..core.eos import internal_energy_from_prho
    from ..core.grid import cell_volumes, masses_from_initial_rho
    from ..core.geometry import planar
except (ImportError, ValueError):
    # Fallback for when running as a standalone script
    import sys
    from pathlib import Path
    _parent_dir = str(Path(__file__).parent.parent.absolute())
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from project_3.hydro_sim.problems.Hydro_case import HydroCase
    from project_3.hydro_sim.core.state import HydroState
    from project_3.hydro_sim.core.eos import internal_energy_from_prho
    from project_3.hydro_sim.core.grid import cell_volumes, masses_from_initial_rho
    from project_3.hydro_sim.core.geometry import planar


@dataclass(frozen=True)
class DrivenShockCase(HydroCase):
    """
    Configuration for driven shock problems.
    
    Models a shock driven by a time-dependent boundary condition
    (pressure) at the left boundary: p(t) = P0 * t^tau
    
    Unique Attributes:
        rho0: Initial uniform density
        p0: Initial uniform pressure (should be small)
        u0: Initial uniform velocity (typically 0)
        P0: Pressure amplitude at boundary
        tau: Power-law exponent for pressure drive: p ~ t^tau
    """
    # Unique to driven shock problems
    rho0: float = 1.0
    p0: float = 1e-6
    u0: float = 0.0
    P0: float = 1.0
    tau: float = 0.0  # tau=0 means constant pressure drive

    def p_left(self, t: float) -> float:
        """Compute boundary pressure at time t."""
        return self.P0 * (t ** self.tau) if t > 0 else 0.0


# Pre-defined driven shock test cases
DRIVEN_SHOCK_TEST_CASES = {
    "constant_drive": DrivenShockCase(
        gamma=5/3,
        x_min=0.0, x_max=1.0, t_end=0.5,
        rho0=1.0, p0=1e-6, u0=0.0,
        P0=1.0, tau=0.0,
        geom=planar(),
        title="Constant pressure drive"
    ),
    "linear_drive": DrivenShockCase(
        gamma=5/3,
        x_min=0.0, x_max=1.0, t_end=0.5,
        rho0=1.0, p0=1e-6, u0=0.0,
        P0=1.0, tau=1.0,
        geom=planar(),
        title="Linear pressure drive (τ=1)"
    ),
    "gold_wall": DrivenShockCase(
        gamma=1.25,
        x_min=0.0, x_max=3e-6 / 19.32, t_end=100e-9,
        rho0=19.32, p0=1e-3, u0=0.0,
        P0=10.0, tau=0.0,
        geom=planar(),
        title="Gold wall (constant drive)"
    ),
    "gold_wall_continuous": DrivenShockCase(
        gamma=1.25,
        x_min=0.0, x_max=3e-3 / 19.32, t_end=5e-3,
        rho0=19.32, p0=1e-3, u0=0.0,
        P0=10.0, tau=1.0,
        geom=planar(),
        title="Gold wall (continuous drive, τ=1)"
    ),
}


def init_driven_shock(x_nodes: np.ndarray, case: DrivenShockCase) -> tuple:
    """
    Initialize HydroState and cell masses for a driven shock problem.
    
    Parameters:
        x_nodes: Node positions (N+1 values for N cells)
        case: DrivenShockCase with problem parameters
        
    Returns:
        state: Initial HydroState
        m: Cell masses (fixed in Lagrangian formulation)
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    N = x_nodes.size - 1
    if N < 2:
        raise ValueError("Need at least 2 cells.")

    geom = case.geom if case.geom is not None else planar()
    gamma = case.gamma
    rho0, p0, u0 = case.rho0, case.p0, case.u0

    # Cell-centered fields (uniform)
    rho = np.full(N, float(rho0))
    p = np.full(N, float(p0))
    e = internal_energy_from_prho(p, rho, gamma)
    q = np.zeros_like(rho)

    # Node-centered velocity (uniform)
    u_nodes = np.full(N + 1, float(u0))

    # Geometric volumes + Lagrangian masses (fixed)
    V_cells = cell_volumes(x_nodes, geom)
    m_cells = masses_from_initial_rho(x_nodes, rho, geom)

    # Initial acceleration = 0
    a_nodes = np.zeros_like(x_nodes)

    state = HydroState(
        t=0.0,
        x=x_nodes,
        u=u_nodes,
        a=a_nodes,
        V=V_cells,
        rho=rho,
        e=e,
        p=p,
        q=q,
        m_cells=m_cells
    )
    return state


# Legacy alias
init_planar_driven_shock_case = lambda x_nodes, geom, gamma, case: init_driven_shock(x_nodes, case)

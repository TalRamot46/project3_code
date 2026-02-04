# problems/driven_shock_problem.py
from dataclasses import dataclass
import numpy as np

from problems.base_problem import ProblemCase
from core.state import HydroState
from core.eos import internal_energy_from_prho
from core.grid import cell_volumes, masses_from_initial_rho


@dataclass(frozen=True)
class DrivenShockCase(ProblemCase):
    """
    Configuration for driven shock problems.
    
    Models a shock driven by a time-dependent boundary condition
    (pressure or velocity) at the left boundary.
    
    Attributes:
        rho0: Initial uniform density
        p0: Initial uniform pressure
        u0: Initial uniform velocity
        gamma: Adiabatic index
    """
    rho0: float = 1.0
    p0: float = 1.0
    u0: float = 0.0
    gamma: float = 5.0/3.0
    tau: float = 0.0
    P0: float = 1.0

    # Boundary driving
    def p_left(self, t):
        return self.P0 * t**self.tau

def init_planar_driven_shock_case(x_nodes, geom, gamma, case):
    x_nodes = np.asarray(x_nodes, dtype=float)
    N = x_nodes.size - 1
    if N < 2:
        raise ValueError("Need at least 2 cells.")

    rho0, p0, u0 = case.rho0, case.p0, case.u0

    # Cell-centered fields (uniform)
    rho = np.full(N, float(rho0))
    p   = np.full(N, float(p0))
    e   = internal_energy_from_prho(p, rho, gamma)
    q   = np.zeros_like(rho)

    # Node-centered velocity (uniform)
    u_nodes = np.full(N + 1, float(u0))

    # Geometric volumes + Lagrangian masses (fixed)
    V_cells = cell_volumes(x_nodes, geom)
    m_cells = masses_from_initial_rho(x_nodes, rho, geom)

    # Initial acceleration = 0 (will be computed after applying BCs and pressure gradients)
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
        q=q
    )
    return state, m_cells

# problems/driven_shock_problem.py
from dataclasses import dataclass
import numpy as np

from core.state import HydroState
from core.eos import internal_energy_from_prho
from core.grid import cell_volumes, masses_from_initial_rho
from core.state import HydroState

@dataclass
class DrivenShockCase:
    name: str
    x_min: float
    x_max: float
    t_end: float
    rho0: float
    p0: float
    u0: float
    gamma: float

    # Boundary driving
    def p_left(self, t):
        raise NotImplementedError

    def u_left(self, t):
        return 0.0  # default: pressure-driven piston

class PowerLawPressureDrive(DrivenShockCase):
    def __init__(self, *, tau, P0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.P0 = P0

    def p_left(self, t):
        return self.P0 * t**self.tau

# problems/driven_shock_init.py
import numpy as np

import numpy as np

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

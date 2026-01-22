# problems/riemann_problem.py
import numpy as np
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eos import internal_energy_from_prho
from grid import cell_volumes, masses_from_initial_rho
from state import HydroState

@dataclass(frozen=True)
class RiemannCase:
    test_id: int
    left: tuple
    right: tuple
    t_end: float

RIEMANN_CASES = {
    1: RiemannCase(1, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), t_end=0.25),
    2: RiemannCase(2, left=(1.0, 0.0, 1000.0), right=(1.0, 0.0, 0.01), t_end=0.012),
    3: RiemannCase(3, left=(1.0, 0.0, 0.01), right=(1.0, 0.0, 100.0), t_end=0.035),
    4: RiemannCase(4, left=(5.99924, 19.5975, 460.894), right=(5.99242, -6.19633, 46.0950), t_end=0.035),
}

def init_planar_riemann_case(x_nodes, geom, gamma, case: RiemannCase, x0=0.0):
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])

    rhoL, uL, pL = case.left
    rhoR, uR, pR = case.right

    left_mask = x_cells < x0

    rho = np.where(left_mask, rhoL, rhoR)
    p   = np.where(left_mask, pL, pR)
    e   = internal_energy_from_prho(p, rho, gamma)
    q   = np.zeros_like(rho)

    u_nodes = np.where(x_nodes < x0, uL, uR)

    V = cell_volumes(x_nodes, geom)
    m = masses_from_initial_rho(x_nodes, rho, geom)

    a_nodes = np.zeros_like(x_nodes)

    state = HydroState(t=0.0, x=x_nodes, u=u_nodes, a=a_nodes, V=V, rho=rho, e=e, p=p, q=q)
    return state, m

# problems/riemann_problem.py
import numpy as np
from dataclasses import dataclass

from problems.base_problem import ProblemCase
from core.eos import internal_energy_from_prho
from core.grid import cell_volumes, masses_from_initial_rho
from core.state import HydroState


@dataclass(frozen=True)
class RiemannCase(ProblemCase):
    """
    Configuration for Riemann (shock tube) problems.
    
    Defines left and right states separated by a discontinuity at x0.
    
    Attributes:
        test_id: Identifier for the test case
        left: Left state tuple (rho, u, p)
        right: Right state tuple (rho, u, p)
        x0: Initial discontinuity position
        sigma_visc: Artificial viscosity coefficient
    """
    test_id: int = 0
    left: tuple = (1.0, 0.0, 1.0)      # (rho,u,p)
    right: tuple = (0.125, 0.0, 0.1)   # (rho,u,p)
    x_min: float = -1.0
    x_max: float = 1.0
    t_end: float = 0.25
    x0: float = 0.0
    sigma_visc: float = 1.0
    title: str = ""

RIEMANN_TEST_CASES = {
    1: RiemannCase(
        test_id=1, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1),
        x_min=-1.0, x_max=1.0, t_end=0.25, sigma_visc=1.0, title="Sod-like"
    ),
    2: RiemannCase(
        test_id=2, left=(1.0, 0.0, 1000.0), right=(1.0, 0.0, 0.01),
        x_min=-1.0, x_max=1.0, t_end=0.012, sigma_visc=1.0, title="Strong pressure jump"
    ),
    3: RiemannCase(
        test_id=3, left=(1.0, 0.0, 0.01), right=(1.0, 0.0, 100.0),
        x_min=-1.0, x_max=1.0, t_end=0.035, sigma_visc=1.0, title="Reverse pressure jump"
    ),
    4: RiemannCase(
        test_id=4, left=(5.99924, 19.5975, 460.894), right=(5.99242, -6.19633, 46.0950),
        x_min=-1.0, x_max=1.0, t_end=0.035, sigma_visc=3.0, title="Colliding streams"
    ),
}

def init_planar_riemann_case(x_nodes, geom, gamma, case: RiemannCase, x0=0.0):
    """
    Build initial HydroState + fixed cell masses for a planar Riemann problem.
    """
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])

    rhoL, uL, pL = case.left
    rhoR, uR, pR = case.right

    left_mask = x_cells < x0

    rho = np.where(left_mask, rhoL, rhoR)
    p   = np.where(left_mask, pL, pR)
    e   = internal_energy_from_prho(p, rho, gamma)
    q   = np.zeros_like(rho)

    # Node velocities: simplest consistent choice is piecewise constant using node position
    u_nodes = np.where(x_nodes < x0, uL, uR)

    V = cell_volumes(x_nodes, geom)
    m = masses_from_initial_rho(x_nodes, rho, geom)

    a_nodes = np.zeros_like(x_nodes)

    state = HydroState(t=0.0, x=x_nodes, u=u_nodes, a=a_nodes, V=V, rho=rho, e=e, p=p, q=q)
    return state, m

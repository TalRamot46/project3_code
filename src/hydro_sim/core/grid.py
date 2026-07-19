from dataclasses import dataclass
import numpy as np
from .geometry import Geometry

@dataclass
class Grid1D:
    x_nodes: np.ndarray   # nodes positions (Nnodes,)
    m_cells: np.ndarray   # cell masses (Ncells,)
    geom: Geometry

def cell_volumes(x_nodes: np.ndarray, geom: Geometry) -> np.ndarray:
    # PDF Eq.(13): V = zeta * (r_{i+1}^{alpha+1} - r_i^{alpha+1})
    a = geom.alpha
    return geom.zeta * (x_nodes[1:]**(a+1) - x_nodes[:-1]**(a+1))

def make_uniform_nodes(x_min: float, x_max: float, Ncells: int) -> np.ndarray:
    # Ncells = 1000 => Nnodes=1001
    return np.linspace(x_min, x_max, Ncells + 1)

def make_nonuniform_nodes(x_min: float, x_max: float, Ncells: int) -> np.ndarray:
    """Build a gradually refined spatial node grid near x_min for non-uniform media (omega != 0), matching lines 261-267 of ablation_solver_og.py."""
    L = x_max - x_min
    num_cells = Ncells
    nodes = np.array(list(sorted(set(
        list(np.linspace(x_min, x_min + L / 1000.0, num_cells * 6)) +
        list(np.linspace(x_min + L / 1000.0, x_min + L / 20.0, num_cells * 6)) +
        list(np.linspace(x_min + L / 20.0, x_min + L / 3.0, num_cells)) +
        list(np.linspace(x_min + L / 3.0, x_max, num_cells + 1))
    ))))
    return nodes

def make_nodes(x_min: float, x_max: float, Ncells: int, omega: float = 0.0) -> np.ndarray:
    """Generate 1D spatial node coordinates. Uses a gradually refined grid if omega != 0."""
    if omega != 0.0:
        return make_nonuniform_nodes(x_min, x_max, Ncells)
    return make_uniform_nodes(x_min, x_max, Ncells)

def masses_from_initial_rho(x_nodes: np.ndarray, rho_cells0: np.ndarray, geom: Geometry) -> np.ndarray:
    V0 = cell_volumes(x_nodes, geom)
    return rho_cells0 * V0

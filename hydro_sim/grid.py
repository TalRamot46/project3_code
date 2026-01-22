from dataclasses import dataclass
import numpy as np
from geometry import Geometry

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

def masses_from_initial_rho(x_nodes: np.ndarray, rho_cells0: np.ndarray, geom: Geometry) -> np.ndarray:
    V0 = cell_volumes(x_nodes, geom)
    return rho_cells0 * V0

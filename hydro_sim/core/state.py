import numpy as np
from dataclasses import dataclass

@dataclass
class HydroState:
    t: float
    x: np.ndarray      # nodes (Nnodes,)
    u: np.ndarray      # nodes
    a: np.ndarray      # nodes
    V: np.ndarray      # cells (Ncells,)
    rho: np.ndarray    # cells
    e: np.ndarray      # cells
    p: np.ndarray      # cells
    q: np.ndarray      # cells
    m_cells: np.ndarray # cells

@dataclass
class RadHydroState(HydroState):
    T: np.ndarray      # cells
    E_rad: np.ndarray  # cells

    def _replace(self, **changes):
        """Utility method to create a new instance with some fields updated."""
        return self.__class__(**{**self.__dict__, **changes})
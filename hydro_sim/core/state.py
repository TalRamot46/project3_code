import numpy as np
from dataclasses import dataclass

@dataclass
class HydroState:
    t: float
    x: np.ndarray      # nodes (Nnodes,)
    u: np.ndarray      # nodes
    a: np.ndarray       # nodes
    V: np.ndarray           # cells (Ncells,)
    rho: np.ndarray         # cells
    e_material: np.ndarray      # cells  (specific internal energy, erg/g)
    p: np.ndarray      # cells
    q: np.ndarray      # cells
    m_cells: np.ndarray # cells
    T_material: np.ndarray  # cells  (material temperature from EOS, K)

@dataclass
class RadHydroState(HydroState):
    T_rad: np.ndarray      # cells  (radiation temperature, K)
    E_rad: np.ndarray  # cells  (radiation energy density, erg/cm^3)

    def _replace(self, **changes):
        """Utility method to create a new instance with some fields updated."""
        return self.__class__(**{**self.__dict__, **changes})

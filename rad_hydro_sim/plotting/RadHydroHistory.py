# inherit from HydroHistory, but add radiation energy density and temperature to the history
from dataclasses import dataclass

import numpy as np

from project_3.hydro_sim.simulations.lagrangian_sim import HydroHistory

@dataclass
class RadHydroHistory(HydroHistory):
    """
    Time history of radiation-hydrodynamics simulation fields.
    
    Inherits from HydroHistory and adds:
        T: Temperature at each snapshot (K, Ncells)
        E_rad: Radiation energy density at each snapshot (K, Ncells)
    """
    T: np.ndarray      # (K, Ncells) temperature
    E_rad: np.ndarray  # (K, Ncells) radiation energy density
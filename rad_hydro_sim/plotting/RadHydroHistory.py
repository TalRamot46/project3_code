# inherit from HydroHistory, but add radiation energy density and temperature to the history
from dataclasses import dataclass

import numpy as np

from project3_code.hydro_sim.simulations.lagrangian_sim import HydroHistory

@dataclass
class RadHydroHistory(HydroHistory):
    """
    Time history of radiation-hydrodynamics simulation fields.
    
    Inherits from HydroHistory and adds:
        T: Radiation temperature at each snapshot (K, Ncells) [Kelvin]
        E_rad: Radiation energy density at each snapshot (K, Ncells) [erg/cm^3]
        T_material: Material temperature at each snapshot (K, Ncells) [Kelvin]
    """
    T: np.ndarray = None      # (K, Ncells) radiation temperature
    E_rad: np.ndarray = None  # (K, Ncells) radiation energy density
    T_material: np.ndarray = None  # (K, Ncells) material temperature (from Rosen EOS)

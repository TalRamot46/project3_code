# verification/hydro_data.py
"""
Convert rad_hydro and hydro_sim histories to data containers for comparison plots.

For hydro-only verification we reuse ``SimulationData`` from ``hydro_sim``.
For full rad_hydro verification we use a dedicated ``RadHydroData`` container
with the same fields but a distinct type.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# Parent of repo root so project_3 package resolves when run as script
_repo_parent = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_parent) not in sys.path:
    sys.path.insert(0, str(_repo_parent))




@dataclass
class RadHydroData:
    """
    Hydrodynamic fields from a rad_hydro run, for verification comparisons.

    Shape/semantics mirror ``SimulationData`` so plotting helpers from
    ``hydro_sim.verification.compare_shock_plots`` can be reused.
    """

    times: np.ndarray               # (nt,)
    m: List[np.ndarray]             # list of (N,) arrays - mass coordinate
    x: List[np.ndarray]             # list of (N,) arrays - position
    rho: List[np.ndarray]           # list of (N,) arrays - density
    p: List[np.ndarray]             # list of (N,) arrays - pressure
    u: List[np.ndarray]             # list of (N,) arrays - velocity
    e: List[np.ndarray]             # list of (N,) arrays - specific internal energy
    T: List[np.ndarray] = field(default_factory=list)      # list of (N,) arrays - temperature [Hev]
    E_rad: Optional[List[np.ndarray]] = field(default_factory=list)  # list of (N,) arrays - radiation energy [erg/cm^3]
    label: str = "Rad-Hydro"
    color: str = "blue"
    linestyle: str = "-"



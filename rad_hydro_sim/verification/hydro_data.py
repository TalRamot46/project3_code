# verification/hydro_data.py
"""
Convert rad_hydro and hydro_sim histories to data containers for comparison plots.

For hydro-only verification we reuse ``SimulationData`` from ``hydro_sim``.
For full rad_hydro verification we use a dedicated ``RadHydroData`` container
with the same fields but a distinct type.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Parent of repo root so project_3 package resolves when run as script
_repo_parent = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_parent) not in sys.path:
    sys.path.insert(0, str(_repo_parent))

from project_3.hydro_sim.verification.compare_shock_plots import (
    SimulationData,
    load_hydro_history as _load_hydro_history,
)


@dataclass
class RadHydroData:
    """
    Hydrodynamic fields from a rad_hydro run, for verification comparisons.

    Shape/semantics mirror ``SimulationData`` so plotting helpers from
    ``hydro_sim.verification.compare_shock_plots`` can be reused.
    """

    times: np.ndarray          # (nt,)
    m: List[np.ndarray]        # list of (N,) arrays - mass coordinate
    x: List[np.ndarray]        # list of (N,) arrays - position
    rho: List[np.ndarray]      # list of (N,) arrays - density
    p: List[np.ndarray]        # list of (N,) arrays - pressure
    u: List[np.ndarray]        # list of (N,) arrays - velocity
    e: List[np.ndarray]        # list of (N,) arrays - specific internal energy
    label: str = "Rad-Hydro"
    color: str = "blue"
    linestyle: str = "-"


def load_hydro_history(history):
    """Convert hydro_sim HydroHistory to SimulationData. Re-export from shock comparison."""
    return _load_hydro_history(history)


def load_rad_hydro_history(history, label: str = "Rad-Hydro") -> RadHydroData:
    """
    Convert RadHydroHistory (hydro fields only) to RadHydroData for comparisons.
    """
    times = np.asarray(history.t, dtype=float)
    nt = len(times)
    m_list: List[np.ndarray] = []
    x_list: List[np.ndarray] = []
    rho_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    e_list: List[np.ndarray] = []
    for k in range(nt):
        # RadHydroHistory has .x, .m, .rho, .p, .u, .e as (K, Ncells) arrays
        x_k = history.x[k]
        m_k = history.m[k]
        rho_k = history.rho[k]
        p_k = history.p[k]
        u_k = history.u[k]
        e_k = history.e[k]
        x_list.append(x_k.copy() if hasattr(x_k, "copy") else np.asarray(x_k))
        m_list.append(m_k.copy() if hasattr(m_k, "copy") else np.asarray(m_k))
        rho_list.append(rho_k.copy() if hasattr(rho_k, "copy") else np.asarray(rho_k))
        p_list.append(p_k.copy() if hasattr(p_k, "copy") else np.asarray(p_k))
        u_list.append(u_k.copy() if hasattr(u_k, "copy") else np.asarray(u_k))
        e_list.append(e_k.copy() if hasattr(e_k, "copy") else np.asarray(e_k))
    return RadHydroData(
        times=times,
        m=m_list,
        x=x_list,
        rho=rho_list,
        p=p_list,
        u=u_list,
        e=e_list,
        label=label,
        color="blue",
        linestyle="-",
    )

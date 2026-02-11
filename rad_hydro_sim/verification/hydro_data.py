# verification/hydro_data.py
"""
Convert rad_hydro and hydro_sim histories to SimulationData for comparison plots.

Reuses SimulationData and load_hydro_history from verification_hydro_sim_with_shock
for hydro-only comparison (rad_hydro vs run_hydro).
"""
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

# Parent of repo root so project_3 package resolves when run as script
_repo_parent = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_parent) not in sys.path:
    sys.path.insert(0, str(_repo_parent))

from project_3.hydro_sim.verification.compare_shock_plots import (
    SimulationData,
    load_hydro_history as _load_hydro_history,
)


def load_hydro_history(history):
    """Convert hydro_sim HydroHistory to SimulationData. Re-export from shock comparison."""
    return _load_hydro_history(history)


def load_rad_hydro_history(history, label: str = "Rad-Hydro") -> SimulationData:
    """Convert RadHydroHistory (hydro fields only) to SimulationData for comparison with run_hydro."""
    times = np.asarray(history.t, dtype=float)
    nt = len(times)
    m_list = []
    x_list = []
    rho_list = []
    p_list = []
    u_list = []
    e_list = []
    for k in range(nt):
        # RadHydroHistory has .x, .m, .rho, .p, .u, .e as (K, Ncells) arrays
        x_list.append(history.x[k].copy() if hasattr(history.x[k], "copy") else np.asarray(history.x[k]))
        m_list.append(history.m[k].copy() if hasattr(history.m[k], "copy") else np.asarray(history.m[k]))
        rho_list.append(history.rho[k].copy() if hasattr(history.rho[k], "copy") else np.asarray(history.rho[k]))
        p_list.append(history.p[k].copy() if hasattr(history.p[k], "copy") else np.asarray(history.p[k]))
        u_list.append(history.u[k].copy() if hasattr(history.u[k], "copy") else np.asarray(history.u[k]))
        e_list.append(history.e[k].copy() if hasattr(history.e[k], "copy") else np.asarray(history.e[k]))
    return SimulationData(
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

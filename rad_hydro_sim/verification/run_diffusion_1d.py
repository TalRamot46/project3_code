# verification/run_diffusion_1d.py
"""
Run the 1D Diffusion self-similar model with parameters matching a RadHydroCase.

Used for radiation-only verification: compare run_rad_hydro (radiation_only_constant_temperature_drive)
to this reference solution.

Returns (times_sec, x_grid, T_list, E_rad_list) in consistent units (s, cm, Hev, erg/cm^3).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np

# Repo root: project_3/rad_hydro_sim/verification -> repo root is 3 levels up
_VERIFICATION_DIR = Path(__file__).resolve().parent
_RAD_HYDRO_DIR = _VERIFICATION_DIR.parent
_PROJECT_3 = _RAD_HYDRO_DIR.parent
_REPO_ROOT = _PROJECT_3.parent

# 1D Diffusion script path (from repo root)
_DIFFUSION_SCRIPT_PATH = Path(__file__).resolve().parent / "1D Diffusion self similar.py"

def run_diffusion_1d(
    x_max: float,
    t_end: float,
    T_bath_hev: float,
    rho0: float,
    n_times: int = 30,
    Nz: int = 500,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Run the 1D Diffusion self-similar code with given physical parameters.

    Parameters
    ----------
    x_max : float
        Domain length [cm].
    t_end : float
        Final time [s].
    T_bath_hev : float
        Boundary temperature [Hev].
    rho0 : float
        Initial density [g/cm^3].
    n_times : int
        Number of time snapshots to store.
    Nz : int
        Number of grid points for the diffusion solver.

    Returns
    -------
    times_sec : np.ndarray
        Stored times [s].
    x_grid : np.ndarray
        Spatial grid (cell centers or nodes as in the script) [cm].
    T_list : list of np.ndarray
        Temperature at each stored time [Hev].
    E_rad_list : list of np.ndarray
        Radiation energy density a*T^4 at each stored time [erg/cm^3].
    """
    if not _DIFFUSION_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"1D Diffusion script not found: {_DIFFUSION_SCRIPT_PATH}\n"
            "Ensure '1D Diffusion self similar in gold/figures/1D Diffusion self similar.py' exists."
        )

    # Load module from file
    spec = importlib.util.spec_from_file_location("diffusion_1d", _DIFFUSION_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {_DIFFUSION_SCRIPT_PATH}")
    diffusion = importlib.util.module_from_spec(spec)

    # Load module (script may write to "1D Diffusion self similar in gold/data/..." relative to cwd)
    sys.path.insert(0, str(_REPO_ROOT))
    try:
        spec.loader.exec_module(diffusion)
    finally:
        if str(_REPO_ROOT) in sys.path:
            sys.path.remove(str(_REPO_ROOT))

    # Patch parameters to match rad_hydro case (constant temperature drive)
    # Script uses: L, t_final, T_bath (or T_bath_hev), rho, Nz, z, dz, dt, etc.
    diffusion.L = float(x_max)
    diffusion.t_final = float(t_end)
    diffusion.Nz = int(Nz)
    diffusion.z = np.linspace(0.0, diffusion.L, diffusion.Nz)
    diffusion.dz = diffusion.z[1] - diffusion.z[0]
    diffusion.rho = float(rho0)

    # Boundary and initial T (script uses T_bath in Kelvin when CGS, T_bath_hev when HEV_NS)
    # Keep script in CGS; it stores T in Hev (T/K_per_Hev). So set T_bath so that T_bath_hev is correct.
    if hasattr(diffusion, "K_per_Hev"):
        diffusion.T_bath_kelvin = T_bath_hev * diffusion.K_per_Hev
        diffusion.T_bath = diffusion.T_bath_kelvin
        diffusion.T_bath_hev = float(T_bath_hev)
    else:
        diffusion.T_bath_hev = float(T_bath_hev)

    # Times to store (in seconds; script run_time_loop uses t in seconds internally)
    times_to_store = np.linspace(t_end * 0.05, t_end * 0.95, n_times)

    # Run the case (tau=0 for constant temperature drive). Script writes CSV under repo root.
    old_cwd = Path.cwd()
    try:
        os.chdir(_REPO_ROOT)
        result = diffusion.run_case(
            tau=0.0,
            times_to_store=times_to_store,
            reset_initial_conditions=True,
        )
    finally:
        os.chdir(old_cwd)

    stored_t = np.asarray(result["stored_t"], dtype=float)
    stored_T = result["stored_T"]  # list or (n_times, Nz); in Hev when CGS
    z = np.asarray(diffusion.z, dtype=float)

    # Script returns stored_t in ns when simulation_unit_system == CGS
    if getattr(diffusion, "simulation_unit_system", None) == "cgs":
        times_sec = stored_t * 1e-9
    else:
        times_sec = stored_t.copy()

    # Ensure T is list of 1d arrays
    if hasattr(stored_T, "shape") and stored_T.ndim == 2:
        T_list = [stored_T[i, :].copy() for i in range(stored_T.shape[0])]
    else:
        T_list = [np.asarray(t, dtype=float).flatten() for t in stored_T]

    # Radiation constant in Hev units (script has a_hev)
    a_hev = getattr(diffusion, "a_hev", None)
    if a_hev is None:
        from project_3.rad_hydro_sim.simulation.radiation_step import a_Hev
        a_hev = a_Hev
    E_rad_list = [a_hev * T**4 for T in T_list]

    return times_sec, z, T_list, E_rad_list

# verification/radiation_data.py
"""
Data containers for radiation-only verification (T, E_rad vs x).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any

import numpy as np


@dataclass
class RadiationData:
    """Temperature and radiation energy density at multiple times."""
    times: np.ndarray           # (n_times,) [s]
    x: List[np.ndarray]        # length n_times, each (N,) [cm]
    T: List[np.ndarray]         # length n_times, each (N,) [Hev]
    E_rad: List[np.ndarray]    # length n_times, each (N,) [erg/cm^3]
    label: str = ""
    color: str = "blue"
    linestyle: str = "-"


def rad_hydro_history_to_radiation_data(history) -> RadiationData:
    """Convert RadHydroHistory to RadiationData (same grid at each time)."""
    times = np.asarray(history.t, dtype=float)
    n = len(times)
    x_list = [history.x[k].copy() for k in range(n)]
    T_list = [history.T[k].copy() for k in range(n)]
    E_rad_list = [history.E_rad[k].copy() for k in range(n)]
    return RadiationData(
        times=times,
        x=x_list,
        T=T_list,
        E_rad=E_rad_list,
        label="Rad-Hydro",
        color="blue",
        linestyle="-",
    )


def diffusion_output_to_radiation_data(
    times_sec: np.ndarray,
    x_grid: np.ndarray,
    T_list: List[np.ndarray],
    E_rad_list: List[np.ndarray],
) -> RadiationData:
    """Wrap 1D Diffusion output into RadiationData (fixed grid, same x for all times)."""
    # Diffusion uses fixed grid; repeat for each time for consistent list-of-arrays format
    x_list = [x_grid.copy() for _ in range(len(times_sec))]
    return RadiationData(
        times=np.asarray(times_sec, dtype=float),
        x=x_list,
        T=T_list,
        E_rad=E_rad_list,
        label="1D Diffusion (reference)",
        color="red",
        linestyle="--",
    )


def supersonic_output_to_radiation_data(
    profiles_dict: dict[str, Any],
    times_sec: np.ndarray,
    label: str = "Supersonic solver (reference)",
    color: str = "green",
    linestyle: str = ":",
) -> RadiationData:
    """
    Convert supersonic solver output (compute_profiles_for_report) to RadiationData.

    The solver uses dimensionless time in (0, 1]; we map to physical time as
    times_sec = profiles_dict["times"] * t_end_sec.
    T_heat from the solver is 100*T0*times^tau*T_tilde; we convert to Kelvin as T_Kelvin = T_heat/100
    when T0 was passed in Hev. E_rad = a_Kelvin * T_Kelvin^4 [erg/cm^3].
    """
    from project_3.rad_hydro_sim.simulation.radiation_step import a_Kelvin, KELVIN_PER_HEV

    # T_heat from solver is in HeV; convert to Kelvin: T_K = T_HeV * KELVIN_PER_HEV
    T_heat_Kelvin = profiles_dict["T_heat"] * KELVIN_PER_HEV  # (n_times, n_xi)
    T_list = [T_heat_Kelvin[i, :].copy() for i in range(T_heat_Kelvin.shape[0])]
    E_rad_list = [a_Kelvin*T_heat_Kelvin[i,:].copy()**4 for i in range(T_heat_Kelvin.shape[0])]
    x_list = [profiles_dict["x_heat"][i, :].copy() for i in range(T_heat_Kelvin.shape[0])]
    return RadiationData(
        times=np.asarray(times_sec, dtype=float),
        x=x_list,
        T=T_list,
        E_rad=E_rad_list,
        label=label,
        color=color,
        linestyle=linestyle,
    )

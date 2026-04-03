# compare_shock_plots.py
"""
Comparison plotting for driven shock simulations.
Compares results from hydro_sim (Lagrangian simulation) with
shussman_shock_solver (self-similar solution).
"""
from __future__ import annotations

import numpy as np
from project3_code.rad_hydro_sim.plotting import mpl_style  # noqa: F401 - apply project style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

from project3_code.rad_hydro_sim.verification.hydro_data import RadHydroData
from project3_code.rad_hydro_sim.plotting import RadHydroHistory
@dataclass
class HydroSimData:
    """Container for simulation data at multiple times."""
    times: np.ndarray      # (nt,)
    m: list                # list of (N,) arrays - mass coordinate
    x: list                # list of (N,) arrays - position  
    rho: list              # list of (N,) arrays - density
    p: list                # list of (N,) arrays - pressure
    u: list                # list of (N,) arrays - velocity
    e: list                # list of (N,) arrays - specific internal energy
    label: str = "Simulation"
    color: str = "blue"
    linestyle: str = "-"


# Type alias: anything with SimulationData layout (RadHydroData is duck-typed compatible)
HydroDataLike = Union[HydroSimData, RadHydroData]

def load_rad_hydro_history(history: RadHydroHistory, label: str) -> RadHydroData:
    """Convert hydro_sim HydroHistory to SimulationData. Re-export from shock comparison."""
    times = np.asarray(history.t, dtype=float)
    nt = len(times)
    m_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    rho_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    u_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    T_list: list[np.ndarray] = []
    E_rad_list: list[np.ndarray] = []
    T_material_list: list[np.ndarray] = []
    has_T_material = hasattr(history, "T_material") and history.T_material is not None
    for k in range(nt):
        x_k = history.x[k]
        m_k = history.m[k]
        rho_k = history.rho[k]
        p_k = history.p[k]
        u_k = history.u[k]
        e_k = history.e[k]
        T_k = history.T[k]
        E_k = history.E_rad[k]
        x_list.append(x_k.copy() if hasattr(x_k, "copy") else np.asarray(x_k))
        m_list.append(m_k.copy() if hasattr(m_k, "copy") else np.asarray(m_k))
        rho_list.append(rho_k.copy() if hasattr(rho_k, "copy") else np.asarray(rho_k))
        p_list.append(p_k.copy() if hasattr(p_k, "copy") else np.asarray(p_k))
        u_list.append(u_k.copy() if hasattr(u_k, "copy") else np.asarray(u_k))
        e_list.append(e_k.copy() if hasattr(e_k, "copy") else np.asarray(e_k))
        if T_k is not None:
            t_arr = T_k
            T_list.append(t_arr.copy() if hasattr(t_arr, "copy") else np.asarray(t_arr))
        if E_k is not None:
            e_arr = E_k
            E_rad_list.append(e_arr.copy() if hasattr(e_arr, "copy") else np.asarray(e_arr))
        if has_T_material:
            Tm_k = history.T_material[k]
            T_material_list.append(Tm_k.copy() if hasattr(Tm_k, "copy") else np.asarray(Tm_k))
    return RadHydroData(
        times=times,
        m=m_list,
        x=x_list,
        rho=rho_list,
        p=p_list,
        u=u_list,
        e=e_list,
        T=T_list,
        E_rad=E_rad_list,
        T_material=T_material_list,
        label=label,
        color="blue",
        linestyle="-",
    )

def _as_list(arr):
    """Convert array to list of 1D arrays (for NPZ compatibility)."""
    if isinstance(arr, list):
        return arr
    arr = np.asarray(arr)
    if arr.dtype == object:
        return [np.asarray(v, float) for v in arr.tolist()]
    if arr.ndim == 2:
        return [arr[i, :].astype(float, copy=False) for i in range(arr.shape[0])]
    if arr.ndim == 1:
        return [arr.astype(float, copy=False)]
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def load_shussman_data(npz_path: str | Path) -> HydroSimData:
    """Load data from shussman_shock_solver NPZ file.
    Supports both formats: 'times_sec' (seconds) or 'times' (nanoseconds).
    """
    data = np.load(npz_path, allow_pickle=True)
    files = list(data.files)

    if "times_sec" in files:
        times_sec = np.asarray(data["times_sec"], dtype=float)
    elif "times" in files:
        times_ns = np.asarray(data["times"], dtype=float)
        times_sec = times_ns * 1e-9
    else:
        raise KeyError(
            f"NPZ archive has no 'times_sec' or 'times'. Keys present: {files}"
        )
    
    # Load profiles
    m = _as_list(data["m_shock"])
    x = _as_list(data.get("x_shock", data.get("m_shock")))  # fallback to m if x not available
    rho = _as_list(data["rho_shock"])
    p = _as_list(data["P_shock"])
    u = _as_list(data["u_shock"])
    
    # Compute energy from EOS if not stored
    if "e_shock" in data.files:
        e = _as_list(data["e_shock"])
    else:
        # Estimate e from P and rho using ideal gas: e = P / (rho * (gamma - 1))
        # This is approximate - real material EOS may differ
        gamma = data.get("gamma", 5/3)
        e = [p_i / (rho_i * (gamma - 1) + 1e-30) for p_i, rho_i in zip(p, rho)]
    
    return HydroSimData(
        times=times_sec,
        m=m, x=x, rho=rho, p=p, u=u, e=e,
        label="Self-Similar (Shussman)",
        color="red",
        linestyle="--"
    )


def load_hydro_history(history) -> HydroSimData:
    """Convert hydro_sim SimulationHistory to SimulationData format."""
    times = history.t
    nt = len(times)
    
    # Extract arrays and compute mass coordinate m = integral(rho * dx)
    m_list = []
    x_list = []
    rho_list = []
    p_list = []
    u_list = []
    e_list = []
    
    for k in range(nt):
        x = history.x[k]
        rho = history.rho[k]
        m = history.m[k]
        
        m_list.append(m)
        x_list.append(x)
        rho_list.append(rho)
        p_list.append(history.p[k])
        u_list.append(history.u[k])
        e_list.append(history.e[k])
    
    return HydroSimData(
        times=times,
        m=m_list, x=x_list, rho=rho_list, p=p_list, u=u_list, e=e_list,
        label="Hydro Simulation",
        color="blue",
        linestyle="-"
    )


def interpolate_to_time(data: HydroDataLike, target_time: float) -> int:
    """Find the index of the closest time in data."""
    idx = np.argmin(np.abs(data.times - target_time))
    return idx


# Stored hydro fields: p [Barye], u [cm/s], e [erg/g] → display MBar, km/s, hJ/g
PLOT_P_SCALE = 1e12
PLOT_U_SCALE = 1e5
PLOT_E_SCALE = 1e9


def _legend_time_s(t_s: float) -> str:
    if abs(t_s) >= 1e-7:
        return f"{t_s * 1e9:.4g} ns"
    return f"{t_s:.3e} s"


# ============================================================================
# 4-Panel comparison: all selected times on one figure (discrete legend)
# ============================================================================

def plot_comparison_in_selected_times(
    sim_data: HydroDataLike,
    ref_data: HydroDataLike,
    times: np.ndarray,
    xaxis: str = "m",
    savepath: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
    shock_data: HydroDataLike | None = None,
    cmap_name: str = "plasma",
) -> list[tuple[Any, Any]]:
    """
    One figure: overlay every requested snapshot (nearest stored time in each dataset).

    Discrete colors per requested time; legend lists each curve with its time.
    Linestyle: simulation solid, reference dashed, optional third dotted.

    Parameters:
        sim_data: Simulation data (hydro_sim / rad_hydro)
        ref_data: Reference data (e.g. Shussman or run_hydro)
        times: 1D ``np.ndarray`` of times in seconds
        xaxis: ``"m"`` for mass coordinate, ``"x"`` for position
        savepath: If set, write a single PNG
        show: If True, display; otherwise close after saving
        title: Figure title (optional)
        shock_data: Optional third dataset (e.g. shock solver P0*t^tau)
        cmap_name: Colormap used to pick distinct colors per snapshot index

    Returns:
        ``[(figure, axes)]`` (length 1) for compatibility with callers expecting a list.
    """
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy.ndarray")
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        raise ValueError("times must be non-empty")

    cmap = plt.get_cmap(cmap_name)
    n_t = int(times.size)
    colors = cmap(np.linspace(0.2, 0.92, max(n_t, 1)))

    if xaxis == "m":
        xlabel = r"Mass coordinate $m$ [g/cm²]"
    else:
        xlabel = r"Position $x$ [cm]"

    fig, axes = plt.subplots(4, 2, figsize=(12.5, 11), sharex=True)
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_e = axes[1, 0], axes[1, 1]
    ax_Tm, ax_T = axes[2, 0], axes[2, 1]
    ax_E_rad = axes[3, 0]
    axes[3, 1].set_visible(False)

    _has_Tm = hasattr(sim_data, "T_material") and sim_data.T_material
    _has_T = hasattr(sim_data, "T") and sim_data.T and hasattr(ref_data, "T") and ref_data.T
    _has_E_rad = (
        hasattr(sim_data, "E_rad")
        and sim_data.E_rad
        and hasattr(ref_data, "E_rad")
        and ref_data.E_rad
    )

    lw_sim, lw_ref = 2.0, 2.0
    lw_shock = 1.65
    legend_handles: list[Line2D] = []

    for j, t_req in enumerate(times):
        t_req = float(t_req)
        color = colors[j % len(colors)]
        sim_idx = interpolate_to_time(sim_data, t_req)
        ref_idx = interpolate_to_time(ref_data, t_req)
        shock_idx = interpolate_to_time(shock_data, t_req) if shock_data is not None else None
        t_lbl = _legend_time_s(t_req)

        if xaxis == "m":
            x_sim = sim_data.m[sim_idx]
            x_ref = ref_data.m[ref_idx]
            x_shock = (
                shock_data.m[shock_idx]
                if shock_data is not None and shock_idx is not None
                else None
            )
        else:
            x_sim = sim_data.x[sim_idx]
            x_ref = ref_data.x[ref_idx]
            x_shock = (
                shock_data.x[shock_idx]
                if shock_data is not None and shock_idx is not None
                else None
            )

        ps = sim_data.p[sim_idx] / PLOT_P_SCALE
        pr = ref_data.p[ref_idx] / PLOT_P_SCALE
        us = sim_data.u[sim_idx] / PLOT_U_SCALE
        ur = ref_data.u[ref_idx] / PLOT_U_SCALE
        es = sim_data.e[sim_idx] / PLOT_E_SCALE
        er = ref_data.e[ref_idx] / PLOT_E_SCALE

        ax_rho.plot(x_sim, sim_data.rho[sim_idx], color=color, linestyle="-", lw=lw_sim, alpha=0.92)
        ax_rho.plot(x_ref, ref_data.rho[ref_idx], color=color, linestyle="--", lw=lw_ref, label=ref_data.label, alpha=0.92)
        if shock_data is not None and shock_idx is not None and x_shock is not None:
            ax_rho.plot(
                x_shock,
                shock_data.rho[shock_idx],
                color=color,
                linestyle=":",
                lw=lw_shock,
                alpha=0.92,
            )

        ax_p.plot(x_sim, ps, color=color, linestyle="-", lw=lw_sim, alpha=0.92)
        ax_p.plot(x_ref, pr, color=color, linestyle="--", lw=lw_ref, alpha=0.92)
        if shock_data is not None and shock_idx is not None and x_shock is not None:
            ax_p.plot(x_shock, shock_data.p[shock_idx] / PLOT_P_SCALE, color=color, linestyle=":", lw=lw_shock, alpha=0.92)

        ax_u.plot(x_sim, us, color=color, linestyle="-", lw=lw_sim, alpha=0.92)
        ax_u.plot(x_ref, ur, color=color, linestyle="--", lw=lw_ref, alpha=0.92)
        if shock_data is not None and shock_idx is not None and x_shock is not None:
            ax_u.plot(x_shock, shock_data.u[shock_idx] / PLOT_U_SCALE, color=color, linestyle=":", lw=lw_shock, alpha=0.92)

        ax_e.plot(x_sim, es, color=color, linestyle="-", lw=lw_sim, alpha=0.92)
        ax_e.plot(x_ref, er, color=color, linestyle="--", lw=lw_ref, alpha=0.92)
        if shock_data is not None and shock_idx is not None and x_shock is not None:
            ax_e.plot(x_shock, shock_data.e[shock_idx] / PLOT_E_SCALE, color=color, linestyle=":", lw=lw_shock, alpha=0.92)

        if _has_Tm:
            ax_Tm.plot(
                x_sim,
                sim_data.T_material[sim_idx],
                color=color,
                linestyle="-",
                lw=lw_sim,
                alpha=0.92,
            )

        if _has_T:
            ax_T.plot(x_sim, sim_data.T[sim_idx], color=color, linestyle="-", lw=lw_sim, alpha=0.92)
            ax_T.plot(x_ref, ref_data.T[ref_idx], color=color, linestyle="--", lw=lw_ref, alpha=0.92)

        if _has_E_rad:
            ax_E_rad.plot(
                x_sim, sim_data.E_rad[sim_idx], color=color, linestyle="-", lw=lw_sim, alpha=0.92
            )
            ax_E_rad.plot(
                x_ref,
                ref_data.E_rad[ref_idx],
                color=color,
                linestyle="--",
                lw=lw_ref,
                alpha=0.92,
            )

        legend_handles.append(
            Line2D([0], [0], color=color, lw=lw_sim, linestyle="-", label=f"{sim_data.label} ({t_lbl})")
        )
        legend_handles.append(
            Line2D([0], [0], color=color, lw=lw_ref, linestyle="--", label=f"{ref_data.label} ({t_lbl})")
        )
        if shock_data is not None:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=lw_shock,
                    linestyle=":",
                    label=f"{shock_data.label} ({t_lbl})",
                )
            )

    ax_rho.set_ylabel(r"$\rho$ [g/cm³]")
    ax_p.set_ylabel(r"$P$ [MBar]")
    ax_u.set_ylabel(r"$u$ [km/s]")
    ax_e.set_ylabel(r"$e$ [hJ/g]")
    ax_Tm.set_ylabel(r"$T_{\mathrm{mat}}$ [HeV]")
    ax_T.set_ylabel(r"$T_{\mathrm{rad}}$ [HeV]")
    ax_E_rad.set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")

    plot_axes = (ax_rho, ax_p, ax_u, ax_e, ax_Tm, ax_T, ax_E_rad)
    for ax in plot_axes:
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.3)

    nh = len(legend_handles)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(6, nh),
        fontsize=6,
        framealpha=0.95,
    )

    if title is None:
        title = "Profile comparison (nearest snapshot per curve; solid=sim, dashed=ref)"
    fig.suptitle(title, fontsize=12, fontweight="medium")
    fig.tight_layout(rect=[0, 0.14, 1, 0.96])

    if savepath:
        sp = Path(savepath)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(sp), dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return [(fig, axes)]


# ============================================================================
# 4-Panel Comparison with Slider
# ============================================================================

def plot_comparison_slider(
    sim_data: HydroDataLike,
    ref_data: HydroDataLike,
    xaxis: str = "m",
    show: bool = True,
    title: str | None = None,
    shock_data: HydroDataLike | None = None,
):
    """
    Interactive 4-panel comparison with time slider.
    
    Parameters:
        sim_data: Simulation data (hydro_sim)
        ref_data: Reference data (shussman_shock_solver)
        xaxis: "m" for mass coordinate, "x" for position
        show: Whether to display the figure
        title: Optional figure title
        shock_data: Optional third dataset (e.g. shock solver P0*t^τ)
    """
    # Use simulation times as reference
    all_times = sim_data.times
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    plt.subplots_adjust(bottom=0.12)
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_e = axes[1, 0], axes[1, 1]
    ax_Tm, ax_T = axes[2, 0], axes[2, 1]
    ax_E_rad = axes[3, 0]
    axes[3, 1].set_visible(False)
    xlabel = r"Mass coordinate $m$ [g/cm²]" if xaxis == "m" else r"Position $x$ [cm]"
    
    # Initial plot (k=0)
    k = 0
    ref_k = interpolate_to_time(ref_data, all_times[k])
    
    x_sim = sim_data.m[k] if xaxis == "m" else sim_data.x[k]
    x_ref = ref_data.m[ref_k] if xaxis == "m" else ref_data.x[ref_k] 

    # Create line objects
    lines = {}
    shock_k = interpolate_to_time(shock_data, all_times[k]) if shock_data is not None else None
    x_shock = shock_data.m[shock_k] if shock_data and xaxis == "m" else (shock_data.x[shock_k] if shock_data else None)

    lines['sim_rho'], = ax_rho.plot(x_sim, sim_data.rho[k], 
                                     color=sim_data.color, lw=2, label=sim_data.label)
    lines['ref_rho'], = ax_rho.plot(x_ref, ref_data.rho[ref_k], 
                                     color=ref_data.color, linestyle='--', lw=2, label=ref_data.label)
    if shock_data is not None:
        lines['shock_rho'], = ax_rho.plot(x_shock, shock_data.rho[shock_k], 
                                          color=shock_data.color, linestyle=shock_data.linestyle, lw=2, label=shock_data.label)
    else:
        lines['shock_rho'], = ax_rho.plot([], [], color="green", linestyle="-.", lw=2)

    lines['sim_p'], = ax_p.plot(x_sim, sim_data.p[k] / PLOT_P_SCALE, color=sim_data.color, lw=2)
    lines['ref_p'], = ax_p.plot(x_ref, ref_data.p[ref_k] / PLOT_P_SCALE, color=ref_data.color, linestyle='--', lw=2)
    if shock_data is not None:
        lines['shock_p'], = ax_p.plot(x_shock, shock_data.p[shock_k] / PLOT_P_SCALE, color=shock_data.color, linestyle=shock_data.linestyle, lw=2)
    else:
        lines['shock_p'], = ax_p.plot([], [], color="green", linestyle="-.")
    lines['sim_u'], = ax_u.plot(x_sim, sim_data.u[k] / PLOT_U_SCALE, color=sim_data.color, lw=2)
    lines['ref_u'], = ax_u.plot(x_ref, ref_data.u[ref_k] / PLOT_U_SCALE, color=ref_data.color, linestyle='--', lw=2)
    if shock_data is not None:
        lines['shock_u'], = ax_u.plot(x_shock, shock_data.u[shock_k] / PLOT_U_SCALE, color=shock_data.color, linestyle=shock_data.linestyle, lw=2)
    else:
        lines['shock_u'], = ax_u.plot([], [], color="green", linestyle="-.")
    lines['sim_e'], = ax_e.plot(x_sim, sim_data.e[k] / PLOT_E_SCALE, color=sim_data.color, lw=2)
    lines['ref_e'], = ax_e.plot(x_ref, ref_data.e[ref_k] / PLOT_E_SCALE, color=ref_data.color, linestyle='--', lw=2)
    if shock_data is not None:
        lines['shock_e'], = ax_e.plot(x_shock, shock_data.e[shock_k] / PLOT_E_SCALE, color=shock_data.color, linestyle=shock_data.linestyle, lw=2)
    else:
        lines['shock_e'], = ax_e.plot([], [], color="green", linestyle="-.")

    # T_material (simulation only)
    _has_Tm = hasattr(sim_data, 'T_material') and sim_data.T_material
    if _has_Tm:
        lines['sim_Tm'], = ax_Tm.plot(x_sim, sim_data.T_material[k], color=sim_data.color, lw=2)
    else:
        lines['sim_Tm'], = ax_Tm.plot([], [], color=sim_data.color, lw=2)

    # T_rad and E_rad (optional; plot only if present)
    _has_T = hasattr(sim_data, 'T') and sim_data.T and hasattr(ref_data, 'T') and ref_data.T
    _has_E_rad = hasattr(sim_data, 'E_rad') and sim_data.E_rad and hasattr(ref_data, 'E_rad') and ref_data.E_rad
    if _has_T:
        lines['sim_T'], = ax_T.plot(x_sim, sim_data.T[k], color=sim_data.color, lw=2)
        lines['ref_T'], = ax_T.plot(x_ref, ref_data.T[ref_k], color=ref_data.color, linestyle='--', lw=2)
    else:
        lines['sim_T'], = ax_T.plot([], [], color=sim_data.color, lw=2)
        lines['ref_T'], = ax_T.plot([], [], color=ref_data.color, linestyle='--', lw=2)
    if _has_E_rad:
        lines['sim_E_rad'], = ax_E_rad.plot(x_sim, sim_data.E_rad[k], color=sim_data.color, lw=2)
        lines['ref_E_rad'], = ax_E_rad.plot(x_ref, ref_data.E_rad[ref_k], color=ref_data.color, linestyle='--', lw=2)
    else:
        lines['sim_E_rad'], = ax_E_rad.plot([], [], color=sim_data.color, lw=2)
        lines['ref_E_rad'], = ax_E_rad.plot([], [], color=ref_data.color, linestyle='--', lw=2)
    
    ax_rho.set_ylabel(r"$\rho$ [g/cm³]")
    ax_rho.legend(loc="best")
    ax_rho.grid(True, alpha=0.3)
    
    ax_p.set_ylabel(r"$P$ [MBar]")
    ax_p.grid(True, alpha=0.3)
    
    ax_u.set_ylabel(r"$u$ [km/s]")
    ax_u.grid(True, alpha=0.3)
    
    ax_e.set_ylabel(r"$e$ [hJ/g]")
    ax_e.grid(True, alpha=0.3)
    
    ax_Tm.set_ylabel(r"$T_{\mathrm{mat}}$ [HeV]")
    ax_Tm.grid(True, alpha=0.3)
    
    ax_T.set_ylabel(r"$T_{\mathrm{rad}}$ [HeV]")
    ax_T.grid(True, alpha=0.3)
    
    ax_E_rad.set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    ax_E_rad.grid(True, alpha=0.3)

    for ax in (ax_rho, ax_p, ax_u, ax_e, ax_Tm, ax_T, ax_E_rad):
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", labelbottom=True)
    
    title_text = fig.suptitle("")
    
    def set_title(k, ref_k):
        sim_t = sim_data.times[k]
        ref_t = ref_data.times[ref_k]
        base_title = title if title else "Shock Comparison"
        title_text.set_text(
            f"{base_title}\nSimulation: t={sim_t:.2e} s, Reference: t={ref_t:.2e} s"
        )
    
    set_title(k, ref_k)
    
    # Slider
    ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, "Frame", 0, len(all_times) - 1, valinit=k, valstep=1)
    
    def update(val):
        k = int(slider.val)
        ref_k = interpolate_to_time(ref_data, all_times[k])
        shock_k = interpolate_to_time(shock_data, all_times[k]) if shock_data is not None else None
        
        x_sim = sim_data.m[k] if xaxis == "m" else sim_data.x[k]
        x_ref = ref_data.m[ref_k] if xaxis == "m" else ref_data.x[ref_k]
        x_shock = shock_data.m[shock_k] if shock_data and xaxis == "m" else (shock_data.x[shock_k] if shock_data else np.array([]))
        
        lines['sim_rho'].set_data(x_sim, sim_data.rho[k])
        lines['ref_rho'].set_data(x_ref, ref_data.rho[ref_k])
        if shock_data is not None:
            lines['shock_rho'].set_data(x_shock, shock_data.rho[shock_k])
        
        lines['sim_p'].set_data(x_sim, sim_data.p[k] / PLOT_P_SCALE)
        lines['ref_p'].set_data(x_ref, ref_data.p[ref_k] / PLOT_P_SCALE)
        if shock_data is not None:
            lines['shock_p'].set_data(x_shock, shock_data.p[shock_k] / PLOT_P_SCALE)
        
        lines['sim_u'].set_data(x_sim, sim_data.u[k] / PLOT_U_SCALE)
        lines['ref_u'].set_data(x_ref, ref_data.u[ref_k] / PLOT_U_SCALE)
        if shock_data is not None:
            lines['shock_u'].set_data(x_shock, shock_data.u[shock_k] / PLOT_U_SCALE)
        
        lines['sim_e'].set_data(x_sim, sim_data.e[k] / PLOT_E_SCALE)
        lines['ref_e'].set_data(x_ref, ref_data.e[ref_k] / PLOT_E_SCALE)
        if shock_data is not None:
            lines['shock_e'].set_data(x_shock, shock_data.e[shock_k] / PLOT_E_SCALE)
        
        if _has_Tm:
            lines['sim_Tm'].set_data(x_sim, sim_data.T_material[k])
        if hasattr(sim_data, 'T') and sim_data.T and hasattr(ref_data, 'T') and ref_data.T:
            lines['sim_T'].set_data(x_sim, sim_data.T[k])
            lines['ref_T'].set_data(x_ref, ref_data.T[ref_k])
        if hasattr(sim_data, 'E_rad') and sim_data.E_rad and hasattr(ref_data, 'E_rad') and ref_data.E_rad:
            lines['sim_E_rad'].set_data(x_sim, sim_data.E_rad[k])
            lines['ref_E_rad'].set_data(x_ref, ref_data.E_rad[ref_k])
        
        set_title(k, ref_k)
        
        for ax in axes.flat:
            ax.relim()
            ax.autoscale_view()
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes, slider


# ============================================================================
# Multi-Time Overlay Plot
# ============================================================================

def plot_comparison_overlay(
    sim_data: HydroDataLike,
    ref_data: HydroDataLike,
    times: list | None = None,
    xaxis: str = "m",
    savepath: str | None = None,
    show: bool = True,
    title: str | None = None,
):
    """
    Plot multiple times overlaid on same axes with color gradient.
    
    Parameters:
        sim_data: Simulation data (hydro_sim)
        ref_data: Reference data (shussman_shock_solver)
        times: List of times to plot (None = use all ref times)
        xaxis: "m" for mass coordinate, "x" for position
        savepath: Optional path to save figure
        show: Whether to display the figure
        title: Optional figure title
    """
    if times is None:
        times = ref_data.times
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_e = axes[1, 0], axes[1, 1]
    
    xlabel = r"Mass coordinate $m$ [g/cm²]" if xaxis == "m" else r"Position $x$ [cm]"
    
    # Color gradients
    cmap_sim = plt.get_cmap("Blues")
    cmap_ref = plt.get_cmap("Reds")
    n_times = len(times)
    colors_sim = cmap_sim(np.linspace(0.3, 0.9, n_times))
    colors_ref = cmap_ref(np.linspace(0.3, 0.9, n_times))
    
    for i, t in enumerate(times):
        sim_k = interpolate_to_time(sim_data, t)
        ref_k = interpolate_to_time(ref_data, t)
        
        x_sim = sim_data.m[sim_k] if xaxis == "m" else sim_data.x[sim_k]
        x_ref = ref_data.m[ref_k] if xaxis == "m" else ref_data.x[ref_k]
        
        label_sim = f"Sim t={t:.1e}" if i == 0 else None
        label_ref = f"Ref t={t:.1e}" if i == 0 else None
        
        # Density
        ax_rho.plot(x_sim, sim_data.rho[sim_k], color=colors_sim[i], lw=1.5)
        ax_rho.plot(x_ref, ref_data.rho[ref_k], color=colors_ref[i], lw=1.5, linestyle='--')
        
        # Pressure
        ax_p.plot(x_sim, sim_data.p[sim_k] / PLOT_P_SCALE, color=colors_sim[i], lw=1.5)
        ax_p.plot(x_ref, ref_data.p[ref_k] / PLOT_P_SCALE, color=colors_ref[i], lw=1.5, linestyle='--')
        
        # Velocity
        ax_u.plot(x_sim, sim_data.u[sim_k] / PLOT_U_SCALE, color=colors_sim[i], lw=1.5)
        ax_u.plot(x_ref, ref_data.u[ref_k] / PLOT_U_SCALE, color=colors_ref[i], lw=1.5, linestyle='--')
        
        # Energy
        ax_e.plot(x_sim, sim_data.e[sim_k] / PLOT_E_SCALE, color=colors_sim[i], lw=1.5)
        ax_e.plot(x_ref, ref_data.e[ref_k] / PLOT_E_SCALE, color=colors_ref[i], lw=1.5, linestyle='--')
    
    # Labels and styling
    ax_rho.set_ylabel(r"$\rho$ [g/cm³]")
    ax_rho.set_xlabel(xlabel)
    ax_rho.tick_params(axis="x", labelbottom=True)
    ax_rho.grid(True, alpha=0.3)
    
    ax_p.set_ylabel(r"$P$ [MBar]")
    ax_p.set_xlabel(xlabel)
    ax_p.tick_params(axis="x", labelbottom=True)
    ax_p.grid(True, alpha=0.3)
    
    ax_u.set_ylabel(r"$u$ [km/s]")
    ax_u.set_xlabel(xlabel)
    ax_u.tick_params(axis="x", labelbottom=True)
    ax_u.grid(True, alpha=0.3)
    
    ax_e.set_ylabel(r"$e$ [hJ/g]")
    ax_e.set_xlabel(xlabel)
    ax_e.tick_params(axis="x", labelbottom=True)
    ax_e.grid(True, alpha=0.3)
    
    # Legend with custom handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Simulation'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Self-Similar'),
    ]
    ax_rho.legend(handles=legend_elements, loc="best")
    
    if title is None:
        title = "Shock Profile Comparison (multiple times)"
    fig.suptitle(title, fontsize=12, fontweight="medium")
    fig.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved figure to {savepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes


# ============================================================================
# Comparison GIF Animation
# ============================================================================

def save_comparison_gif(
    sim_data: HydroDataLike,
    ref_data: HydroDataLike,
    gif_path: str,
    xaxis: str = "m",
    fps: int = 10,
    stride: int = 1,
    title: str | None = None,
):
    """
    Save animated GIF comparing simulation and reference data.
    
    Parameters:
        sim_data: Simulation data
        ref_data: Reference data
        gif_path: Output file path
        xaxis: "m" for mass coordinate, "x" for position
        fps: Frames per second
        stride: Frame stride
        title: Optional figure title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_e = axes[1, 0], axes[1, 1]
    
    xlabel = r"Mass coordinate $m$ [g/cm²]" if xaxis == "m" else r"Position $x$ [cm]"
    
    # Initial plot
    k = 0
    ref_k = interpolate_to_time(ref_data, sim_data.times[k])
    
    x_sim = sim_data.m[k] if xaxis == "m" else sim_data.x[k]
    x_ref = ref_data.m[ref_k] if xaxis == "m" else ref_data.x[ref_k]
    
    lines = {}
    lines['sim_rho'], = ax_rho.plot(x_sim, sim_data.rho[k], 'b-', lw=2, label=sim_data.label)
    lines['ref_rho'], = ax_rho.plot(x_ref, ref_data.rho[ref_k], 'r--', lw=2, label=ref_data.label)
    lines['sim_p'], = ax_p.plot(x_sim, sim_data.p[k] / PLOT_P_SCALE, 'b-', lw=2)
    lines['ref_p'], = ax_p.plot(x_ref, ref_data.p[ref_k] / PLOT_P_SCALE, 'r--', lw=2)
    lines['sim_u'], = ax_u.plot(x_sim, sim_data.u[k] / PLOT_U_SCALE, 'b-', lw=2)
    lines['ref_u'], = ax_u.plot(x_ref, ref_data.u[ref_k] / PLOT_U_SCALE, 'r--', lw=2)
    lines['sim_e'], = ax_e.plot(x_sim, sim_data.e[k] / PLOT_E_SCALE, 'b-', lw=2)
    lines['ref_e'], = ax_e.plot(x_ref, ref_data.e[ref_k] / PLOT_E_SCALE, 'r--', lw=2)
    
    ax_rho.set_ylabel(r"$\rho$ [g/cm³]")
    ax_rho.legend(loc="best")
    ax_rho.grid(True, alpha=0.3)
    ax_p.set_ylabel(r"$P$ [MBar]")
    ax_p.grid(True, alpha=0.3)
    ax_u.set_ylabel(r"$u$ [km/s]")
    ax_u.grid(True, alpha=0.3)
    ax_e.set_ylabel(r"$e$ [hJ/g]")
    ax_e.grid(True, alpha=0.3)
    for ax in (ax_rho, ax_p, ax_u, ax_e):
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", labelbottom=True)
    
    title_text = fig.suptitle("")
    
    frame_ids = np.arange(0, len(sim_data.times), stride)
    
    def init():
        return list(lines.values())
    
    def update(frame_idx):
        k = frame_ids[frame_idx]
        ref_k = interpolate_to_time(ref_data, sim_data.times[k])
        
        x_sim = sim_data.m[k] if xaxis == "m" else sim_data.x[k]
        x_ref = ref_data.m[ref_k] if xaxis == "m" else ref_data.x[ref_k]
        
        lines['sim_rho'].set_data(x_sim, sim_data.rho[k])
        lines['ref_rho'].set_data(x_ref, ref_data.rho[ref_k])
        lines['sim_p'].set_data(x_sim, sim_data.p[k] / PLOT_P_SCALE)
        lines['ref_p'].set_data(x_ref, ref_data.p[ref_k] / PLOT_P_SCALE)
        lines['sim_u'].set_data(x_sim, sim_data.u[k] / PLOT_U_SCALE)
        lines['ref_u'].set_data(x_ref, ref_data.u[ref_k] / PLOT_U_SCALE)
        lines['sim_e'].set_data(x_sim, sim_data.e[k] / PLOT_E_SCALE)
        lines['ref_e'].set_data(x_ref, ref_data.e[ref_k] / PLOT_E_SCALE)
        
        sim_t = sim_data.times[k]
        ref_t = ref_data.times[ref_k]
        base_title = title if title else "Shock Comparison"
        title_text.set_text(f"{base_title}\nSim: t={sim_t:.2e}, Ref: t={ref_t:.2e}")
        
        for ax in axes.flat:
            ax.relim()
            ax.autoscale_view()
        
        return list(lines.values())
    
    anim = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=False)
    
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved comparison GIF to {gif_path}")

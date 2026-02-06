# compare_shock_plots.py
"""
Comparison plotting for driven shock simulations.
Compares results from hydro_sim (Lagrangian simulation) with
shussman_shock_solver (self-similar solution).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SimulationData:
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


def load_shussman_data(npz_path: str | Path) -> SimulationData:
    """Load data from shussman_shock_solver NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    
    times = np.asarray(data["times"], float)
    
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
    
    return SimulationData(
        times=times,
        m=m, x=x, rho=rho, p=p, u=u, e=e,
        label="Self-Similar (Shussman)",
        color="red",
        linestyle="--"
    )


def load_hydro_history(history) -> SimulationData:
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
    
    return SimulationData(
        times=times,
        m=m_list, x=x_list, rho=rho_list, p=p_list, u=u_list, e=e_list,
        label="Hydro Simulation",
        color="blue",
        linestyle="-"
    )


def interpolate_to_time(data: SimulationData, target_time: float) -> int:
    """Find the index of the closest time in data."""
    idx = np.argmin(np.abs(data.times - target_time))
    return idx


# ============================================================================
# 4-Panel Comparison Plot (Single Time)
# ============================================================================

def plot_comparison_single_time(
    sim_data: SimulationData,
    ref_data: SimulationData,
    time: float,
    xaxis: str = "m",  # "m" for mass coordinate, "x" for position
    savepath: str | None = None,
    show: bool = True,
    title: str | None = None,
):
    """
    Plot 4-panel comparison (rho, P, u, e) at a single time.
    
    Parameters:
        sim_data: Simulation data (hydro_sim)
        ref_data: Reference data (shussman_shock_solver)
        time: Time to plot
        xaxis: "m" for mass coordinate, "x" for position
        savepath: Optional path to save figure
        show: Whether to display the figure
        title: Optional figure title
    """
    # Find closest time indices
    sim_idx = interpolate_to_time(sim_data, time)
    ref_idx = interpolate_to_time(ref_data, time)
    
    actual_sim_time = sim_data.times[sim_idx]
    actual_ref_time = ref_data.times[ref_idx]
    
    # Get data
    if xaxis == "m":
        x_sim = sim_data.m[sim_idx]
        x_ref = ref_data.m[ref_idx]
        xlabel = r"Mass coordinate $m$ [g/cm²]"
    else:
        x_sim = sim_data.x[sim_idx]
        x_ref = ref_data.x[ref_idx]
        xlabel = r"Position $x$ [cm]"
    
    print(ref_data.times[ref_idx])
    print(sim_data.times[sim_idx])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_e = axes[1, 0], axes[1, 1]
    
    # Plot density
    ax_rho.plot(x_sim, sim_data.rho[sim_idx], 
                color=sim_data.color, linestyle=sim_data.linestyle, 
                lw=2, label=f"{sim_data.label} (t={actual_sim_time:.2e})")
    ax_rho.plot(x_ref, ref_data.rho[ref_idx], 
                color=ref_data.color, linestyle=ref_data.linestyle, 
                lw=2, label=f"{ref_data.label} (t={actual_ref_time:.2e})")
    ax_rho.set_ylabel(r"Density $\rho$ [g/cm³]")
    ax_rho.legend(loc="best", fontsize=9)
    ax_rho.grid(True, alpha=0.3)
    
    # Plot pressure
    ax_p.plot(x_sim, sim_data.p[sim_idx], 
              color=sim_data.color, linestyle=sim_data.linestyle, lw=2)
    ax_p.plot(x_ref, ref_data.p[ref_idx], 
              color=ref_data.color, linestyle=ref_data.linestyle, lw=2)
    ax_p.set_ylabel(r"Pressure $P$ [dyne/cm²]")
    ax_p.grid(True, alpha=0.3)
    
    # Plot velocity
    ax_u.plot(x_sim, sim_data.u[sim_idx], 
              color=sim_data.color, linestyle=sim_data.linestyle, lw=2)
    ax_u.plot(x_ref, ref_data.u[ref_idx], 
              color=ref_data.color, linestyle=ref_data.linestyle, lw=2)
    ax_u.set_ylabel(r"Velocity $u$ [cm/s]")
    ax_u.set_xlabel(xlabel)
    ax_u.grid(True, alpha=0.3)
    
    # Plot energy
    ax_e.plot(x_sim, sim_data.e[sim_idx], 
              color=sim_data.color, linestyle=sim_data.linestyle, lw=2)
    ax_e.plot(x_ref, ref_data.e[ref_idx], 
              color=ref_data.color, linestyle=ref_data.linestyle, lw=2)
    ax_e.set_ylabel(r"Specific energy $e$ [erg/g]")
    ax_e.set_xlabel(xlabel)
    ax_e.grid(True, alpha=0.3)
    
    # Title
    if title is None:
        title = f"Shock Profile Comparison at t ≈ {time:.2e} s"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes


# ============================================================================
# 4-Panel Comparison with Slider
# ============================================================================

def plot_comparison_slider(
    sim_data: SimulationData,
    ref_data: SimulationData,
    xaxis: str = "m",
    show: bool = True,
    title: str | None = None,
):
    """
    Interactive 4-panel comparison with time slider.
    
    Parameters:
        sim_data: Simulation data (hydro_sim)
        ref_data: Reference data (shussman_shock_solver)
        xaxis: "m" for mass coordinate, "x" for position
        show: Whether to display the figure
        title: Optional figure title
    """
    # Use simulation times as reference
    all_times = sim_data.times
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    plt.subplots_adjust(bottom=0.15)
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_e = axes[1, 0], axes[1, 1]
    
    xlabel = r"Mass coordinate $m$" if xaxis == "m" else r"Position $x$"
    
    # Initial plot (k=0)
    k = 0
    ref_k = interpolate_to_time(ref_data, all_times[k])
    
    x_sim = sim_data.m[k] if xaxis == "m" else sim_data.x[k]
    x_ref = ref_data.m[ref_k] if xaxis == "m" else ref_data.x[ref_k]
    
    # Create line objects
    lines = {}
    lines['sim_rho'], = ax_rho.plot(x_sim, sim_data.rho[k], 
                                     color=sim_data.color, lw=2, label=sim_data.label)
    lines['ref_rho'], = ax_rho.plot(x_ref, ref_data.rho[ref_k], 
                                     color=ref_data.color, linestyle='--', lw=2, label=ref_data.label)
    
    lines['sim_p'], = ax_p.plot(x_sim, sim_data.p[k], color=sim_data.color, lw=2)
    lines['ref_p'], = ax_p.plot(x_ref, ref_data.p[ref_k], color=ref_data.color, linestyle='--', lw=2)
    
    lines['sim_u'], = ax_u.plot(x_sim, sim_data.u[k], color=sim_data.color, lw=2)
    lines['ref_u'], = ax_u.plot(x_ref, ref_data.u[ref_k], color=ref_data.color, linestyle='--', lw=2)
    
    lines['sim_e'], = ax_e.plot(x_sim, sim_data.e[k], color=sim_data.color, lw=2)
    lines['ref_e'], = ax_e.plot(x_ref, ref_data.e[ref_k], color=ref_data.color, linestyle='--', lw=2)
    
    # Labels
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.legend(loc="best")
    ax_rho.grid(True, alpha=0.3)
    
    ax_p.set_ylabel(r"$P$")
    ax_p.grid(True, alpha=0.3)
    
    ax_u.set_ylabel(r"$u$")
    ax_u.set_xlabel(xlabel)
    ax_u.grid(True, alpha=0.3)
    
    ax_e.set_ylabel(r"$e$")
    ax_e.set_xlabel(xlabel)
    ax_e.grid(True, alpha=0.3)
    
    title_text = fig.suptitle("", fontsize=12)
    
    def set_title(k, ref_k):
        sim_t = sim_data.times[k]
        ref_t = ref_data.times[ref_k]
        base_title = title if title else "Shock Comparison"
        title_text.set_text(f"{base_title}\nSim: t={sim_t:.2e}, Ref: t={ref_t:.2e}")
    
    set_title(k, ref_k)
    
    # Slider
    ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, "Frame", 0, len(all_times) - 1, valinit=k, valstep=1)
    
    def update(val):
        k = int(slider.val)
        ref_k = interpolate_to_time(ref_data, all_times[k])
        
        x_sim = sim_data.m[k] if xaxis == "m" else sim_data.x[k]
        x_ref = ref_data.m[ref_k] if xaxis == "m" else ref_data.x[ref_k]
        
        lines['sim_rho'].set_data(x_sim, sim_data.rho[k])
        lines['ref_rho'].set_data(x_ref, ref_data.rho[ref_k])
        
        lines['sim_p'].set_data(x_sim, sim_data.p[k])
        lines['ref_p'].set_data(x_ref, ref_data.p[ref_k])
        
        lines['sim_u'].set_data(x_sim, sim_data.u[k])
        lines['ref_u'].set_data(x_ref, ref_data.u[ref_k])
        
        lines['sim_e'].set_data(x_sim, sim_data.e[k])
        lines['ref_e'].set_data(x_ref, ref_data.e[ref_k])
        
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
    sim_data: SimulationData,
    ref_data: SimulationData,
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
    
    xlabel = r"Mass coordinate $m$" if xaxis == "m" else r"Position $x$"
    
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
        ax_p.plot(x_sim, sim_data.p[sim_k], color=colors_sim[i], lw=1.5)
        ax_p.plot(x_ref, ref_data.p[ref_k], color=colors_ref[i], lw=1.5, linestyle='--')
        
        # Velocity
        ax_u.plot(x_sim, sim_data.u[sim_k], color=colors_sim[i], lw=1.5)
        ax_u.plot(x_ref, ref_data.u[ref_k], color=colors_ref[i], lw=1.5, linestyle='--')
        
        # Energy
        ax_e.plot(x_sim, sim_data.e[sim_k], color=colors_sim[i], lw=1.5)
        ax_e.plot(x_ref, ref_data.e[ref_k], color=colors_ref[i], lw=1.5, linestyle='--')
    
    # Labels and styling
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.grid(True, alpha=0.3)
    
    ax_p.set_ylabel(r"$P$")
    ax_p.grid(True, alpha=0.3)
    
    ax_u.set_ylabel(r"$u$")
    ax_u.set_xlabel(xlabel)
    ax_u.grid(True, alpha=0.3)
    
    ax_e.set_ylabel(r"$e$")
    ax_e.set_xlabel(xlabel)
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
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
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
    sim_data: SimulationData,
    ref_data: SimulationData,
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
    
    xlabel = r"Mass coordinate $m$" if xaxis == "m" else r"Position $x$"
    
    # Initial plot
    k = 0
    ref_k = interpolate_to_time(ref_data, sim_data.times[k])
    
    x_sim = sim_data.m[k] if xaxis == "m" else sim_data.x[k]
    x_ref = ref_data.m[ref_k] if xaxis == "m" else ref_data.x[ref_k]
    
    lines = {}
    lines['sim_rho'], = ax_rho.plot(x_sim, sim_data.rho[k], 'b-', lw=2, label=sim_data.label)
    lines['ref_rho'], = ax_rho.plot(x_ref, ref_data.rho[ref_k], 'r--', lw=2, label=ref_data.label)
    lines['sim_p'], = ax_p.plot(x_sim, sim_data.p[k], 'b-', lw=2)
    lines['ref_p'], = ax_p.plot(x_ref, ref_data.p[ref_k], 'r--', lw=2)
    lines['sim_u'], = ax_u.plot(x_sim, sim_data.u[k], 'b-', lw=2)
    lines['ref_u'], = ax_u.plot(x_ref, ref_data.u[ref_k], 'r--', lw=2)
    lines['sim_e'], = ax_e.plot(x_sim, sim_data.e[k], 'b-', lw=2)
    lines['ref_e'], = ax_e.plot(x_ref, ref_data.e[ref_k], 'r--', lw=2)
    
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.legend(loc="best")
    ax_rho.grid(True, alpha=0.3)
    ax_p.set_ylabel(r"$P$")
    ax_p.grid(True, alpha=0.3)
    ax_u.set_ylabel(r"$u$")
    ax_u.set_xlabel(xlabel)
    ax_u.grid(True, alpha=0.3)
    ax_e.set_ylabel(r"$e$")
    ax_e.set_xlabel(xlabel)
    ax_e.grid(True, alpha=0.3)
    
    title_text = fig.suptitle("", fontsize=12)
    
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
        lines['sim_p'].set_data(x_sim, sim_data.p[k])
        lines['ref_p'].set_data(x_ref, ref_data.p[ref_k])
        lines['sim_u'].set_data(x_sim, sim_data.u[k])
        lines['ref_u'].set_data(x_ref, ref_data.u[ref_k])
        lines['sim_e'].set_data(x_sim, sim_data.e[k])
        lines['ref_e'].set_data(x_ref, ref_data.e[ref_k])
        
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

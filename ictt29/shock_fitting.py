# ictt29/plot_piston_shock.py
"""
verification and similarity profile fitting script for the Piston Shock regime.

Compiles and compares:
1. 1D Rad-Hydro Simulation (hydro-only scenario).
2. PistonShock reference solver (piston-driven shock wave).

Outputs generated inside results/ictt/ (split by graph types):
- xt/ Space-time trajectory and fronts (PNG)
- material_hydro/ e, u, p, rho vs mass coordinate m (PNG)
- self_similar/ Dimensionless similarity profiles R, P, U, T with curve-fits (PNG)
- self_similar/ Dimensionless velocity fits standalone comparing alternative fits (PNG)
- evolution/ Animated 5-way GIFs (Simulation vs PistonShock).
"""
from __future__ import annotations

import os
import sys
import pickle
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import scipy.integrate

# Monkeypatch scipy.integrate.simps to scipy.integrate.simpson for modern Scipy versions
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

# Monkeypatch numpy.trapz to scipy.integrate.trapezoid for modern Numpy 2.x versions
if not hasattr(np, "trapz"):
    if hasattr(scipy.integrate, "trapezoid"):
        np.trapz = scipy.integrate.trapezoid
    else:
        np.trapz = scipy.integrate.trapz


# Ensure proper package and solver imports
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_MENAHEM_DIR = Path(__file__).resolve().parents[1] / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_CONSTANT_PRESSURE,
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE,
)
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _shock_kwargs_from_case,
    _ns_amplitude_rescale,
    _build_mass_grid,
)
from project3_code.hydro_sim.plotting.hydro_plots import _create_7panel_vertical_figure

from piston_shock import PistonShock


# =============================================================================
# Helper functions for shock detection and rolling average
# =============================================================================

def find_shock_front(
    rho: np.ndarray,
    m_coordinate: np.ndarray,
    *,
    rho_unshocked: float,
    gamma: float,
    Hugoniot_threshold: float = 0.5,
) -> tuple[int, float]:
    """Detect the shock as the right edge of the compressed region."""
    rho_arr = np.asarray(rho, dtype=float)
    m_arr = np.asarray(m_coordinate, dtype=float)
    n = rho_arr.size
    if n < 3:
        return -1, float("nan")

    rho_thresh = float(rho_unshocked) * Hugoniot_threshold * (gamma + 1.0) / (gamma - 1.0)
    compressed = rho_arr > rho_thresh
    compressed_idx = np.flatnonzero(compressed)
    if compressed_idx.size > 0:
        i = int(compressed_idx[-1])
        return i, float(rho_arr[i])

    drho_dm = np.gradient(rho_arr, m_arr)
    i_steep = int(np.argmin(drho_dm))
    if np.isfinite(drho_dm[i_steep]):
        return i_steep, float(rho_arr[i_steep])
    return -1, float("nan")


def _sim_piston_position(
    x_cells: np.ndarray,
    p_cells: np.ndarray,
    p_drive: float,
    rel_tol: float,
) -> tuple[int, float]:
    """First (leftmost) cell close enough to drive pressure; fallback to nearest."""
    denom = max(abs(float(p_drive)), 1e-30)
    rel_err = np.abs(np.asarray(p_cells, dtype=float) - float(p_drive)) / denom
    close_idx = np.flatnonzero(rel_err <= rel_tol)
    if close_idx.size > 0:
        i = int(close_idx[0])
        return i, float(x_cells[i])
    i = int(np.argmin(rel_err))
    return i, float(x_cells[i])


def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(a, dtype=float)
    w = int(max(1, window))
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(np.asarray(a, dtype=float), kernel, mode="same")


def _get_equally_spaced_elements(times: np.ndarray, n: int) -> np.ndarray:
    ideal_times = np.linspace(times.min(), times.max(), n)
    indices = np.searchsorted(times, ideal_times)
    indices = np.clip(indices, 0, len(times) - 1)
    for i in range(len(indices)):
        idx = indices[i]
        if idx > 0 and abs(times[idx-1] - ideal_times[i]) < abs(times[idx] - ideal_times[i]):
            indices[i] = idx - 1
    unique_indices = np.unique(indices)
    return times[unique_indices]


# =============================================================================
# Plotting functions
# =============================================================================

def plot_xt_trajectories(history, case, xt_path, case_title, solver=None):
    """Plot cell boundaries x(t) and diagnosed fronts (Simulation vs Analytic)."""
    print(f"Generating space-time (xt) plot for {case_title}...")
    times = np.asarray(history.t, dtype=float)
    x_sim = np.asarray(history.x, dtype=float)
    n_cells = x_sim.shape[1] - 1
    
    if solver is None:
        solver = PistonShock(**_shock_kwargs_from_case(case))
    mass_grid = _build_mass_grid(case, num_cells=n_cells)
    
    times_model = _get_equally_spaced_elements(times, 200)
    results = []
    for t in times_model:
        sol = solver.solve(mass=mass_grid, time=max(float(t), 1e-18))
        results.append(sol)
        
    position_times = np.array([r["position"] for r in results]).T
    shock_position = np.array([r["shock_position"] for r in results], dtype=float)
    piston_position = np.array([r["piston_position"] for r in results], dtype=float)
    
    # 2) Simulation piston/shock trajectories
    x_piston_sim = np.full(times.size, np.nan, dtype=float)
    x_shock_sim = np.full(times.size, np.nan, dtype=float)
    p0_barye = float(case.P0_Barye)
    tau = float(case.tau)
    
    for k, t in enumerate(times):
        xk = np.asarray(history.x[k], dtype=float)
        pk = np.asarray(history.p[k], dtype=float)
        rhok = np.asarray(history.rho[k], dtype=float)
        mk = np.asarray(history.m[k], dtype=float)
        
        # Boundary pressure
        if t <= 0.0:
            p_drive = 0.0
        else:
            p_drive = p0_barye * ((t / 1e-9) ** tau)
            
        _, x_p = _sim_piston_position(xk, pk, p_drive, rel_tol=0.05)
        x_piston_sim[k] = x_p
        
        rhok_smooth = _rolling_mean(rhok, 5)
        ishock, _ = find_shock_front(
            rhok_smooth,
            mk,
            rho_unshocked=float(case.rho0),
            gamma=float(case.r) + 1.0,
            Hugoniot_threshold=0.5,
        )
        if ishock >= 1:
            x_shock_sim[k] = float(xk[ishock])
            
    # Apply linear regression correction for small times (t < 0.002 ns) in shock detection
    later_times = np.array([])
    later_x = np.array([])
    for k in range(1, times.size):
        t_ns = times[k] * 1e9
        if t_ns >= 0.002 and not np.isnan(x_shock_sim[k]):
            later_times = np.append(later_times, times[k])
            later_x = np.append(later_x, x_shock_sim[k])
            
    if len(later_times) >= 2:
        slope, intercept = np.polyfit(np.log(later_times+1e-20), np.log(later_x), 1)
        for k in range(1, times.size):
            t_ns = times[k] * 1e9
            if t_ns < 0.002:
                x_shock_sim[k] = np.exp(slope * np.log(times[k] + 1e-20) + intercept)
                
    # 3) Setup figure
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    
    # Plot mass trajectories
    NUM_PRESENTED_CELLS = 50
    chosen_cell_indices = _get_equally_spaced_elements(np.arange(n_cells), NUM_PRESENTED_CELLS)
    
    matched_sim_coordinates = np.zeros_like(x_sim, dtype=float)
    matched_sim_coordinates[:, 1:] = 0.5 * (x_sim[:, 1:] + x_sim[:, :-1])
    
    legend_added = False
    for j in chosen_cell_indices:
        lbl_sim = "Simulation Cells" if not legend_added else None
        lbl_men = "Analytic Cells" if not legend_added else None
        ax.plot(times * 1e9, matched_sim_coordinates[:, j + 1], color="black", lw=0.4, alpha=0.8, label=lbl_sim)
        ax.plot(times_model * 1e9, position_times[j + 2], color="blue", lw=0.4, alpha=0.7, label=lbl_men)
        legend_added = True
        
    # Plot bold fronts
    ax.plot(times * 1e9, x_shock_sim, lw=2.5, c="red", label="Shock (simulation)")
    ax.plot(times_model * 1e9, shock_position, lw=2.0, ls="--", c="darkred", label="Shock (Menahem)")
    ax.plot(times * 1e9, x_piston_sim, lw=2.5, c="green", label="Piston (simulation)")
    ax.plot(times_model * 1e9, piston_position, lw=2.0, ls="--", c="darkgreen", label="Piston (Menahem)")
    
    ax.set_xlabel(r"$t$ [ns]", fontsize=12)
    ax.set_ylabel(r"$x$ [cm]", fontsize=12)
    ax.set_title(f"Space-Time (xt) Trajectories and Fronts\n{case_title}", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_xlim(0, times.max() * 1e9)
    ax.set_ylim(0, max(shock_position.max(), x_shock_sim[np.isfinite(x_shock_sim)].max()) * 1.05)
    
    fig.tight_layout()
    fig.savefig(xt_path, dpi=200)
    plt.close(fig)
    print(f"Saved xt trajectory to {xt_path}")


def plot_material_hydro_profiles(history, menahem_ref, material_hydro_path, case_title):
    """Plot overlay profiles of e, u, p, rho vs m (mass coordinate) at evolution snippets."""
    print(f"Generating material hydrodynamics profiles for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    t_max = np.max(history.t)
    target_times = [0.25 * t_max, 0.50 * t_max, 0.75 * t_max]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(target_times)))
    
    # Loop over times
    for t_target, color in zip(target_times, colors):
        # 1) Simulation
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        
        # 2) Menahem
        idx_men = np.argmin(np.abs(np.array(menahem_ref.times) - t_target))
        m_men = menahem_ref.m[idx_men]
        
        # Panel (0,0): Density rho
        axes[0, 0].plot(m_sim, history.rho[idx_sim], color=color, linestyle='-', lw=1.8)
        axes[0, 0].plot(m_men, menahem_ref.rho[idx_men], color=color, linestyle='--', lw=1.5)
        
        # Panel (0,1): Pressure p [MBar]
        axes[0, 1].plot(m_sim, history.p[idx_sim] / 1e12, color=color, linestyle='-', lw=1.8)
        axes[0, 1].plot(m_men, menahem_ref.p[idx_men] / 1e12, color=color, linestyle='--', lw=1.5)
        
        # Panel (1,0): Velocity u [km/s]
        axes[1, 0].plot(m_sim, history.u[idx_sim] / 1e5, color=color, linestyle='-', lw=1.8)
        axes[1, 0].plot(m_men, menahem_ref.u[idx_men] / 1e5, color=color, linestyle='--', lw=1.5)
        
        # Panel (1,1): Specific Energy e [1e9 erg/g]
        axes[1, 1].plot(m_sim, history.e[idx_sim] / 1e9, color=color, linestyle='-', lw=1.8)
        axes[1, 1].plot(m_men, menahem_ref.e[idx_men] / 1e9, color=color, linestyle='--', lw=1.5)
        
    # Labels and Titles
    axes[0, 0].set_ylabel(r"$\rho$ [g/cm³]", fontsize=12)
    axes[0, 1].set_ylabel(r"$P$ [MBar]", fontsize=12)
    axes[1, 0].set_ylabel(r"$u$ [km/s]", fontsize=12)
    axes[1, 1].set_ylabel(r"$e$ [$10^9$ erg/g]", fontsize=12)
    
    for ax in axes.flat:
        ax.set_xlabel(r"Mass coordinate $m$ [g/cm²]", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        
    # Legend
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', lw=1.8, label='simulation'),
        Line2D([0], [0], color='black', linestyle='--', lw=1.5, label='Menahem'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[0], markersize=8, label=f'{0.25*t_max*1e9:.2f} ns'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[1], markersize=8, label=f'{0.50*t_max*1e9:.2f} ns'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[2], markersize=8, label=f'{0.75*t_max*1e9:.2f} ns'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc='best', fontsize=9.5)
    
    fig.suptitle(f"Material Hydrodynamics Profiles\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(material_hydro_path, dpi=200)
    plt.close(fig)
    print(f"Saved material hydro profiles to {material_hydro_path}")


# =============================================================================
# Self-similar extraction and fitting
# =============================================================================

def plot_and_fit_self_similar(case, self_similar_path, standalone_path, case_title):
    """Solve the PistonShock similarity ODEs, extract R, P, U, T, fit and plot."""
    print(f"Solving self-similar similarity ODEs for {case_title}...")
    
    # 1) Setup solver with case parameters
    tau = float(case.tau or 0.0)
    p0 = _ns_amplitude_rescale(float(case.P0_Barye), tau)
    # Get omega generally, defaults to 0.0 if not specified
    omega = float(getattr(case, "omega", 0.0))
    gamma = float(case.r) + 1.0
    
    solver = PistonShock(
        rho0=float(case.rho0),
        omega=omega,
        p0=p0,
        tau=tau,
        gamma=gamma
    )
    
    # Grid of y = xsi / xsi_s in [0, 1]
    y_grid = np.linspace(0.0, 1.0, 500)
    xsi_vec = y_grid * solver.xsi_s
    # Avoid y=0 exactly to prevent singular divisions
    xsi_vec[0] = 1e-10
    
    V_val, U_val, P_val = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    # Translate specific volume back to density
    R_val = np.where(V_val > 0, 1.0 / V_val, np.nan)
    T_val = P_val * V_val / solver.r
    
    valid_idx = (y_grid > 0.005) & np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(R_val) & np.isfinite(T_val)
    
    y_valid = y_grid[valid_idx]
    V_valid = V_val[valid_idx]
    U_valid = U_val[valid_idx]
    P_valid = P_val[valid_idx]
    R_valid = R_val[valid_idx]
    T_valid = T_val[valid_idx]
    
    # Boundary values at front (shock, y=1) and origin (piston, y=0)
    V_s = V_valid[-1]
    U_s = U_valid[-1]
    P_s = P_valid[-1]
    R_s = R_valid[-1]
    T_s = T_valid[-1]
    U_0 = U_valid[0]
    
    # Determine if profiles are constant (constant pressure drive in uniform medium)
    is_constant = (np.abs(P_valid.max() - P_valid.min()) < 1e-6) and (np.abs(U_valid.max() - U_valid.min()) < 1e-6)
    
    # Perform curve fitting for R, P, U, T using minimal parameter formulations
    def compute_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1.0 - ss_res / (ss_tot + 1e-30)

    # 1-parameter power laws
    def power_law_R(y, d):
        return R_s * (y**(-d))
        
    def power_law_P(y, d):
        return 1.0 - (1.0 - P_s) * (y**d)
        
    def power_law_U(y, d):
        return U_0 - (U_0 - U_s) * (y**d)
        
    def power_law_T(y, d):
        return T_s * (y**d)
        
    # Extra fitting options for Velocity (U) to find the most clever fit
    def fit_u_1(y, d):
        # 1-param baseline
        return U_0 - (U_0 - U_s) * (y**d)
    def fit_u_2(y, c, d):
        # 2-param c - (c-Us)*y^d
        return c - (c - U_s) * (y**d)
    def fit_u_3(y, c, a, b):
        # 3-param generalized: c - a*y^b
        return c - a * (y**b)
    def fit_u_4(y, c, d):
        # 2-param rational
        return c * (1.0 - y) / (1.0 + d * y) + U_s * y
    def fit_u_5(y, c, a, b):
        # 3-param singular power law: c * (1-y^a) * y^-b + U_s * y
        return c * (1.0 - y**a) * (y**(-b)) + U_s * y
    def fit_u_6(y, c, a, b):
        # 3-param generalized power law
        return c * (1.0 - y**a)**b + U_s * y

    # Fit calculations
    if is_constant:
        popt_R, popt_P, popt_U, popt_T = [0.0], [0.0], [0.0], [0.0]
        R_fit, P_fit, U_fit, T_fit = np.full_like(y_valid, R_s), np.full_like(y_valid, P_s), np.full_like(y_valid, U_s), np.full_like(y_valid, T_s)
        r2_R, r2_P, r2_U, r2_T = 1.0, 1.0, 1.0, 1.0
    else:
        try:
            popt_R, _ = curve_fit(power_law_R, y_valid, R_valid, p0=[0.3])
            R_fit = power_law_R(y_valid, *popt_R)
            r2_R = compute_r2(R_valid, R_fit)
        except Exception:
            popt_R, R_fit, r2_R = [np.nan], R_valid, 0.0
            
        try:
            popt_P, _ = curve_fit(power_law_P, y_valid, P_valid, p0=[1.0])
            P_fit = power_law_P(y_valid, *popt_P)
            r2_P = compute_r2(P_valid, P_fit)
        except Exception:
            popt_P, P_fit, r2_P = [np.nan], P_valid, 0.0
            
        try:
            popt_U, _ = curve_fit(power_law_U, y_valid, U_valid, p0=[1.0])
            U_fit = power_law_U(y_valid, *popt_U)
            r2_U = compute_r2(U_valid, U_fit)
        except Exception:
            popt_U, U_fit, r2_U = [np.nan], U_valid, 0.0
            
        try:
            popt_T, _ = curve_fit(power_law_T, y_valid, T_valid, p0=[0.3])
            T_fit = power_law_T(y_valid, *popt_T)
            r2_T = compute_r2(T_valid, T_fit)
        except Exception:
            popt_T, T_fit, r2_T = [np.nan], T_valid, 0.0
            
    # Fit U options
    fits_u = {}
    if is_constant:
        for i in range(1, 7):
            fits_u[i] = (None, np.full_like(y_valid, U_s), 1.0)
    else:
        # Fit 1
        try:
            popt, _ = curve_fit(fit_u_1, y_valid, U_valid, p0=[1.0], maxfev=10000)
            fits_u[1] = (popt, fit_u_1(y_valid, *popt), compute_r2(U_valid, fit_u_1(y_valid, *popt)))
        except Exception:
            fits_u[1] = (None, None, 0.0)
        # Fit 2
        try:
            popt, _ = curve_fit(fit_u_2, y_valid, U_valid, p0=[U_0, 1.0], maxfev=10000)
            fits_u[2] = (popt, fit_u_2(y_valid, *popt), compute_r2(U_valid, fit_u_2(y_valid, *popt)))
        except Exception:
            fits_u[2] = (None, None, 0.0)
        # Fit 3
        try:
            popt, _ = curve_fit(fit_u_3, y_valid, U_valid, p0=[U_0, U_0-U_s, 1.0], maxfev=10000)
            fits_u[3] = (popt, fit_u_3(y_valid, *popt), compute_r2(U_valid, fit_u_3(y_valid, *popt)))
        except Exception:
            fits_u[3] = (None, None, 0.0)
        # Fit 4
        try:
            popt, _ = curve_fit(fit_u_4, y_valid, U_valid, p0=[U_0, 1.0], maxfev=10000)
            fits_u[4] = (popt, fit_u_4(y_valid, *popt), compute_r2(U_valid, fit_u_4(y_valid, *popt)))
        except Exception:
            fits_u[4] = (None, None, 0.0)
        # Fit 5
        try:
            popt, _ = curve_fit(fit_u_5, y_valid, U_valid, p0=[U_0, 1.0, 0.3], maxfev=10000)
            fits_u[5] = (popt, fit_u_5(y_valid, *popt), compute_r2(U_valid, fit_u_5(y_valid, *popt)))
        except Exception:
            fits_u[5] = (None, None, 0.0)
        # Fit 6
        try:
            popt, _ = curve_fit(fit_u_6, y_valid, U_valid, p0=[U_0, 1.0, 1.0], maxfev=10000)
            fits_u[6] = (popt, fit_u_6(y_valid, *popt), compute_r2(U_valid, fit_u_6(y_valid, *popt)))
        except Exception:
            fits_u[6] = (None, None, 0.0)

    colors_u = {1: 'crimson', 2: 'darkorange', 3: 'forestgreen', 4: 'darkviolet', 5: 'deeppink', 6: 'teal'}

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel (0,0): Density R
    axes[0, 0].plot(y_valid, R_valid, 'b-', label='Numerical', lw=2)
    if is_constant:
        lbl_r = f'Fit: R = {R_s:.3f} (const)'
    else:
        lbl_r = f'Fit: R = {R_s:.3f}*y^{{-{popt_R[0]:.3f}}}\n(R² = {r2_R:.4f})'
    axes[0, 0].plot(y_valid, R_fit, 'r--', label=lbl_r, lw=1.5)
    axes[0, 0].set_ylabel(r"Density $R(y)$ [dimensionless]", fontsize=12)
    axes[0, 0].legend(loc='best', fontsize=9.0)
    
    # Panel (0,1): Pressure P
    axes[0, 1].plot(y_valid, P_valid, 'b-', label='Numerical', lw=2)
    if is_constant:
        lbl_p = f'Fit: P = {P_s:.3f} (const)'
    else:
        lbl_p = f'Fit: P = 1.0 - {1.0-P_s:.3f}*y^{{{popt_P[0]:.3f}}}\n(R² = {r2_P:.4f})'
    axes[0, 1].plot(y_valid, P_fit, 'r--', label=lbl_p, lw=1.5)
    axes[0, 1].set_ylabel(r"Pressure $P(y)$ [dimensionless]", fontsize=12)
    axes[0, 1].legend(loc='best', fontsize=9.0)
    
    # Panel (1,0): Velocity U
    axes[1, 0].plot(y_valid, U_valid, 'b-', label='Numerical', lw=2.5)
    selected_2x2_fits = [1, 2, 3]
    for i in selected_2x2_fits:
        popt, fit_val, r2 = fits_u[i]
        if fit_val is not None:
            if i == 1:
                lbl = f"1P U0-(U0-Us)*y^d (R²={r2:.4f})"
            elif i == 2:
                lbl = f"2P c-(c-Us)*y^d (R²={r2:.4f})"
            elif i == 3:
                lbl = f"3P c-a*y^b (R²={r2:.4f})"
            axes[1, 0].plot(y_valid, fit_val, colors_u[i], linestyle='--', label=lbl, lw=1.5)
    axes[1, 0].set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    axes[1, 0].legend(loc='best', fontsize=8.5)
    
    # Panel (1,1): Temperature T
    axes[1, 1].plot(y_valid, T_valid, 'b-', label='Numerical', lw=2)
    if is_constant:
        lbl_t = f'Fit: T = {T_s:.3f} (const)'
    else:
        lbl_t = f'Fit: T = {T_s:.3f}*y^{{{popt_T[0]:.3f}}}\n(R² = {r2_T:.4f})'
    axes[1, 1].plot(y_valid, T_fit, 'r--', label=lbl_t, lw=1.5)
    axes[1, 1].set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    axes[1, 1].legend(loc='best', fontsize=9.0)
    
    for ax in axes.flat:
        ax.set_xlabel(r"$y = \xi / \xi_s$", fontsize=12)
        ax.grid(True, alpha=0.3)
        
    fig.suptitle(f"Self-Similar Similarity Profiles & Curve-Fits\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(self_similar_path, dpi=200)
    plt.close(fig)
    print(f"Saved similarity fits grid to {self_similar_path}")

    # Plot standalone velocity fits
    fig_sa, ax_sa = plt.subplots(figsize=(12, 6.5))
    ax_sa.plot(y_valid, U_valid, 'b-', label='Numerical (Piston Shock Similarity Solver)', lw=3.0)
    
    for i in range(1, 7):
        popt, fit_val, r2 = fits_u[i]
        if fit_val is not None:
            if popt is None:
                lbl = f"Fit {i}: Constant value {U_s:.4f} (R² = {r2:.5f})"
            else:
                if i == 1:
                    lbl = f"Fit 1: 1-Param Power Law: U_0 - (U_0 - U_s) * y^d\n      d={popt[0]:.4f} (R² = {r2:.5f})"
                elif i == 2:
                    lbl = f"Fit 2: 2-Param Power Law: c - (c - U_s) * y^d\n      c={popt[0]:.3f}, d={popt[1]:.4f} (R² = {r2:.5f})"
                elif i == 3:
                    lbl = f"Fit 3: 3-Param Power Law: c - a * y^b\n      c={popt[0]:.3f}, a={popt[1]:.3f}, b={popt[2]:.4f} (R² = {r2:.5f})"
                elif i == 4:
                    lbl = f"Fit 4: 2-Param Rational: c*(1-y)/(1+d*y) + Us*y\n      c={popt[0]:.3f}, d={popt[1]:.3f} (R² = {r2:.5f})"
                elif i == 5:
                    lbl = f"Fit 5: 3-Param Singular: c*(1-y^a)*y^-b + Us*y\n      c={popt[0]:.3f}, a={popt[1]:.3f}, b={popt[2]:.4f} (R² = {r2:.5f})"
                elif i == 6:
                    lbl = f"Fit 6: 3-Param Gen Power: c*(1-y^a)^b + Us*y\n      c={popt[0]:.3f}, a={popt[1]:.3f}, b={popt[2]:.3f} (R² = {r2:.5f})"
                
            ax_sa.plot(y_valid, fit_val, colors_u[i], linestyle='--', label=lbl, lw=1.6)
            
    ax_sa.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=13)
    ax_sa.set_ylabel(r"Dimensionless Velocity $U(y)$", fontsize=13)
    ax_sa.set_title(f"Dimensionless Velocity Profile U(y) vs 6 Fitting Formulas\n{case_title}", fontsize=14, fontweight='bold')
    
    # Dynamic y-axis scaling
    ax_sa.set_ylim(U_valid.min() - 0.1*(U_valid.max() - U_valid.min()), U_valid.max() + 0.1*(U_valid.max() - U_valid.min()))
    ax_sa.grid(True, which='both', linestyle=':', alpha=0.5)
    ax_sa.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9.5, borderaxespad=0., frameon=True, shadow=True)
    
    plt.tight_layout()
    fig_sa.savefig(standalone_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sa)
    print(f"Saved standalone velocity fits to {standalone_path}")


# =============================================================================
# Custom 5-way animated GIF Evolution saver
# =============================================================================

def save_custom_evolution_gif(
    history,
    case,
    menahem_ref,
    gif_path: str,
    fps: int = 12,
    stride: int = 1,
    subtitle: str | None = None,
):
    """Generate animated 5-way GIF comparing Simulation vs Menahem PistonShock reference."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    p_scale, u_scale, e_scale = 1e12, 1e5, 1e9
    gamma = float(case.r) + 1.0
    
    # Construct 5 vertical panels
    fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    k0 = 0
    m0 = history.m[k0]
    
    # 1) Simulation lines (blue solid)
    sim_lines = []
    sim_lines.append(axes[0].plot(m0, history.rho[k0], lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[1].plot(m0, history.p[k0] / p_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[2].plot(m0, history.u[k0] / u_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[3].plot(m0, history.e[k0] / e_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[4].plot(m0, history.T[k0] if hasattr(history, 'T') else np.zeros_like(m0), lw=2, label="simulation", color="blue")[0])
    
    # 2) Menahem PistonShock lines (magenta/dashed)
    men_lines = []
    men_lines.append(axes[0].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[1].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[2].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[3].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[4].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        
    x_mass = r"Mass coordinate $m$ [g/cm²]"
    axes[0].set_ylabel(r"$\rho$ [g/cm³]")
    axes[1].set_ylabel(r"$P$ [MBar]")
    axes[2].set_ylabel(r"$u$ [km/s]")
    axes[3].set_ylabel(r"$e_{\mathrm{mat}}$ [$10^9$ erg/g]")
    axes[4].set_ylabel(r"$T_{\mathrm{mat}}$ [K]")
    axes[4].set_xlabel(x_mass)
    
    title = fig.suptitle("", fontweight="medium")
    frame_ids = np.arange(0, len(history.t), stride)
    
    def init():
        return sim_lines + men_lines
        
    def update(frame_idx):
        k = int(frame_ids[frame_idx])
        mk = history.m[k]
        t = history.t[k]
        sim_lines[0].set_data(mk, history.rho[k])
        sim_lines[1].set_data(mk, history.p[k] / p_scale)
        sim_lines[2].set_data(mk, history.u[k] / u_scale)
        sim_lines[3].set_data(mk, history.e[k] / e_scale)
        
        # Temp from ideal gas relation: p / (rho * r)
        with np.errstate(divide='ignore', invalid='ignore'):
            Tk_sim = np.where(history.rho[k] > 0, history.p[k] / (history.rho[k] * float(case.r)), 0.0)
        sim_lines[4].set_data(mk, Tk_sim)
        
        if menahem_ref is not None:
            men_idx = int(np.argmin(np.abs(menahem_ref.times - t)))
            mm = menahem_ref.m[men_idx]
            men_lines[0].set_data(mm, menahem_ref.rho[men_idx])
            men_lines[1].set_data(mm, menahem_ref.p[men_idx] / p_scale)
            men_lines[2].set_data(mm, menahem_ref.u[men_idx] / u_scale)
            men_lines[3].set_data(mm, menahem_ref.e[men_idx] / e_scale)
            
            # Temp from ideal gas relation
            with np.errstate(divide='ignore', invalid='ignore'):
                Tk_men = np.where(menahem_ref.rho[men_idx] > 0, menahem_ref.p[men_idx] / (menahem_ref.rho[men_idx] * float(case.r)), 0.0)
            men_lines[4].set_data(mm, Tk_men)
            
        case_title = case.title if hasattr(case, "title") and case.title else "Simulation"
        header = f"{case_title}\n{subtitle}" if subtitle else case_title
        title.set_text(
            f"{header}\n"
            f"$P(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0_Barye:.3e},\\; \\tau={case.tau},\\; t={t*1e9:.3f}$ ns"
        )
        
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        return sim_lines + men_lines
        
    anim = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=False)
    
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved custom GIF to {gif_path}")


class ReferenceContainer:
    def __init__(self, times, m, x, rho, p, u, e):
        self.times = times
        self.m = m
        self.x = x
        self.rho = rho
        self.p = p
        self.u = u
        self.e = e


# =============================================================================
# Main Orchestration Loop
# =============================================================================

def run_simulation_and_references(preset_name: str, case_label: str):
    """Run full simulation and build PistonShock reference solver, or load from cache."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_label}_cache.pkl"
    
    case, config = get_preset(preset_name)
    config = replace(config, N=400)
    
    if cache_path.exists():
        print(f"Loading cached simulation and reference data from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return case, data["history"], data["menahem_ref"], data["solver"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running simulation...")
            
    # Run simulation
    print("Running simulation...")
    _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)
    
    # Run PistonShock reference solver
    print("Building PistonShock reference...")
    times_sec = np.array(history.t)
    times_sec = times_sec[times_sec > 0.0]
    
    kwargs = _shock_kwargs_from_case(case)
    # Set general omega if preset case specifies one, defaults to 0.0
    kwargs["omega"] = float(getattr(case, "omega", 0.0))
    solver = PistonShock(**kwargs)
    
    # Build a structure similar to HydroSimData
    mass = _build_mass_grid(case, num_cells=400)
    
    m_list, x_list = [], []
    rho_list, p_list, u_list, e_list = [], [], [], []
    for t in times_sec:
        sol = solver.solve(mass=mass, time=float(t))
        m_list.append(mass.copy())
        x_list.append(np.asarray(sol["position"], dtype=float))
        rho_list.append(np.asarray(sol["density"], dtype=float))
        p_list.append(np.asarray(sol["pressure"], dtype=float))
        u_list.append(np.asarray(sol["velocity"], dtype=float))
        e_list.append(np.asarray(sol["sie"], dtype=float))
        
    menahem_ref = ReferenceContainer(times_sec, m_list, x_list, rho_list, p_list, u_list, e_list)
    
    cache_data = {
        "case": case,
        "history": history,
        "menahem_ref": menahem_ref,
        "solver": solver,
    }
    
    print(f"Saving simulation and reference data cache to {cache_path}...")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Failed to save cache: {e}")
        
    return case, history, menahem_ref, solver


def generate_verification_plots(
    history,
    case,
    menahem_ref,
    solver,
    case_label: str,
    case_title: str,
):
    """Generate all verification comparisons, plotting, fitting, and animated GIF."""
    out_dir = Path("results/ictt")
    xt_dir = out_dir / "xt"
    mh_dir = out_dir / "material_hydro"
    ss_dir = out_dir / "self_similar"
    ev_dir = out_dir / "evolution"
    
    for d in [xt_dir, mh_dir, ss_dir, ev_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    xt_path = str(xt_dir / f"{case_label}_xt.png")
    material_hydro_path = str(mh_dir / f"{case_label}_material_hydro.png")
    self_similar_path = str(ss_dir / f"{case_label}_self_similar.png")
    standalone_path = str(ss_dir / f"{case_label}_velocity_fits_standalone.png")
    gif_path = str(ev_dir / f"{case_label}_evolution.gif")
    
    # 1) Generate space-time trajectories and fronts
    plot_xt_trajectories(history, case, xt_path, case_title, solver=solver)
    
    # 2) Plot physical material hydrodynamics comparison
    plot_material_hydro_profiles(history, menahem_ref, material_hydro_path, case_title)
    
    # 3) Solve similarity ODEs, fit, and plot self-similar profiles
    plot_and_fit_self_similar(case, self_similar_path, standalone_path, case_title)
    
    # 4) Generate animated evolution GIF
    print(f"Saving animated custom 5-way evolution GIF to {gif_path}...")
    save_custom_evolution_gif(
        history=history,
        case=case,
        menahem_ref=menahem_ref,
        gif_path=gif_path,
        fps=12,
        stride=max(1, len(history.t) // 60),
        subtitle="Simulation vs Menahem PistonShock",
    )


def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run full verification comparison pipeline for a given preset."""
    print("=" * 80)
    print(f"PROCESSING PRESET: {preset_name} -> {case_label}")
    print("=" * 80)
    
    cache_path = Path("results/ictt/cache") / f"{case_label}_cache.pkl"
    if not cache_path.exists():
        print("Pickle file doesn't exist, running simulation...")
        case, history, menahem_ref, solver = run_simulation_and_references(
            preset_name, case_label
        )
    else:
        print("Pickle file exists, loading simulation...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        case = data.get("case")
        if case is None:
            case, _ = get_preset(preset_name)
        history = data["history"]
        menahem_ref = data["menahem_ref"]
        solver = data["solver"]
        
    generate_verification_plots(
        history=history,
        case=case,
        menahem_ref=menahem_ref,
        solver=solver,
        case_label=case_label,
        case_title=case_title,
    )
    print(f"Preset {preset_name} processed successfully.")


def main():
    # Process constant drive piston shock (tau=0) and power-law drive piston shock (tau=-0.45)
    run_preset_workflow(
        PRESET_CONSTANT_PRESSURE,
        "constant_pressure_drive",
        "Constant Pressure Drive (tau=0)"
    )
    run_preset_workflow(
        PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE,
        "power_law_pressure_drive",
        "Power-Law Pressure Drive (tau=-0.45)"
    )
    print("\nAll custom simulations, PistonShock comparisons, plotting, fitting, and exports completed successfully!")


if __name__ == "__main__":
    main()

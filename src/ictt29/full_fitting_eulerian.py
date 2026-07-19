# ictt29/full_fitting_eulerian.py
"""
Unified patched ablation and shock fitting verification script in Eulerian coordinates.

Combines:
1. 1D Rad-Hydro Simulation (cell boundaries mapped to cell-center positions).
2. AblationSolver patched reference solver (heat wave + piston shock) in Eulerian frame.
3. Patched self-similar analytical fits dynamically mapped to Eulerian coordinate.

Produces three comparison plots:
1. fig_8_patched_fit_comparison_eulerian.png: individual subsonic (solid) and shock (dashed) overlays in Eulerian coordinates.
2. fig_8_fully_patched_comparison_eulerian.png: fully patched, seamless profiles in Eulerian coordinates.
3. fig_8_front_trajectories_eulerian.png: time-dependent trajectories of the ablation boundary, shock piston, and shock front.
"""
from __future__ import annotations

import os
import sys
sys.setrecursionlimit(200000)
import pickle
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.integrate

# Ensure proper package and solver imports
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_REPO_ROOT = _REPO_PARENT / "project3_code"
_MENAHEM_DIR = _REPO_ROOT / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

_ICTT29_DIR = Path(__file__).resolve().parent
if str(_ICTT29_DIR) not in sys.path:
    sys.path.insert(0, str(_ICTT29_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
)
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _ablation_kwargs_from_case,
    _build_mass_grid,
)
from project3_code.menahem_new.ablation_solver_og import AblationSolver

# Import fitting and dimensional scaling functions from shock and sub fitting scripts
from sub_fitting import perform_subsonic_fitting, calculate_dimensional_fits as calculate_dimensional_fits_sub, fit_by_params as sub_fit_by_params
from shock_fitting import perform_shock_fitting, calculate_dimensional_fits as calculate_dimensional_fits_shock, fit_by_params as shock_fit_by_params

# Monkeypatch scipy.integrate.simps to scipy.integrate.simpson for modern Scipy versions
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

# Monkeypatch numpy.trapz to scipy.integrate.trapezoid for modern Numpy 2.x versions
if not hasattr(np, "trapz"):
    if hasattr(scipy.integrate, "trapezoid"):
        np.trapz = scipy.integrate.trapezoid
    else:
        np.trapz = scipy.integrate.trapz

USE_CACHE = True  # Set to True to use pre-saved pickle files, False to run again


# =============================================================================
# Helper functions for shock detection and rolling average
# (imported from the shared utility module)
# =============================================================================

from sim_front_utils import (  # noqa: E402
    find_shock_front,
    _rolling_mean,
    detect_sim_ablation_boundary,
    detect_sim_ablation_front,
    detect_sim_shock_front_trajectory,
    compute_fit_front_trajectories,
)


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
# Eulerian Coordinate Fit Calculations
# =============================================================================

def calculate_patched_eulerian_positions_fit(mass_grid, t_actual, ablation_solver, sub_params, shock_params):
    """
    Computes the absolute Eulerian coordinate x (in cm) for the patched fits on the given mass_grid at t_actual.
    """
    m_f = ablation_solver.heat_solver.ablated_mass(time=t_actual)
    m_s = ablation_solver.shock_solver.shocked_mass(time=t_actual)
    
    hs = ablation_solver.heat_solver
    ss = ablation_solver.shock_solver
    
    # Calculate fit's shock piston position (x_p_fit) and ablation front position (x_af_fit)
    pos_scale = ss._position_temporal_factor(time=t_actual)
    q1 = 1.0 - ss.omega
    q2 = (2.0 - ss.omega) / (ss.tau + 2.0)
    
    x_p_fit = pos_scale * q2 * ss.U0
    
    # Ablation front is at y_mf in shock coordinates
    xsi_mf = m_f * ss.xsi_over_m(time=t_actual)
    y_mf = xsi_mf / ss.xsi_s
    
    # Evaluate shock fit at y_mf
    _, _, U_fit_sh_mf, rho_fit_sh_mf = shock_fit_by_params(np.array([y_mf]), shock_params)
    V_fit_sh_mf = 1.0 / rho_fit_sh_mf[0]
    x_af_fit = pos_scale * (q1 * xsi_mf * V_fit_sh_mf + q2 * U_fit_sh_mf[0])
    
    # Subsonic position coefficient
    C_pos = hs._position_temporal_factor(time=t_actual)
    
    x_fit = np.zeros_like(mass_grid)
    
    sub_mask = mass_grid <= m_f
    shock_mask = (mass_grid > m_f) & (mass_grid <= m_s)
    unshocked_mask = mass_grid > m_s
    
    # 1) Subsonic region
    if np.any(sub_mask):
        m_sub = mass_grid[sub_mask]
        xsi_sub = m_sub * hs.xsi_over_m(time=t_actual)
        y_sub = xsi_sub / hs.xsi_f
        _, _, U_fit_sub, rho_fit_sub = sub_fit_by_params(y_sub, sub_params)
        V_fit_sub = 1.0 / rho_fit_sub
        
        # Guard against front singularity
        fac_sub = U_fit_sub - hs.c * xsi_sub * V_fit_sub
        fac_sub = np.where((y_sub >= 1.0 - 1e-6) | (fac_sub > 0.0), 0.0, fac_sub)
        
        x_fit[sub_mask] = x_af_fit + C_pos * fac_sub
        
    # 2) Shocked region
    if np.any(shock_mask):
        m_sh = mass_grid[shock_mask]
        y_sh = m_sh / m_s
        _, _, U_fit_sh, rho_fit_sh = shock_fit_by_params(y_sh, shock_params)
        V_fit_sh = 1.0 / rho_fit_sh
        x_fit[shock_mask] = pos_scale * (q1 * (y_sh * ss.xsi_s) * V_fit_sh + q2 * U_fit_sh)
        
    # 3) Unshocked region
    if np.any(unshocked_mask):
        m_un = mass_grid[unshocked_mask]
        x_fit[unshocked_mask] = ss.initial_position(mass=m_un)
        
    return x_fit


# =============================================================================
# Data Loading and Management
# =============================================================================

from data_loader import get_sim_history, get_ablation_solver

def get_data(preset_name, case_label, N=None):
    """Run full simulation and build AblationSolver reference solver, or load from cache."""
    case, history = get_sim_history(preset_name, case_label, N=N)
    ablation_solver = get_ablation_solver(case, case_label)
    return case, history, ablation_solver


def calculate_patched_dimensional_fits(mass_grid, t_actual, ablation_solver, sub_params, shock_params):
    """Constructs fully patched, seamless analytical physical profiles."""
    m_f = ablation_solver.heat_solver.ablated_mass(time=t_actual)

    sub_mask = mass_grid <= m_f
    shock_mask = mass_grid > m_f

    n = len(mass_grid)
    rho = np.zeros(n)
    p = np.zeros(n)
    u = np.zeros(n)
    T = np.zeros(n)

    if np.any(sub_mask):
        sub_fits = calculate_dimensional_fits_sub(mass_grid[sub_mask], t_actual, ablation_solver.heat_solver, sub_params)
        rho[sub_mask] = sub_fits["density"]
        p[sub_mask] = sub_fits["pressure"]
        u[sub_mask] = sub_fits["velocity"]
        T[sub_mask] = sub_fits["temperature"]

    if np.any(shock_mask):
        shock_fits = calculate_dimensional_fits_shock(mass_grid[shock_mask], t_actual, ablation_solver.shock_solver, shock_params)
        rho[shock_mask] = shock_fits["density"]
        p[shock_mask] = shock_fits["pressure"]
        u[shock_mask] = shock_fits["velocity"]
        T[shock_mask] = shock_fits["temperature"]

    return {"density": rho, "pressure": p, "velocity": u, "temperature": T}


# =============================================================================
# Plotting - Profiles vs Eulerian Coordinate x
# =============================================================================

def plot_patched_dimensional_fit_comparison_eulerian(
    history,
    ablation_solver,
    sub_params,
    shock_params,
    case,
    plot_path,
    case_title,
):
    """
    Plots unified 2x2 CGS overlays vs Eulerian Coordinate x [um] showing individual subsonic and shock region profiles.
    Overlays vertical lines for front positions of Simulation, Solver, and Fits.
    """
    print(f"Generating physical patched profiles comparison in Eulerian for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_max = max(history.t)
    target_times = [t_max]
    sim_colors = ["#1a5fb4"]

    ax_rho = axes[0, 0]
    ax_p = axes[0, 1]
    ax_u = axes[1, 0]
    ax_T = axes[1, 1]

    p_scale = 1e12
    u_scale = 1e5
    T_scale = 1.160451812e6

    hs = ablation_solver.heat_solver
    ss = ablation_solver.shock_solver
    q1 = 1.0 - ss.omega
    q2 = (2.0 - ss.omega) / (ss.tau + 2.0)

    for i, (t_target, sim_color) in enumerate(zip(target_times, sim_colors)):
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        x_sim_raw = np.asarray(history.x[idx_sim], dtype=float)
        x_sim_center_um = x_sim_raw * 1e4
        
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]

        sim_rho = history.rho[idx_sim]
        sim_p = history.p[idx_sim] / p_scale
        sim_u = history.u[idx_sim] / u_scale
        sim_T = history.T[idx_sim] / T_scale

        # Determine fronts at this actual time
        m_f = hs.ablated_mass(time=t_actual)
        m_s = ss.shocked_mass(time=t_actual)

        # Define grids (natively sized 1000) with quadratic clustering near m=0
        mass_sub = 1e-12 + (m_f - 1e-12) * (np.linspace(0.0, 1.0, 1000) ** 2.0)
        mass_shock = np.linspace(1e-12, m_sim[-1], 1000)

        # 2) Solver exact subsonic & shock
        # Solve patched exact solver to get heat_position and front locations
        sol_exact = ablation_solver.solve(mass=mass_shock, time=t_actual)
        heat_position = sol_exact["heat_position"]

        sol_sub_absolute = hs.solve(mass=mass_sub, time=t_actual)
        sol_sub_absolute["position"] += heat_position
        sol_sub = sol_sub_absolute
        
        exact_sub_rho = sol_sub_absolute["density"]
        exact_sub_p = sol_sub_absolute["pressure"] / p_scale
        exact_sub_u = sol_sub_absolute["velocity"] / u_scale
        exact_sub_T = sol_sub_absolute["temperature"] / T_scale
        exact_sub_x_um = sol_sub_absolute["position"] * 1e4

        x_b_sol = (sol_sub_absolute["boundary_position"] + heat_position) * 1e4

        # Prepend/heal subsonic solver arrays to remove ODE integration cutoff near the boundary
        valid_idx = np.flatnonzero(np.isfinite(exact_sub_rho) & (exact_sub_rho > 0.0))
        first_valid_i = valid_idx[0] if len(valid_idx) > 0 else 0

        # Boundary values at m = 0
        v_b = hs.V0 * (hs.A**hs.a1) * (hs.B**hs.b1) * (t_actual**hs.c1)
        rho_b = 1.0 / v_b
        p_b = hs.P0 * (hs.A**hs.a3) * (hs.B**hs.b3) * (t_actual**hs.c3)
        u_b = hs.U0 * (hs.A**hs.a2) * (hs.B**hs.b2) * (t_actual**hs.c2)
        T_b = ((p_b * v_b**(1.0 - hs.mu)) / (hs.r * hs.f))**(1.0 / hs.beta)

        exact_sub_x_um = np.concatenate([[x_b_sol], exact_sub_x_um[first_valid_i:]])
        exact_sub_rho = np.concatenate([[rho_b], exact_sub_rho[first_valid_i:]])
        exact_sub_p = np.concatenate([[p_b / p_scale], exact_sub_p[first_valid_i:]])
        exact_sub_u = np.concatenate([[u_b / u_scale], exact_sub_u[first_valid_i:]])
        exact_sub_T = np.concatenate([[T_b / T_scale], exact_sub_T[first_valid_i:]])

        sol_shock_absolute = ss.solve(mass=mass_shock, time=t_actual)
        exact_shock_rho = sol_shock_absolute["density"]
        exact_shock_p = sol_shock_absolute["pressure"] / p_scale
        exact_shock_u = sol_shock_absolute["velocity"] / u_scale
        exact_shock_T_kelvin = ((sol_shock_absolute["pressure"] * sol_shock_absolute["density"]**(ablation_solver.mu_shock - 1.0)) / (ss.r * ablation_solver.f_shock))**(1.0 / ablation_solver.beta_shock)
        exact_shock_T = exact_shock_T_kelvin / T_scale
        exact_shock_x_um = sol_shock_absolute["position"] * 1e4

        # 3) Analytical subsonic & shock fits
        fits_sub = calculate_dimensional_fits_sub(mass_sub, t_actual, hs, sub_params)
        fit_sub_rho = fits_sub["density"]
        fit_sub_p = fits_sub["pressure"] / p_scale
        fit_sub_u = fits_sub["velocity"] / u_scale
        fit_sub_T = fits_sub["temperature"] / T_scale
        
        fit_sub_x_um = calculate_patched_eulerian_positions_fit(mass_sub, t_actual, ablation_solver, sub_params, shock_params) * 1e4

        fits_shock = calculate_dimensional_fits_shock(mass_shock, t_actual, ss, shock_params)
        fit_shock_rho = fits_shock["density"]
        fit_shock_p = fits_shock["pressure"] / p_scale
        fit_shock_u = fits_shock["velocity"] / u_scale
        fit_shock_T = fits_shock["temperature"] / T_scale
        
        fit_shock_x_um = calculate_patched_eulerian_positions_fit(mass_shock, t_actual, ablation_solver, sub_params, shock_params) * 1e4

        show_label = i == 0

        # Plot Simulation (entire domain) - Blue
        ax_rho.plot(x_sim_center_um, sim_rho, '-', color='#1a5fb4', lw=2.2)
        ax_p.plot(x_sim_center_um, sim_p, '-', color='#1a5fb4', lw=2.2)
        ax_u.plot(x_sim_center_um, sim_u, '-', color='#1a5fb4', lw=2.2)
        ax_T.plot(x_sim_center_um, sim_T, '-', color='#1a5fb4', lw=2.2)

        # Plot Subsonic exact solutions (solid black)
        ax_rho.plot(exact_sub_x_um, exact_sub_rho, '-', color='#333333', lw=2.0)
        ax_p.plot(exact_sub_x_um, exact_sub_p, '-', color='#333333', lw=2.0)
        ax_u.plot(exact_sub_x_um, exact_sub_u, '-', color='#333333', lw=2.0)
        ax_T.plot(exact_sub_x_um, exact_sub_T, '-', color='#333333', lw=2.0)

        # Plot Subsonic fits (dotted green)
        ax_rho.plot(fit_sub_x_um, fit_sub_rho, ':', color='#26a269', lw=2.0)
        ax_p.plot(fit_sub_x_um, fit_sub_p, ':', color='#26a269', lw=2.0)
        ax_u.plot(fit_sub_x_um, fit_sub_u, ':', color='#26a269', lw=2.0)
        ax_T.plot(fit_sub_x_um, fit_sub_T, ':', color='#26a269', lw=2.0)

        # Plot Shock exact solutions (dashed black)
        ax_rho.plot(exact_shock_x_um, exact_shock_rho, '--', color='#333333', lw=1.8)
        ax_p.plot(exact_shock_x_um, exact_shock_p, '--', color='#333333', lw=1.8)
        ax_u.plot(exact_shock_x_um, exact_shock_u, '--', color='#333333', lw=1.8)
        ax_T.plot(exact_shock_x_um, exact_shock_T, '--', color='#333333', lw=1.8)

        # Plot Shock fits (dashed orange)
        ax_rho.plot(fit_shock_x_um, fit_shock_rho, '-.', color='#e67e22', lw=1.8)
        ax_p.plot(fit_shock_x_um, fit_shock_p, '-.', color='#e67e22', lw=1.8)
        ax_u.plot(fit_shock_x_um, fit_shock_u, '-.', color='#e67e22', lw=1.8)
        ax_T.plot(fit_shock_x_um, fit_shock_T, '-.', color='#e67e22', lw=1.8)

        # Solver fronts
        x_b_sol = (sol_sub_absolute["boundary_position"] + heat_position) * 1e4
        x_af_sol = heat_position * 1e4
        x_s_sol = sol_shock_absolute["shock_position"] * 1e4
        
        # Fit fronts
        pos_scale = ss._position_temporal_factor(time=t_actual)
        m_f = hs.ablated_mass(time=t_actual)
        xsi_mf = m_f * ss.xsi_over_m(time=t_actual)
        y_mf = xsi_mf / ss.xsi_s
        _, _, U_fit_sh_mf, rho_fit_sh_mf = shock_fit_by_params(np.array([y_mf]), shock_params)
        V_fit_sh_mf = 1.0 / rho_fit_sh_mf[0]
        x_af_fit = pos_scale * (q1 * xsi_mf * V_fit_sh_mf + q2 * U_fit_sh_mf[0])
        
        x_b_fit = (hs.boundary_position(time=t_actual) + x_af_fit) * 1e4
        x_s_fit = ss.shock_position(time=t_actual) * 1e4

        # Simulation fronts
        dx_sim = x_sim_center_um[1] - x_sim_center_um[0]
        x_b_sim = x_sim_center_um[0] - 0.5 * dx_sim
        # Detect simulation shock (Hugoniot_threshold defaults to 0.9 via sim_front_utils)
        rhok_smooth = _rolling_mean(sim_rho, 5)
        ishock, _ = find_shock_front(rhok_smooth, m_sim, rho_unshocked=float(case.rho0), gamma=float(case.r) + 1.0)
        x_s_sim = x_sim_center_um[ishock] if ishock >= 1 else np.nan

        # Draw vertical lines for the fronts on all subplots with matching styles
        for ax in [ax_rho, ax_p, ax_u, ax_T]:
            # Ablation boundary
            ax.axvline(x=x_b_sim, color='#1a5fb4', linestyle='-', lw=1.2, alpha=0.6)
            ax.axvline(x=x_b_sol, color='#333333', linestyle='--', lw=1.2, alpha=0.6)
            ax.axvline(x=x_b_fit, color='#e67e22', linestyle=':', lw=1.4, alpha=0.6)
            
            # Ablation front / shock piston
            ax.axvline(x=x_af_sol, color='#333333', linestyle='--', lw=1.2, alpha=0.6)
            ax.axvline(x=x_af_fit, color='#e67e22', linestyle=':', lw=1.4, alpha=0.6)
            
            # Shock front
            if not np.isnan(x_s_sim):
                ax.axvline(x=x_s_sim, color='#1a5fb4', linestyle='-', lw=1.2, alpha=0.6)
            ax.axvline(x=x_s_sol, color='#333333', linestyle='--', lw=1.2, alpha=0.6)
            ax.axvline(x=x_s_fit, color='#e67e22', linestyle=':', lw=1.4, alpha=0.6)

    # Build clear style/front legend entries
    legend_handles = [
        Line2D([0], [0], color='#1a5fb4', lw=2.5, linestyle='-', label='Simulation (2.0 ns)'),
        Line2D([0], [0], color='#333333', lw=2.0, linestyle='-', label='Subsonic Solver'),
        Line2D([0], [0], color='#26a269', lw=2.0, linestyle=':', label='Subsonic Fit'),
        Line2D([0], [0], color='#333333', lw=1.8, linestyle='--', label='Shock Solver'),
        Line2D([0], [0], color='#e67e22', lw=1.8, linestyle='-.', label='Shock Fit'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle='-', label='Simulation Fronts'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle='--', label='Solver Fronts'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle=':', label='Fit Fronts'),
    ]
    ax_rho.legend(handles=legend_handles, loc="lower left", fontsize=11)

    # Set y limits based on simulation bounds
    all_sim_rho = []
    all_sim_p = []
    all_sim_u = []
    all_sim_T = []
    for t_target in target_times:
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        all_sim_rho.extend(history.rho[idx_sim])
        all_sim_p.extend(history.p[idx_sim] / p_scale)
        all_sim_u.extend(history.u[idx_sim] / u_scale)
        all_sim_T.extend(history.T[idx_sim] / T_scale)

    max_sim_rho = np.max(all_sim_rho)
    max_sim_p = np.max(all_sim_p)
    min_sim_u = np.min(all_sim_u)
    max_sim_u = np.max(all_sim_u)
    max_sim_T = np.max(all_sim_T)

    ax_rho.set_ylim(-5.0, max_sim_rho * 1.15)
    ax_p.set_ylim(-0.05 * max_sim_p, max_sim_p * 1.15)
    ax_u.set_ylim(min_sim_u * 1.15 - 5.0, max_sim_u + 10.0)
    ax_T.set_ylim(-0.05 * max_sim_T, max_sim_T * 1.15)

    # Styling
    titles = ["Density", "Pressure", "Velocity", "Temperature"]
    math_labels = [r"$\rho$ [g/cm³]", r"$P$ [MBar]", r"$u$ [km/s]", r"$T$ [HeV]"]
    for j, ax in enumerate([ax_rho, ax_p, ax_u, ax_T]):
        ax.grid(True, alpha=0.3)
        ax.set_title(titles[j], fontsize=14, fontweight='bold')
        ax.set_ylabel(math_labels[j], fontsize=13)
        ax.set_xlabel(r"Eulerian Position Coordinate $x$ [$\mu$m]", fontsize=13)
        ax.tick_params(labelsize=11)

    plt.suptitle(f"Unified Patched Ablation & Shock Verification in Eulerian Coordinate\n{case_title}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved region overlays plot in Eulerian to {plot_path}")
    plt.close(fig)


def plot_fully_patched_comparison_eulerian(
    history,
    ablation_solver,
    sub_params,
    shock_params,
    case,
    plot_path,
    case_title,
    compare_imc=False,
):
    """
    Plots a unified 2x2 comparison showing the fully patched (seamless) profiles vs Eulerian coordinate x.
    Compares Rad-Hydro Simulation, AblationSolver (exact patched), and fully patched Fits or IMC data.
    """
    print(f"Generating physical fully patched comparison in Eulerian for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_max = max(history.t)
    target_times = [t_max]
    sim_colors = ["#1a5fb4"]

    ax_rho = axes[0, 0]
    ax_p = axes[0, 1]
    ax_u = axes[1, 0]
    ax_T = axes[1, 1]

    p_scale = 1e12
    u_scale = 1e5
    T_scale = 1.160451812e6

    hs = ablation_solver.heat_solver
    ss = ablation_solver.shock_solver
    q1 = 1.0 - ss.omega
    q2 = (2.0 - ss.omega) / (ss.tau + 2.0)

    for i, (t_target, sim_color) in enumerate(zip(target_times, sim_colors)):
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        x_sim_raw = np.asarray(history.x[idx_sim], dtype=float)
        x_sim_center_um = x_sim_raw * 1e4
        
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]

        sim_rho = history.rho[idx_sim]
        sim_p = history.p[idx_sim] / p_scale
        sim_u = history.u[idx_sim] / u_scale
        sim_T = history.T[idx_sim] / T_scale

        # Define grid quadratically clustered near m=0
        mass_solver = 1e-12 + (m_sim[-1] - 1e-12) * (np.linspace(0.0, 1.0, 1000) ** 2.0)

        # Solve fully patched exact profiles from AblationSolver
        sol_exact = ablation_solver.solve(mass=mass_solver, time=t_actual)
        exact_rho = sol_exact["density"]
        exact_p = sol_exact["pressure"] / p_scale
        exact_u = sol_exact["velocity"] / u_scale
        exact_T = sol_exact["temperature"] / T_scale
        exact_x_um = sol_exact["position"] * 1e4

        x_b_sol = (sol_exact["boundary_position"] + sol_exact["heat_position"]) * 1e4

        # Prepend/heal solver arrays to remove ODE integration cutoff near the boundary
        valid_idx = np.flatnonzero(np.isfinite(exact_rho) & (exact_rho > 0.0))
        first_valid_i = valid_idx[0] if len(valid_idx) > 0 else 0

        # Boundary values at m = 0
        v_b = hs.V0 * (hs.A**hs.a1) * (hs.B**hs.b1) * (t_actual**hs.c1)
        rho_b = 1.0 / v_b
        p_b = hs.P0 * (hs.A**hs.a3) * (hs.B**hs.b3) * (t_actual**hs.c3)
        u_b = hs.U0 * (hs.A**hs.a2) * (hs.B**hs.b2) * (t_actual**hs.c2)
        T_b = ((p_b * v_b**(1.0 - hs.mu)) / (hs.r * hs.f))**(1.0 / hs.beta)

        exact_x_um = np.concatenate([[x_b_sol], exact_x_um[first_valid_i:]])
        exact_rho = np.concatenate([[rho_b], exact_rho[first_valid_i:]])
        exact_p = np.concatenate([[p_b / p_scale], exact_p[first_valid_i:]])
        exact_u = np.concatenate([[u_b / u_scale], exact_u[first_valid_i:]])
        exact_T = np.concatenate([[T_b / T_scale], exact_T[first_valid_i:]])

        # Solve fully patched Fits profiles
        fits = calculate_patched_dimensional_fits(mass_solver, t_actual, ablation_solver, sub_params, shock_params)
        fit_rho = fits["density"]
        fit_p = fits["pressure"] / p_scale
        fit_u = fits["velocity"] / u_scale
        fit_T = fits["temperature"] / T_scale
        
        fit_x_um = calculate_patched_eulerian_positions_fit(mass_solver, t_actual, ablation_solver, sub_params, shock_params) * 1e4

        show_label = i == 0

        # Plot Simulation (entire domain) - Blue
        ax_rho.plot(x_sim_center_um, sim_rho, '-', color='#1a5fb4', lw=2.2)
        ax_p.plot(x_sim_center_um, sim_p, '-', color='#1a5fb4', lw=2.2)
        ax_u.plot(x_sim_center_um, sim_u, '-', color='#1a5fb4', lw=2.2)
        ax_T.plot(x_sim_center_um, sim_T, '-', color='#1a5fb4', lw=2.2)

        # Plot Exact patched solver (solid black)
        ax_rho.plot(exact_x_um, exact_rho, '-', color='#333333', lw=2.0)
        ax_p.plot(exact_x_um, exact_p, '-', color='#333333', lw=2.0)
        ax_u.plot(exact_x_um, exact_u, '-', color='#333333', lw=2.0)
        ax_T.plot(exact_x_um, exact_T, '-', color='#333333', lw=2.0)

        if compare_imc:
            # IMC replaces the fits. We only plot it for the final target time step (t_actual ~ 2.0 ns)
            if i == len(target_times) - 1:
                from project3_code.shteinberg_comparison.shteinberg_comparison import extract_steinberg_IMC_for_num_cells
                num_cells = len(m_sim)
                st_imc = extract_steinberg_IMC_for_num_cells(num_cells)
                if st_imc is None:
                    st_imc = extract_steinberg_IMC_for_num_cells(1024)
                
                if st_imc is not None:
                    imc_x_um = st_imc["x"] * 1e4
                    imc_rho = st_imc["density"]
                    imc_p = st_imc["pressure"] / p_scale
                    imc_u = st_imc["velocity_x"] / u_scale
                    imc_T = st_imc["temperature"] / T_scale

                    ax_rho.plot(imc_x_um, imc_rho, '--', color='forestgreen', lw=1.8)
                    ax_p.plot(imc_x_um, imc_p, '--', color='forestgreen', lw=1.8)
                    ax_u.plot(imc_x_um, imc_u, '--', color='forestgreen', lw=1.8)
                    ax_T.plot(imc_x_um, imc_T, '--', color='forestgreen', lw=1.8)
        else:
            # Plot Patched fits (dotted orange)
            ax_rho.plot(fit_x_um, fit_rho, ':', color='#e67e22', lw=2.0)
            ax_p.plot(fit_x_um, fit_p, ':', color='#e67e22', lw=2.0)
            ax_u.plot(fit_x_um, fit_u, ':', color='#e67e22', lw=2.0)
            ax_T.plot(fit_x_um, fit_T, ':', color='#e67e22', lw=2.0)

        # Solver fronts (ablation front and shock front)
        x_af_sol_um = sol_exact["heat_position"] * 1e4
        x_sf_sol_um = sol_exact["shock_position"] * 1e4

        # Draw vertical dashed lines for ablation and shock fronts from the solver
        for ax in [ax_rho, ax_p, ax_u, ax_T]:
            ax.axvline(x=x_af_sol_um, color='#333333', linestyle='--', lw=1.5, alpha=0.7,
                       label='ablation front (solver)' if i == 0 else None)
            ax.axvline(x=x_sf_sol_um, color='#e63946', linestyle='--', lw=1.5, alpha=0.7,
                       label='shock front (solver)' if i == 0 else None)

    # Build legend: profiles + front lines
    if compare_imc:
        legend_handles = [
            Line2D([0], [0], color='#1a5fb4', lw=2.5, linestyle='-',  label='Diffusion'),
            Line2D([0], [0], color='#333333', lw=2.0, linestyle='-',  label='Analytic'),
            Line2D([0], [0], color='forestgreen', lw=1.8, linestyle='--', label='IMC (2 ns)'),
            Line2D([0], [0], color='#333333', lw=1.5, linestyle='--', alpha=0.7, label='Ablation front (Analytic)'),
            Line2D([0], [0], color='#e63946', lw=1.5, linestyle='--', alpha=0.7, label='Shock front (Analytic)'),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], color='#1a5fb4', lw=2.5, linestyle='-',  label='Simulation'),
            Line2D([0], [0], color='#333333', lw=2.0, linestyle='-',  label='Exact Patched Solver'),
            Line2D([0], [0], color='#e67e22', lw=2.0, linestyle=':',  label='Patched Fit'),
            Line2D([0], [0], color='#333333', lw=1.5, linestyle='--', alpha=0.7, label='ablation front (solver)'),
            Line2D([0], [0], color='#e63946', lw=1.5, linestyle='--', alpha=0.7, label='shock front (solver)'),
        ]
    ax_rho.legend(handles=legend_handles, loc="lower left", fontsize=11, ncol=2)

    # Set y limits based on simulation bounds
    all_sim_rho = []
    all_sim_p = []
    all_sim_u = []
    all_sim_T = []
    for t_target in target_times:
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        all_sim_rho.extend(history.rho[idx_sim])
        all_sim_p.extend(history.p[idx_sim] / p_scale)
        all_sim_u.extend(history.u[idx_sim] / u_scale)
        all_sim_T.extend(history.T[idx_sim] / T_scale)

    max_sim_rho = np.max(all_sim_rho)
    max_sim_p = np.max(all_sim_p)
    max_sim_u_abs = np.max(np.abs(all_sim_u))
    max_sim_T = np.max(all_sim_T)

    ax_rho.set_yscale('log')
    # Keep other graphs linear as requested

    ax_rho.set_ylim(1e-2, max_sim_rho * 2.0)
    ax_p.set_ylim(-0.05 * max_sim_p, max_sim_p * 1.15)
    ax_u.set_ylim(-max_sim_u_abs * 0.15, max_sim_u_abs * 0.05)
    ax_T.set_ylim(-0.05 * max_sim_T, max_sim_T * 1.15)

    # Styling
    titles = ["Density", "Pressure", "Velocity", "Temperature"]
    math_labels = [r"$\rho$ [g/cm³]", r"$P$ [MBar]", r"$u$ [km/s]", r"$T$ [HeV]"]
    for j, ax in enumerate([ax_rho, ax_p, ax_u, ax_T]):
        ax.grid(True, alpha=0.3)
        ax.set_title(titles[j], fontsize=14, fontweight='bold')
        ax.set_ylabel(math_labels[j], fontsize=13)
        ax.set_xlabel(r"Eulerian Position Coordinate $x$ [$\mu$m]", fontsize=13)
        ax.set_xlim(left=-10.0, right=20)
        ax.tick_params(labelsize=11)

    if compare_imc:
        plt.suptitle(f"Unified Patched Ablation & Shock Verification vs IMC in Eulerian Coordinate\n{case_title}", fontsize=16, fontweight='bold')
    else:
        plt.suptitle(f"Unified Patched Ablation & Shock Verification (Fully Patched) in Eulerian Coordinate\n{case_title}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved fully patched comparison plot in Eulerian to {plot_path}")
    plt.close(fig)


def plot_multitime_profiles_eulerian(
    history,
    ablation_solver,
    sub_params,
    shock_params,
    case,
    plot_path,
    case_title,
    n_times: int = 6,
):
    """Plot Eulerian density/pressure/velocity/temperature profiles at several
    evenly-spaced simulation times, each time shown in a distinct colour.

    Piston positions (sim, solver, fit) are marked as vertical lines in the
    same colour as the profile with different line styles.

    Parameters
    ----------
    n_times:
        Number of time snapshots to overlay (evenly spaced from 10 % to 100 %
        of the simulation duration).
    """
    print(f"Generating multi-time Eulerian profiles for {case_title}...")

    times_all = np.asarray(history.t, dtype=float)
    t_max = times_all[-1]

    # Pick n_times equally spaced target times (skip t=0)
    fracs = np.linspace(0.1, 1.0, n_times)
    target_times = fracs * t_max

    # Vivid colour palette — one colour per time snapshot
    palette = [
        "#e6194b", "#f58231", "#ffe119", "#3cb44b",
        "#4363d8", "#911eb4", "#42d4f4", "#f032e6",
        "#a9a9a9", "#800000",
    ]
    colours = [palette[i % len(palette)] for i in range(n_times)]

    hs = ablation_solver.heat_solver
    ss = ablation_solver.shock_solver
    q1 = 1.0 - ss.omega
    q2 = (2.0 - ss.omega) / (ss.tau + 2.0)

    p_scale = 1e12
    u_scale = 1e5
    T_scale = 1.160451812e6

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u,   ax_T = axes[1, 0], axes[1, 1]
    all_axes = [ax_rho, ax_p, ax_u, ax_T]
    labels   = ["Density [g/cm³]", "Pressure [MBar]", "Velocity [km/s]", "Temperature [HeV]"]

    legend_handles = []

    for i, (t_target, col) in enumerate(zip(target_times, colours)):
        idx_sim = int(np.argmin(np.abs(times_all - t_target)))
        t_actual = float(times_all[idx_sim])
        t_ns     = t_actual * 1e9

        x_sim_raw      = np.asarray(history.x[idx_sim], dtype=float)
        x_sim_um       = x_sim_raw * 1e4
        m_sim          = np.asarray(history.m[idx_sim], dtype=float)
        sim_rho        = np.asarray(history.rho[idx_sim], dtype=float)
        sim_p          = np.asarray(history.p[idx_sim],   dtype=float) / p_scale
        sim_u          = np.asarray(history.u[idx_sim],   dtype=float) / u_scale
        sim_T          = np.asarray(history.T[idx_sim],   dtype=float) / T_scale

        # --- Simulation profiles (solid) ---
        for ax, ydata in zip(all_axes, [sim_rho, sim_p, sim_u, sim_T]):
            ax.plot(x_sim_um, ydata, '-', color=col, lw=1.8, alpha=0.85)

        x_piston_sim_um = np.nan

        # Solver piston = heat_position
        sol = ablation_solver.solve(mass=m_sim, time=t_actual)
        x_piston_sol_um = float(sol["heat_position"]) * 1e4

        # Fit piston
        pos_scale = ss._position_temporal_factor(time=t_actual)
        m_f   = hs.ablated_mass(time=t_actual)
        xsi_mf = m_f * ss.xsi_over_m(time=t_actual)
        y_mf   = xsi_mf / ss.xsi_s
        _, _, U_fit_mf, rho_fit_mf = shock_fit_by_params(np.array([y_mf]), shock_params)
        x_af_fit = pos_scale * (q1 * xsi_mf / rho_fit_mf[0] + q2 * U_fit_mf[0])
        x_piston_fit_um = x_af_fit * 1e4

        for ax in all_axes:
            if np.isfinite(x_piston_sim_um):
                ax.axvline(x_piston_sim_um, color=col, ls='-',  lw=1.2, alpha=0.7)
            ax.axvline(x_piston_sol_um,     color=col, ls='--', lw=1.2, alpha=0.7)
            ax.axvline(x_piston_fit_um,     color=col, ls=':',  lw=1.4, alpha=0.7)

        legend_handles.append(
            Line2D([0], [0], color=col, lw=2.0, ls='-',
                   label=f"t = {t_ns:.2f} ns")
        )

    # Style legend: line-style key
    legend_handles += [
        Line2D([0], [0], color='k', lw=1.2, ls='-',  label='piston (sim)'),
        Line2D([0], [0], color='k', lw=1.2, ls='--', label='piston (solver)'),
        Line2D([0], [0], color='k', lw=1.4, ls=':',  label='piston (fit)'),
    ]

    for j, ax in enumerate(all_axes):
        ax.grid(True, alpha=0.3)
        
    titles = ["Density", "Pressure", "Velocity", "Temperature"]
    math_labels = [r"$\rho$ [g/cm³]", r"$P$ [MBar]", r"$u$ [km/s]", r"$T$ [HeV]"]
    for j, ax in enumerate(all_axes):
        ax.set_title(titles[j], fontsize=14, fontweight='bold')
        ax.set_ylabel(math_labels[j], fontsize=13)
        ax.set_xlabel(r"Eulerian Position $x$ [$\mu$m]", fontsize=13)
        ax.tick_params(labelsize=11)

    ax_rho.legend(handles=legend_handles, loc="upper left", fontsize=11, ncol=2)

    plt.suptitle(
        f"Multi-Time Eulerian Profiles (Simulation)\n{case_title}",
        fontsize=16, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved multi-time Eulerian profiles to {plot_path}")
    plt.close(fig)


# =============================================================================
# Time-Dependent Front Trajectory Plotting
# =============================================================================

def plot_front_trajectories_eulerian(history, ablation_solver, sub_params, shock_params, case, plot_path, case_title):
    """
    Plots the trajectories of the ablation boundary, shock piston, and shock front
    as functions of time, comparing Simulation, Solver, and Analytical fits.
    """
    print(f"Generating front trajectories plot in Eulerian for {case_title}...")
    times = np.asarray(history.t, dtype=float)
    x_sim = np.asarray(history.x, dtype=float)
    
    # Grid sizes – x_sim stores cell centres (Ncells columns), so n_cells = Ncells.
    n_cells = x_sim.shape[1]
    mass_grid = _build_mass_grid(case, num_cells=n_cells)

    times_model = _get_equally_spaced_elements(times, 200)

    # 1) Simulation front tracking
    #    detect_sim_shock_front_trajectory / detect_sim_ablation_boundary work
    #    directly on the full (unmasked) history arrays, so indexing is safe.
    x_shock_sim_raw = detect_sim_shock_front_trajectory(
        history.rho,
        history.m,
        x_sim,
        rho_unshocked=float(case.rho0),
        gamma=float(case.r) + 1.0,
        smooth_window=5,
        extrap_t_ns=0.1,
        extrap_times=times,
    )
    # Convert to µm and apply a light rolling mean to reduce jitter
    x_shock_sim = _rolling_mean(x_shock_sim_raw * 1e4, 3)

    # Ablation boundary: left face of the leftmost cell
    x_boundary_sim = detect_sim_ablation_boundary(x_sim) * 1e4

    x_piston_sim = np.full_like(times, np.nan, dtype=float)

    # 2) Solver exact front tracking
    x_boundary_sol       = np.zeros_like(times_model)
    x_piston_sol         = np.zeros_like(times_model)
    x_ablation_front_sol = np.zeros_like(times_model)
    x_shock_sol          = np.zeros_like(times_model)

    for i, t in enumerate(times_model):
        t_val = max(float(t), 1e-18)
        sol = ablation_solver.solve(mass=mass_grid, time=t_val)
        x_boundary_sol[i]       = (sol["boundary_position"] + sol["heat_position"]) * 1e4
        x_piston_sol[i]         = sol["piston_position"]    * 1e4
        x_ablation_front_sol[i] = sol["heat_position"]      * 1e4
        x_shock_sol[i]          = sol["shock_position"]     * 1e4

    # 3) Analytic-fit front tracking (shared utility)
    fit_fronts = compute_fit_front_trajectories(
        times_model, ablation_solver, sub_params, shock_params
    )
    x_boundary_fit       = fit_fronts["boundary"]       * 1e4
    x_ablation_front_fit = fit_fronts["ablation_front"] * 1e4
    x_piston_fit         = fit_fronts["piston"]         * 1e4
    x_shock_fit          = fit_fronts["shock"]          * 1e4
        
    # Plot curves — no auto-labels; legend is built manually below
    fig, ax = plt.subplots(figsize=(10, 7))
    t_ns = times * 1e9
    t_ns_model = times_model * 1e9

    # Simulation (solid)
    ax.plot(t_ns, x_boundary_sim, '-',  color='black',   lw=2.5)
    ax.plot(t_ns, x_piston_sim,   '-',  color='blue',    lw=2.0)
    ax.plot(t_ns, x_shock_sim,    '-',  color='red',     lw=2.5)

    # Solver (dashed)
    ax.plot(t_ns_model, x_boundary_sol,       '--', color='grey',    lw=2.0)
    ax.plot(t_ns_model, x_piston_sol,         '--', color='green',   lw=2.0)
    ax.plot(t_ns_model, x_shock_sol,          '--', color='darkred', lw=2.0)
    # Ablation front solver (background reference, no legend entry)
    ax.plot(t_ns_model, x_ablation_front_sol, '--', color='fuchsia', lw=1.0, alpha=0.35)

    # Fit (dotted)
    ax.plot(t_ns_model, x_boundary_fit, ':', color='dimgray', lw=2.2)
    ax.plot(t_ns_model, x_piston_fit,   ':', color='cyan',    lw=2.2)
    ax.plot(t_ns_model, x_shock_fit,    ':', color='orange',  lw=2.2)
    # Ablation front fit (background reference, no legend entry)
    ax.plot(t_ns_model, x_ablation_front_fit, ':', color='purple', lw=1.0, alpha=0.35)

    ax.set_xlabel(r"$t$ [ns]", fontsize=13)
    ax.set_ylabel(r"$x$ [$\mu$m]", fontsize=13)
    ax.set_title(f"Front Trajectory Evolution (Simulation vs Solver vs Fit)\n{case_title}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- Structured legend: rows = front type, columns = source ---
    # Layout (ncol=3): [Simulation | Solver | Fit] per row
    legend_handles = [
        # Row: shock
        Line2D([0], [0], color='red',     lw=2.5, ls='-',  label='shock Simulation'),
        Line2D([0], [0], color='darkred', lw=2.0, ls='--', label='shock Solver'),
        Line2D([0], [0], color='orange',  lw=2.2, ls=':',  label='shock Fit'),

        # Row: piston
        Line2D([0], [0], color='blue',    lw=2.0, ls='-',  label='piston Simulation'),
        Line2D([0], [0], color='green',   lw=2.0, ls='--', label='piston Solver'),
        Line2D([0], [0], color='cyan',    lw=2.2, ls=':',  label='piston Fit'),

        # Row: boundary
        Line2D([0], [0], color='black',   lw=2.5, ls='-',  label='boundary Simulation'),
        Line2D([0], [0], color='grey',    lw=2.0, ls='--', label='boundary Solver'),
        Line2D([0], [0], color='dimgray', lw=2.2, ls=':',  label='boundary Fit'),
    ]
    ax.legend(
        handles=legend_handles,
        ncol=3,
        fontsize=10,
        loc='best',
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    # Limit time window
    ax.set_xlim(0, times[-1] * 1e9)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved front trajectories plot in Eulerian to {plot_path}")
    plt.close(fig)


def plot_fronts_vs_time_comparison(history, ablation_solver, sub_params, shock_params, case, plot_path, case_title):
    """
    Plots front positions (x in um) and front mass coordinates (m in mg/cm2) vs time,
    comparing Simulation, Solver, and Fits.
    """
    print(f"Generating time-dependent front comparisons for {case_title}...")
    times = np.asarray(history.t, dtype=float)
    x_sim = np.asarray(history.x, dtype=float)
    t_ns = times * 1e9
    
    n_cells = x_sim.shape[1]
    mass_grid = _build_mass_grid(case, num_cells=n_cells)
    times_model = _get_equally_spaced_elements(times, 200)
    t_ns_model = times_model * 1e9

    # 1) Simulation shock front and ablation front detection
    # Shock front (position and mass)
    x_shock_sim_raw = detect_sim_shock_front_trajectory(
        history.rho,
        history.m,
        x_sim,
        rho_unshocked=float(case.rho0),
        gamma=float(case.r) + 1.0,
        smooth_window=5,
        extrap_t_ns=0.1,
        extrap_times=times,
    )
    x_shock_sim_um = _rolling_mean(x_shock_sim_raw * 1e4, 3)
    
    m_shock_sim_mg = np.full_like(times, np.nan, dtype=float)
    for k in range(len(times)):
        rhok = _rolling_mean(history.rho[k], 5)
        ishock, _ = find_shock_front(
            rhok,
            history.m[k],
            rho_unshocked=float(case.rho0),
            gamma=float(case.r) + 1.0,
        )
        if ishock >= 1:
            m_shock_sim_mg[k] = history.m[k][ishock] * 1e3 # g/cm2 -> mg/cm2

    # Ablation front (position and mass)
    x_ablation_sim_raw = detect_sim_ablation_front(
        history.rho,
        history.m,
        x_sim,
        rho_unshocked=float(case.rho0),
        gamma=float(case.r) + 1.0,
        smooth_window=5,
    )
    x_ablation_sim_um = x_ablation_sim_raw * 1e4
    
    m_ablation_sim_mg = np.full_like(times, np.nan, dtype=float)
    for k in range(len(times)):
        if np.isfinite(x_ablation_sim_raw[k]):
            i_ab = np.nanargmin(np.abs(x_sim[k] - x_ablation_sim_raw[k]))
            m_ablation_sim_mg[k] = history.m[k][i_ab] * 1e3

    # Apply rolling mean to clean up any minor detection jitter
    x_ablation_sim_um = _rolling_mean(x_ablation_sim_um, 3)
    m_ablation_sim_mg = _rolling_mean(m_ablation_sim_mg, 3)

    # 2) Solver fronts
    x_ablation_sol = np.zeros_like(times_model)
    x_shock_sol = np.zeros_like(times_model)
    m_ablation_sol_mg = np.zeros_like(times_model)
    m_shock_sol_mg = np.zeros_like(times_model)

    for i, t in enumerate(times_model):
        t_val = max(float(t), 1e-18)
        sol = ablation_solver.solve(mass=mass_grid, time=t_val)
        x_ablation_sol[i] = sol["heat_position"] * 1e4
        x_shock_sol[i] = sol["shock_position"] * 1e4
        m_ablation_sol_mg[i] = ablation_solver.heat_solver.ablated_mass(time=t_val) * 1e3
        m_shock_sol_mg[i] = ablation_solver.shock_solver.shocked_mass(time=t_val) * 1e3

    # 3) Fit fronts
    fit_fronts = compute_fit_front_trajectories(
        times_model, ablation_solver, sub_params, shock_params
    )
    x_ablation_fit = fit_fronts["ablation_front"] * 1e4
    x_shock_fit = fit_fronts["shock"] * 1e4
    # Fits use the same ablated/shocked mass formulas as solver
    m_ablation_fit_mg = m_ablation_sol_mg
    m_shock_fit_mg = m_shock_sol_mg

    # 4) Plotting
    fig, (ax_x, ax_m) = plt.subplots(1, 2, figsize=(14, 6.5))

    # Left Panel: Position x
    # Simulation (solid)
    ax_x.plot(t_ns, x_ablation_sim_um, '-', color='#1a5fb4', lw=2.5, label='Ablation (Diffusion)')
    ax_x.plot(t_ns_model, x_ablation_sol, '--', color='#333333', lw=2.0, label='Ablation (Analytic)')
    ax_x.plot(t_ns_model, x_ablation_fit, ':', color='#26a269', lw=2.2, label='Ablation (Fit)')
    ax_x.plot(t_ns, x_shock_sim_um, '-', color='#e63946', lw=2.5, label='Shock (Diffusion)')
    ax_x.plot(t_ns_model, x_shock_sol, '--', color='#8b0000', lw=2.0, label='Shock (Analytic)')
    ax_x.plot(t_ns_model, x_shock_fit, ':', color='#e67e22', lw=2.2, label='Shock (Fit)')

    ax_x.set_title("Front Position $x$", fontsize=14, fontweight='bold')
    ax_x.set_ylabel(r"$x$ [$\mu$m]", fontsize=13)
    ax_x.set_xlabel(r"$t$ [ns]", fontsize=13)
    ax_x.grid(True, alpha=0.3)
    ax_x.tick_params(labelsize=11)
    ax_x.legend(loc='best', fontsize=11)
    ax_x.set_xlim(0, t_ns[-1])

    # Right Panel: Mass Coordinate m
    # Simulation (solid)
    ax_m.plot(t_ns, m_ablation_sim_mg, '-', color='#1a5fb4', lw=2.5, label='Ablation (Diffusion)')
    ax_m.plot(t_ns_model, m_ablation_sol_mg, '--', color='#333333', lw=2.0, label='Ablation (Analytic)')
    ax_m.plot(t_ns_model, m_ablation_fit_mg, ':', color='#26a269', lw=2.2, label='Ablation (Fit)')
    ax_m.plot(t_ns, m_shock_sim_mg, '-', color='#e63946', lw=2.5, label='Shock (Diffusion)')
    ax_m.plot(t_ns_model, m_shock_sol_mg, '--', color='#8b0000', lw=2.0, label='Shock (Analytic)')
    ax_m.plot(t_ns_model, m_shock_fit_mg, ':', color='#e67e22', lw=2.2, label='Shock (Fit)')

    ax_m.set_title("Front Mass Coordinate $m$", fontsize=14, fontweight='bold')
    ax_m.set_ylabel(r"$m$ [mg/cm²]", fontsize=13)
    ax_m.set_xlabel(r"$t$ [ns]", fontsize=13)
    ax_m.grid(True, alpha=0.3)
    ax_m.tick_params(labelsize=11)
    ax_m.legend(loc='best', fontsize=11)
    ax_m.set_xlim(0, t_ns[-1])

    plt.suptitle(f"Front Trajectory & Mass Coordinate Comparison\n{case_title}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved front position and mass comparisons vs time to {plot_path}")
    plt.close(fig)


# =============================================================================
# Main Execution Workflow
# =============================================================================

def run_preset_workflow(preset_name, case_label, case_title, compare_imc=False, N=None):
    """Run simulation, load reference solvers, run fitting pipelines, and plot eulerian profiles and trajectories."""
    case, history, ablation_solver = get_data(preset_name, case_label, N=N)

    # 1) Run Subsonic fitting pipeline using ablation_solver.heat_solver
    print("--- Running Subsonic Ablation Fitting ---")
    sub_solver = ablation_solver.heat_solver
    sub_params = perform_subsonic_fitting(sub_solver)

    # 2) Run Piston Shock fitting pipeline using ablation_solver.shock_solver
    print("--- Running Piston Shock Fitting ---")
    shock_solver = ablation_solver.shock_solver
    shock_params = perform_shock_fitting(shock_solver)

    ev_dir = _REPO_ROOT / "results" / "ictt" / case_label / "eulerian_verification"
    ev_dir.mkdir(parents=True, exist_ok=True)

    # 3) Plot 1: Subsonic & Shock Overlays in Eulerian coordinate (Disabled as requested)
    # plot_path_overlays = ev_dir / f"{case_label}_patched_fit_comparison_eulerian.png"
    # plot_patched_dimensional_fit_comparison_eulerian(
    #     history=history,
    #     ablation_solver=ablation_solver,
    #     sub_params=sub_params,
    #     shock_params=shock_params,
    #     case=case,
    #     plot_path=str(plot_path_overlays),
    #     case_title=case_title,
    # )

    # 4) Plot 2: Fully Patched seamless profiles in Eulerian coordinate
    plot_path_patched = ev_dir / f"{case_label}_fully_patched_comparison_eulerian.png"
    plot_fully_patched_comparison_eulerian(
        history=history,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
        case=case,
        plot_path=str(plot_path_patched),
        case_title=case_title,
        compare_imc=compare_imc,
    )
    
    # 5) Plot 3: Time-dependent front trajectories in Eulerian coordinate
    plot_path_trajectories = ev_dir / f"{case_label}_front_trajectories_eulerian.png"
    plot_front_trajectories_eulerian(
        history=history,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
        case=case,
        plot_path=str(plot_path_trajectories),
        case_title=case_title,
    )

    # Plot 3b: Front Position and Mass Coordinate vs Time (2 subplots)
    plot_path_fronts_vs_time = ev_dir / f"{case_label}_fronts_vs_time.png"
    plot_fronts_vs_time_comparison(
        history=history,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
        case=case,
        plot_path=str(plot_path_fronts_vs_time),
        case_title=case_title,
    )

    # 6) Plot 4: Multi-time Eulerian profiles (simulation, colour-coded by time)
    plot_path_multitime = ev_dir / f"{case_label}_multitime_profiles_eulerian.png"
    plot_multitime_profiles_eulerian(
        history=history,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
        case=case,
        plot_path=str(plot_path_multitime),
        case_title=case_title,
        n_times=6,
    )


def main():
    from project3_code.rad_hydro_sim.problems.presets_config import (
        PRESET_FIG_8_CONSTANT_TEMPERATURE,
        PRESET_FIG_9_CONSTANT_FLUX,
        PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
    )
    run_preset_workflow(
        PRESET_FIG_8_CONSTANT_TEMPERATURE,
        "const_T",
        "Fig 8 Constant Temperature Drive",
        compare_imc=True,
        N=1024,
    )
    run_preset_workflow(
        PRESET_FIG_9_CONSTANT_FLUX,
        "const_S",
        "Fig 9 Constant Flux Drive",
        compare_imc=False,
    )
    run_preset_workflow(
        PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
        "const_P_shock",
        "Fig 10 Constant Ablation Pressure Drive",
        compare_imc=False,
    )
    print("\nPatched eulerian ablation and shock simulations, fittings, comparisons, and plots generated successfully!")


if __name__ == "__main__":
    main()

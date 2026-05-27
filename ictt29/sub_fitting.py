# ictt29/plot_test_cases.py
"""
verification and similarity profile fitting script for Fig 8 and Fig 9.

Compiles and compares:
1. 1D Rad-Hydro Simulation.
2. Shussman piecewise reference solver (subsonic + shock).
3. Menahem AblationSolver (piecewise subsonic + shock).

Outputs generated inside results/ictt/ (split by graph types):
- xt/ Space-time trajectory and fronts (PNG)
- material_hydro/ e, u, p, rho vs mass coordinate m (PNG)
- rad_hydro/ E_rad, E_mat, T_rad, T_mat vs mass coordinate m (PNG)
- self_similar/ Dimensionless similarity profiles T, P, U, S with curve-fits (PNG)
- evolution/ Animated 3-way GIFs (Simulation vs Shussman vs Menahem).
"""
from __future__ import annotations

import os
import sys
import pickle
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
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_9_CONSTANT_FLUX,
)
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.verification.shussman_comparison import run_shussman_piecewise_reference
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    run_menahem_piecewise_reference,
    _ablation_kwargs_from_case,
    _heat_kwargs_from_case,
    _ns_amplitude_rescale,
    _build_mass_grid,
)
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV
from project3_code.hydro_sim.plotting.hydro_plots import _create_7panel_vertical_figure

from ablation_solver import AblationSolver
from subsonic_heat_wave import SubsonicHeatWave
from piston_shock import PistonShock

RUN_GIFS = False  # Set to True to generate animated evolution GIFs (which takes time)

def get_cached_subsonic_solver(case, case_label):
    """Solve subsonic similarity ODEs once and cache the solver object (with found xsi_f)."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver_cache_path = cache_dir / f"{case_label}_similarity_solver.pkl"
    
    if solver_cache_path.exists():
        print(f"Loading cached subsonic similarity solver from {solver_cache_path}...")
        try:
            with open(solver_cache_path, "rb") as f:
                solver = pickle.load(f)
            # Re-bind the ODE solver which contains method callbacks
            solver.ode_solver = scipy.integrate.ode(solver.fode).set_integrator(solver.ode_scheme)
            print("Similarity solver loaded successfully.")
            return solver
        except Exception as e:
            print(f"Failed to load solver cache: {e}. Re-solving subsonic ODEs...")
            
    print("Solving subsonic similarity ODEs (finding xsi_f via shooting method)...")
    heat_kwargs = _heat_kwargs_from_case(case)
    solver = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
    
    # Save cache by removing ode_solver temporarily to avoid pickling issues
    try:
        ode_solver = solver.ode_solver
        del solver.ode_solver
        with open(solver_cache_path, "wb") as f:
            pickle.dump(solver, f, protocol=pickle.HIGHEST_PROTOCOL)
        solver.ode_solver = ode_solver
        print(f"Saved subsonic similarity solver to cache: {solver_cache_path}")
    except Exception as e:
        print(f"Failed to save solver cache: {e}")
        
    return solver



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

def plot_xt_trajectories(history, case, xt_path, case_title, ablation_solver=None):
    """Plot cell boundaries x(t) and diagnosed fronts (Simulation vs Analytic)."""
    print(f"Generating space-time (xt) plot for {case_title}...")
    times = np.asarray(history.t, dtype=float)
    x_sim = np.asarray(history.x, dtype=float)
    n_cells = x_sim.shape[1] - 1
    
    # 1) Menahem AblationSolver analytic solution
    if ablation_solver is None:
        ablation_solver = AblationSolver(**_ablation_kwargs_from_case(case))
    mass_grid = _build_mass_grid(case, num_cells=n_cells)
    
    times_model = _get_equally_spaced_elements(times, 200)
    results = []
    for t in times_model:
        sol = ablation_solver.solve(mass=mass_grid, time=max(float(t), 1e-18))
        results.append(sol)
        
    position_times = np.array([r["position"] for r in results]).T
    shock_position = np.array([r["shock_position"] for r in results], dtype=float)
    piston_position = np.array([r["piston_position"] for r in results], dtype=float)
    heat_position = np.array([r["heat_position"] for r in results], dtype=float)
    boundary_position = np.array([r["boundary_position"] for r in results], dtype=float)
    
    # 2) Simulation shock from density profile
    x_shock_sim = np.full(times.size - 1, np.nan, dtype=float)
    for k in range(1, times.size):
        rhok = np.asarray(history.rho[k], dtype=float)
        mk = np.asarray(history.m[k], dtype=float)
        rhok_smooth = _rolling_mean(rhok, 5)
        ishock, _ = find_shock_front(
            rhok_smooth,
            mk,
            rho_unshocked=float(case.rho0),
            gamma=float(case.r) + 1.0,
            Hugoniot_threshold=0.5,
        )
        if ishock >= 1:
            x_shock_sim[k - 1] = float(x_sim[k, ishock])
            
    # Apply linear regression correction for small times (t < 0.002 ns)
    later_times = np.array([])
    later_x = np.array([])
    for k in range(1, times.size):
        t_ns = times[k] * 1e9
        if t_ns >= 0.002 and not np.isnan(x_shock_sim[k - 1]):
            later_times = np.append(later_times, times[k])
            later_x = np.append(later_x, x_shock_sim[k - 1])
            
    if len(later_times) >= 2:
        slope, intercept = np.polyfit(np.log(later_times+1e-20), np.log(later_x), 1)
        for k in range(1, times.size):
            t_ns = times[k] * 1e9
            if t_ns < 0.002:
                x_shock_sim[k - 1] = np.exp(slope * np.log(times[k] + 1e-20) + intercept)
            
    # 3) Setup figure
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    
    # Plot mass trajectories - thin background grid so they don't cover the fronts
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
    ax.plot(times[1:] * 1e9, x_shock_sim, lw=2.5, c="red", label="Shock (simulation)")
    ax.plot(times_model * 1e9, shock_position, lw=2.0, ls="--", c="darkred", label="Shock (Menahem)")
    ax.plot(times_model * 1e9, piston_position, lw=2.0, ls="--", c="green", label="Piston (Menahem)")
    ax.plot(times_model * 1e9, heat_position, lw=2.0, ls="--", c="purple", label="Heat Wave (Menahem)")
    # ax.plot(times_model * 1e9, boundary_position, lw=2.0, ls="--", c="black", label="Boundary (Menahem)")
    
    ax.set_xlabel(r"$t$ [ns]", fontsize=12)
    ax.set_ylabel(r"$x$ [cm]", fontsize=12)
    ax.set_title(f"Space-Time (xt) Trajectories and Fronts\n{case_title}", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9.5)
    # ax.set_xlim(times.min() * 1e9, times.max() * 1e9)
    ax.set_ylim(0, 0.00016)
    
    fig.tight_layout()
    fig.savefig(xt_path, dpi=200)
    plt.close(fig)


def evaluate_subsonic_fits(mass_grid, t_sec, solver, popt_T, popt_P, best_u):
    """
    Computes the explicit self-similar analytical fits for subsonic regime in CGS units.
    Uses the optimized velocity fit.
    """
    t_ns = t_sec * 1e9
    m_f = solver.xsi_f / (solver.A**solver.a * solver.B**solver.b * (1e-9)**solver.c) * (t_ns ** 0.515625)
    
    n = len(mass_grid)
    rho = np.zeros(n)
    p = np.zeros(n)
    u = np.zeros(n)
    T = np.zeros(n)
    
    # Subsonic Fits
    r_val = 0.25
    beta_val = 1.6
    mu_val = 0.14
    f_cgs = 3.4e13
    
    for i, m in enumerate(mass_grid):
        if m < m_f:
            y = m / m_f
            
            # 1. Temperature fit in HeV
            T_hev = ((1.0 - y) * (1.0 + popt_T[0] * y)) ** (10.0 / 39.0)
            T[i] = T_hev * 1.160451812e6  # Kelvin
            
            # 2. Pressure fit in MBar
            pre_p = solver.Pf * (solver.A**solver.a3) * (solver.B**solver.b3)
            p_val_mbar = (pre_p * 1e-12) * (t_ns ** -0.447917) * (popt_P[0] * y**popt_P[2] + popt_P[1] * y**(popt_P[2]+popt_P[3]))
            p[i] = p_val_mbar * 1e12  # Barye
            
            # 3. Velocity fit in km/s
            coeff = (solver.A**solver.a2) * (solver.B**solver.b2) * 1e-5
            U_dim_kms = coeff * best_u["func"](y, *best_u["popt"]) * (t_ns ** 0.036458)
            u[i] = U_dim_kms * 1e5  # cm/s
            
            # 4. Density derived via CGS EOS
            rho[i] = ((r_val * f_cgs * T_hev**beta_val) / np.maximum(p[i], 1e-15)) ** (1.0 / (mu_val - 1.0))
        else:
            # Outside subsonic front
            rho[i] = 19.32
            p[i] = 1e-6
            u[i] = 0.0
            T[i] = 300.0
            
    return {"density": rho, "pressure": p, "velocity": u, "temperature": T, "m_f": m_f}


def plot_material_hydro_profiles(history, solver, popt_T, popt_P, best_u, material_hydro_path, case_title):
    """Plot overlay profiles of T, rho, P, u vs m comparing Simulation, Exact Solver, and Analytic fits."""
    print(f"Generating subsonic physical comparison profiles for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    target_times = [1e-9, 1.5e-9, 2e-9]
    colors = ["red", "green", "blue"]
    p_scale, u_scale = 1e12, 1e5
    
    ax_T = axes[0, 0]
    ax_rho = axes[0, 1]
    ax_p = axes[1, 0]
    ax_u = axes[1, 1]
    
    for t_target, color in zip(target_times, colors):
        # 1) Simulation
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]
        
        sim_rho = history.rho[idx_sim]
        sim_p = history.p[idx_sim] / p_scale
        sim_u = history.u[idx_sim] / u_scale
        sim_T = history.T[idx_sim]
        
        # 2) Exact Solver
        m_f_exact = solver.ablated_mass(time=t_actual)
        mass_exact = np.linspace(1e-12, m_f_exact, 200)
        sol_exact = solver.solve(mass=mass_exact, time=t_actual)
        
        exact_rho = sol_exact["density"]
        exact_p = sol_exact["pressure"] / p_scale
        exact_u = sol_exact["velocity"] / u_scale
        exact_T = sol_exact["temperature"]
        
        # 3) Analytical fits mapped to CGS
        fits = evaluate_subsonic_fits(mass_exact, t_actual, solver, popt_T, popt_P, best_u)
        fit_rho = fits["density"]
        fit_p = fits["pressure"] / p_scale
        fit_u = fits["velocity"] / u_scale
        fit_T = fits["temperature"]
        
        # Plot Temperature (K)
        ax_T.plot(m_sim * 1e3, sim_T, '.', color=color, alpha=0.3, markersize=3, label=f'Simulation ({t_target*1e9:.1f} ns)' if color=='red' else None)
        ax_T.plot(mass_exact * 1e3, exact_T, '-', color=color, lw=2.0, label=f'Exact Solver ({t_target*1e9:.1f} ns)' if color=='red' else None)
        ax_T.plot(mass_exact * 1e3, fit_T, '--', color=color, lw=1.5, label=f'Analytic Fit ({t_target*1e9:.1f} ns)' if color=='red' else None)
        
        # Plot Density (g/cm^3)
        ax_rho.plot(m_sim * 1e3, sim_rho, '.', color=color, alpha=0.3, markersize=3)
        ax_rho.plot(mass_exact * 1e3, exact_rho, '-', color=color, lw=2.0)
        ax_rho.plot(mass_exact * 1e3, fit_rho, '--', color=color, lw=1.5)
        
        # Plot Pressure (MBar)
        ax_p.plot(m_sim * 1e3, sim_p, '.', color=color, alpha=0.3, markersize=3)
        ax_p.plot(mass_exact * 1e3, exact_p, '-', color=color, lw=2.0)
        ax_p.plot(mass_exact * 1e3, fit_p, '--', color=color, lw=1.5)
        
        # Plot Velocity (km/s)
        ax_u.plot(m_sim * 1e3, sim_u, '.', color=color, alpha=0.3, markersize=3)
        ax_u.plot(mass_exact * 1e3, exact_u, '-', color=color, lw=2.0)
        ax_u.plot(mass_exact * 1e3, fit_u, '--', color=color, lw=1.5)

    # Labels and Titles
    ax_T.set_ylabel(r"$T$ [Kelvin]", fontsize=12)
    ax_rho.set_ylabel(r"$\rho$ [g/cm³]", fontsize=12)
    ax_p.set_ylabel(r"$P$ [MBar]", fontsize=12)
    ax_u.set_ylabel(r"$u$ [km/s]", fontsize=12)
    
    ax_T.legend(loc='best', fontsize=9)
    
    for ax in axes.flat:
        ax.set_xlabel(r"Mass coordinate $m$ [mg/cm²]", fontsize=11)
        ax.grid(True, alpha=0.25)
        
    fig.suptitle(f"Dimensional Subsonic Profiles Comparison\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(material_hydro_path, dpi=200)
    plt.close(fig)


def plot_rad_hydro_profiles(history, shussman_ref, menahem_ref, rad_hydro_path, case_title):
    """Plot overlay profiles of T_mat, T_rad, E_rad, and coupling vs m (mass coordinate) at 0.05, 0.1, 0.15 ns (evolution snippets)."""
    print(f"Generating radiation-hydrodynamics profiles for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    target_times = [1e-9, 1.5e-9, 2e-9]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(target_times)))
    
    # Loop over times
    for t_target, color in zip(target_times, colors):
        # 1) Simulation
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        T_rad_sim = history.T[idx_sim]
        T_mat_sim = (history.T_material[idx_sim] if (hasattr(history, 'T_material') and history.T_material is not None) else history.T[idx_sim])
        E_rad_sim = history.E_rad[idx_sim]
        
        # 2) Shussman
        idx_sh = np.argmin(np.abs(np.array(shussman_ref.times) - t_target))
        m_sh = shussman_ref.m[idx_sh]
        T_rad_sh = shussman_ref.T[idx_sh] * KELVIN_PER_HEV
        T_mat_sh = T_rad_sh  # Subsonic radiation diffusion equilibrium
        E_rad_sh = shussman_ref.E_rad[idx_sh]
        
        # 3) Menahem
        idx_men = np.argmin(np.abs(np.array(menahem_ref.times) - t_target))
        m_men = menahem_ref.m[idx_men]
        T_rad_men = menahem_ref.T[idx_men] * KELVIN_PER_HEV
        T_mat_men = (menahem_ref.T_material[idx_men] if menahem_ref.T_material else menahem_ref.T[idx_men]) * KELVIN_PER_HEV
        E_rad_men = menahem_ref.E_rad[idx_men]
        
        # Panel (0,0): T_mat [K]
        axes[0, 0].plot(m_sim, T_mat_sim, color=color, linestyle='-', lw=1.8)
        axes[0, 0].plot(m_sh, T_mat_sh, color=color, linestyle='--', lw=1.5)
        axes[0, 0].plot(m_men, T_mat_men, color=color, linestyle='--', lw=1.5)
        
        # Panel (0,1): T_rad [K]
        axes[0, 1].plot(m_sim, T_rad_sim, color=color, linestyle='-', lw=1.8)
        axes[0, 1].plot(m_sh, T_rad_sh, color=color, linestyle='--', lw=1.5)
        axes[0, 1].plot(m_men, T_rad_men, color=color, linestyle='--', lw=1.5)
        
        # Panel (1,0): E_rad [erg/cm³]
        axes[1, 0].plot(m_sim, E_rad_sim, color=color, linestyle='-', lw=1.8)
        axes[1, 0].plot(m_sh, E_rad_sh, color=color, linestyle='--', lw=1.5)
        axes[1, 0].plot(m_men, E_rad_men, color=color, linestyle='--', lw=1.5)
        
        # Panel (1,1): T_mat and T_rad Coupling Comparison [K]
        axes[1, 1].plot(m_sim, T_mat_sim, color=color, linestyle='-', lw=1.8)
        axes[1, 1].plot(m_sim, T_rad_sim, color=color, linestyle='-.', lw=1.2)
        axes[1, 1].plot(m_men, T_mat_men, color=color, linestyle='--', lw=1.2)
        axes[1, 1].plot(m_men, T_rad_men, color=color, linestyle='--', lw=1.0)
        
    # Labels and Titles
    axes[0, 0].set_ylabel(r"$T_{\mathrm{mat}}$ [K]", fontsize=12)
    axes[0, 1].set_ylabel(r"$T_{\mathrm{rad}}$ [K]", fontsize=12)
    axes[1, 0].set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]", fontsize=12)
    axes[1, 1].set_ylabel(r"Coupling $T_{\mathrm{mat}}, T_{\mathrm{rad}}$ [K]", fontsize=12)
    
    for ax in axes.flat:
        ax.set_xlabel(r"Mass coordinate $m$ [g/cm²]", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        
    # Legend
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', lw=1.8, label='simulation'),
        Line2D([0], [0], color='black', linestyle='--', lw=1.5, label='Shussman'),
        Line2D([0], [0], color='black', linestyle='--', lw=1.5, label='Menahem'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[0], markersize=8, label='0.05 ns'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[1], markersize=8, label='0.10 ns'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[2], markersize=8, label='0.15 ns'),
        Line2D([0], [0], color='grey', linestyle='-.', lw=1.2, label=r'T_rad (Simulation)'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc='best', fontsize=9.5)
    
    fig.suptitle(f"Radiation-Hydrodynamics Profiles\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(rad_hydro_path, dpi=200)
def perform_subsonic_fitting(solver):
    y_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
    xsi_vec = y_grid * solver.xsi_f
    profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    P_val = profiles["P"]
    T_val = profiles["T"]
    U_val = profiles["U"]
    V_val = profiles["V"]
    rho_val = 1.0 / V_val
    
    valid_idx = (y_grid > 0.005) & np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(T_val)
    y_valid = y_grid[valid_idx]
    T_valid = T_val[valid_idx]
    P_valid = P_val[valid_idx]
    U_valid = U_val[valid_idx]
    rho_valid = rho_val[valid_idx]
    
    mask = y_valid < 0.99
    y_valid_masked = y_valid[mask]
    T_valid_masked = T_valid[mask]
    rho_valid_masked = rho_valid[mask]
    
    # Fits
    def smith_approximation(y, R):
        return ((1.0 - y) * (1.0 + R * y))**(10.0 / 39.0)
        
    def power_law_origin(y, a, b, c, d):
        return a * y**c + b * y**(c+d)
        
    popt_T, _ = curve_fit(smith_approximation, y_valid_masked, T_valid_masked, p0=[0.5])
    popt_P, _ = curve_fit(power_law_origin, y_valid, P_valid, p0=[0.355, 0.5, 0.04, 2.3])
    
    # Velocity fits
    def fit_u_1(y, c, d): return c * (1.0 - y)**d
    def fit_u_2(y, c, b): return c * (1.0 - y) / (1.0 + b * y)
    def fit_u_3(y, c, b): return c * (1.0 - y) * (y**(-b))
    def fit_u_4(y, c, a, b): return c * (1.0 - y**a) * (y**(-b))
    def fit_u_5(y, c, a, b): return c * (1.0 - y) * (y**a + y**(-b))
    def fit_u_6(y, c, a, b): return c * (1.0 - y**a) / (1.0 + b * y)
    
    candidates = [
        {"id": 1, "func": fit_u_1, "name": "Power Law: $c(1-y)^d$", "latex": r"U(y) \approx %.5f (1 - y)^{%.5f}", "p0": [U_valid[0], 0.5]},
        {"id": 2, "func": fit_u_2, "name": "Rational: $c(1-y)/(1+b y)$", "latex": r"U(y) \approx \frac{%.5f (1 - y)}{1 + %.5f y}", "p0": [U_valid[0], 0.5]},
        {"id": 3, "func": fit_u_3, "name": "Singular 2P: $c(1-y)y^{-b}$", "latex": r"U(y) \approx %.5f (1 - y) y^{-%.5f}", "p0": [-1.0, 0.3]},
        {"id": 4, "func": fit_u_4, "name": "Singular 3P: $c(1-y^a)y^{-b}$", "latex": r"U(y) \approx %.5f (1 - y^{%.5f}) y^{-%.5f}", "p0": [-1.0, 1.0, 0.3]},
        {"id": 5, "func": fit_u_5, "name": "User 3P: $c(1-y)(y^a + y^{-b})$", "latex": r"U(y) \approx %.5f (1 - y) (y^{%.5f} + y^{-%.5f)}", "p0": [-1.0, 1.0, 0.3]},
        {"id": 6, "func": fit_u_6, "name": "Rational Frac: $c(1-y^a)/(1+b y)$", "latex": r"U(y) \approx \frac{%.5f (1 - y^{%.5f})}{1 + %.5f y}", "p0": [U_valid[0], 1.0, 1.0]}
    ]
    
    mask_u = y_valid < 0.99
    y_valid_u = y_valid[mask_u]
    U_valid_u = U_valid[mask_u]
    
    best_u = None
    min_avg_err = float("inf")
    fits_u = {}
    
    for cand in candidates:
        try:
            popt, _ = curve_fit(cand["func"], y_valid_u, U_valid_u, p0=cand["p0"], maxfev=10000)
            U_fit = cand["func"](y_valid, *popt)
            U_fit_u = cand["func"](y_valid_u, *popt)
            rel_err_u = np.abs((U_fit_u - U_valid_u) / (U_valid_u + 1e-15)) * 100
            avg_err = np.mean(rel_err_u)
            max_err = np.max(rel_err_u)
            
            fits_u[cand["id"]] = (popt, U_fit, avg_err, max_err, cand["name"], cand["latex"])
            
            if avg_err < min_avg_err:
                min_avg_err = avg_err
                best_u = {
                    "id": cand["id"],
                    "popt": popt,
                    "func": cand["func"],
                    "name": cand["name"],
                    "latex": cand["latex"],
                    "avg_err": avg_err,
                    "max_err": max_err,
                    "fit_val": U_fit
                }
        except Exception as e:
            print(f"Subsonic velocity fit {cand['id']} failed: {e}")
            fits_u[cand["id"]] = (None, None, 0.0, 0.0, cand["name"], cand["latex"])
            
    return solver, y_grid, y_valid, T_valid, P_valid, U_valid, rho_valid, popt_T, popt_P, best_u, fits_u


def plot_and_fit_self_similar(solver, self_similar_path, standalone_path, relative_errors_path, case_title):
    """Plot similarity profiles, fits, error curves using the pre-solved solver."""
    # Run the exact fitting pipeline
    _, y_grid, y_valid, T_valid, P_valid, U_valid, rho_valid, popt_T, popt_P, best_u, fits_u = perform_subsonic_fitting(solver)
    
    # ----------------------------------------------------
    # Re-evaluate all chosen fits on y_valid
    # ----------------------------------------------------
    T_fit = ((1.0 - y_valid) * (1.0 + popt_T[0] * y_valid)) ** (10.0 / 39.0)
    P_fit = popt_P[0] * y_valid**popt_P[2] + popt_P[1] * y_valid**(popt_P[2]+popt_P[3])
    U_fit = best_u["fit_val"]
    
    r_val, beta_val, mu_val = 0.25, 1.6, 0.14
    rho_fit = ((r_val * T_fit**beta_val) / P_fit) ** (1.0 / (mu_val - 1.0))
    
    # Relative Errors (restricted to y < 0.99 for T and rho to avoid numerical singularities at the front boundary)
    mask_bulk = y_valid < 0.99
    y_bulk = y_valid[mask_bulk]
    
    err_T = np.abs((T_fit[mask_bulk] - T_valid[mask_bulk]) / T_valid[mask_bulk]) * 100
    err_rho = np.abs((rho_fit[mask_bulk] - rho_valid[mask_bulk]) / rho_valid[mask_bulk]) * 100
    err_P = np.abs((P_fit - P_valid) / P_valid) * 100
    err_U = np.abs((U_fit[mask_bulk] - U_valid[mask_bulk]) / (U_valid[mask_bulk] + 1e-15)) * 100
    
    avg_T, max_T = np.mean(err_T), np.max(err_T)
    avg_rho, max_rho = np.mean(err_rho), np.max(err_rho)
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = best_u["avg_err"], best_u["max_err"]
    
    # ----------------------------------------------------
    # 1) 2x2 Similarity Profiles Fit Plot (Task 1)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # Panel (0,0): Temperature
    ax = axes[0, 0]
    ax.plot(y_valid, T_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, T_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Temperature", fontsize=13, fontweight='bold')
    lbl_T = f"$T(y) \\approx [(1-y)(1+{popt_T[0]:.5f}y)]^{{10/39}}$\nAvg Err: {avg_T:.4f}%, Max Err: {max_T:.4f}%"
    ax.text(0.05, 0.05, lbl_T, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (0,1): Density
    ax = axes[0, 1]
    ax.plot(y_valid, rho_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, rho_fit, 'r--', label='EOS Derived Fit', lw=1.5)
    ax.set_ylabel(r"Density $\rho(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Density", fontsize=13, fontweight='bold')
    lbl_rho = r"$\rho(y) \approx \left(\frac{r \cdot T(y)^\beta}{P(y)}\right)^{\frac{1}{\mu - 1}}$" + f"\nAvg Err: {avg_rho:.4f}%, Max Err: {max_rho:.4f}%"
    ax.text(0.05, 0.05, lbl_rho, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,0): Pressure
    ax = axes[1, 0]
    ax.plot(y_valid, P_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, P_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Pressure $P(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Pressure", fontsize=13, fontweight='bold')
    lbl_P = f"$P(y) \\approx {popt_P[0]:.5f} y^{{{popt_P[2]:.5f}}} + {popt_P[1]:.5f} y^{{{popt_P[2]+popt_P[3]:.5f}}}$\nAvg Err: {avg_P:.4f}%, Max Err: {max_P:.4f}%"
    ax.text(0.05, 0.70, lbl_P, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,1): Velocity (using optimal selected fit)
    ax = axes[1, 1]
    ax.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, U_fit, 'r--', label='Optimized Fit', lw=1.5)
    ax.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax.set_title(f"Subsonic: Velocity ({best_u['name']})", fontsize=13, fontweight='bold')
    
    # Format the dynamic velocity formula in latex
    if best_u["id"] == 1:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1-y)^{{{best_u['popt'][1]:.5f}}}$"
    elif best_u["id"] == 2:
        u_formula = f"$U(y) \\approx \\frac{{{best_u['popt'][0]:.5f} (1-y)}}{{1 + {best_u['popt'][1]:.5f} y}}$"
    elif best_u["id"] == 3:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1-y) y^{{-{best_u['popt'][1]:.5f}}}$"
    elif best_u["id"] == 4:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1-y^{{{best_u['popt'][1]:.5f}}}) y^{{-{best_u['popt'][2]:.5f}}}$"
    elif best_u["id"] == 5:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1-y) (y^{{{best_u['popt'][1]:.5f}}} + y^{{-{best_u['popt'][2]:.5f}}})$"
    elif best_u["id"] == 6:
        u_formula = f"$U(y) \\approx \\frac{{{best_u['popt'][0]:.5f} (1-y^{{{best_u['popt'][1]:.5f}}} )}}{{1 + {best_u['popt'][2]:.5f} y}}$"
    
    lbl_U = u_formula + f"\nAvg Err: {avg_U:.4f}%, Max Err: {max_U:.4f}%"
    ax.text(0.05, 0.70, lbl_U, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    for ax in axes.flat:
        ax.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=9.5)
        
    fig.suptitle(f"Subsonic self-similar Profiles & Analytical Fits\n{case_title}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(self_similar_path, dpi=200)
    plt.close(fig)
    print(f"Saved self-similar subsonic profiles to {self_similar_path}")
    
    # ----------------------------------------------------
    # 2) Profiles Combined Relative Error Plot (Task 2)
    # ----------------------------------------------------
    fig_err, ax_err = plt.subplots(figsize=(10, 7.5))
    ax_err.plot(y_bulk, err_T, label=f'Temperature $T(y)$ (Avg: {avg_T:.4f}%, Max: {max_T:.4f}%)', lw=2.0)
    ax_err.plot(y_bulk, err_rho, label=f'Density $\\rho(y)$ (Avg: {avg_rho:.4f}%, Max: {max_rho:.4f}%)', lw=2.0)
    ax_err.plot(y_valid, err_P, label=f'Pressure $P(y)$ (Avg: {avg_P:.4f}%, Max: {max_P:.4f}%)', lw=2.0)
    ax_err.plot(y_bulk, err_U, label=f'Velocity $U(y)$ (Avg: {avg_U:.4f}%, Max: {max_U:.4f}%)', lw=2.0)
    
    ax_err.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_err.set_ylabel(r"Relative Error [$\%$]", fontsize=12)
    ax_err.set_yscale('log')
    ax_err.grid(True, which="both", ls=":", alpha=0.5)
    ax_err.legend(loc="best", fontsize=10.5)
    ax_err.set_title(f"Relative Errors of Subsonic self-similar Fits (semi-log)\n{case_title}", fontsize=13, fontweight='bold')
    
    fig_err.tight_layout()
    fig_err.savefig(relative_errors_path, dpi=200)
    plt.close(fig_err)
    print(f"Saved subsonic relative errors to {relative_errors_path}")
    
    # ----------------------------------------------------
    # 3) Velocity Standalone Fits Comparison Plot (Task 4)
    # ----------------------------------------------------
    fig_sa, (ax_sa1, ax_sa2) = plt.subplots(1, 2, figsize=(18, 8.5))
    
    # Left: Fits vs Numerical
    ax_sa1.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=3.0)
    colors_u = {1: 'crimson', 2: 'darkorange', 3: 'forestgreen', 4: 'darkviolet', 5: 'deeppink', 6: 'teal'}
    
    for i in range(1, 7):
        popt, U_fit, avg_err, max_err, name, latex = fits_u[i]
        if popt is not None:
            lbl = f"Fit {i}: {name}\nAvg Err: {avg_err:.3f}%, Max Err: {max_err:.3f}%"
            lw = 2.2 if i == best_u["id"] else 1.5
            ax_sa1.plot(y_valid, U_fit, colors_u[i], linestyle='--', label=lbl, lw=lw)
            
            # Right: Semi-log Error curves
            err_curve = np.abs((U_fit[mask_bulk] - U_valid[mask_bulk]) / (U_valid[mask_bulk] + 1e-15)) * 100
            ax_sa2.plot(y_bulk, err_curve, colors_u[i], label=f"Fit {i} (Avg: {avg_err:.3f}%)", lw=lw)
            
    ax_sa1.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_sa1.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax_sa1.legend(loc='best', fontsize=9.0)
    ax_sa1.grid(True, alpha=0.3)
    ax_sa1.set_title("Dimensionless Velocity $U(y)$ vs 6 Candidates", fontsize=13, fontweight='bold')
    
    ax_sa2.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_sa2.set_ylabel(r"Relative Error [$\%$]", fontsize=12)
    ax_sa2.set_yscale('log')
    ax_sa2.legend(loc='best', fontsize=9.5)
    ax_sa2.grid(True, which="both", ls=":", alpha=0.5)
    ax_sa2.set_title("Relative Errors of Velocity Fits (semi-log)", fontsize=13, fontweight='bold')
    
    fig_sa.suptitle(f"Subsonic Velocity Profile Curve Fitting & Optimization\nChosen Formal Fit: Fit {best_u['id']} ({best_u['name']})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_sa.savefig(standalone_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sa)
    print(f"Saved standalone velocity fits to {standalone_path}")

def save_custom_evolution_gif(
    history,
    case,
    shussman_ref,
    menahem_ref,
    gif_path: str,
    fps: int = 12,
    stride: int = 1,
    subtitle: str | None = None,
):
    """Generate animated 3-way GIF comparing Simulation, Shussman reference, and Menahem reference."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    p_scale, u_scale, e_scale = 1e12, 1e5, 1e9
    has_T_material = hasattr(history, "T_material") and history.T_material is not None
    
    fig, axes = _create_7panel_vertical_figure()
    k0 = 0
    m0 = history.m[k0] if hasattr(history, "m") else history.x[k0]
    
    # 1) Simulation lines (blue solid)
    sim_lines = []
    sim_lines.append(axes[0].plot(m0, history.rho[k0], lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[1].plot(m0, history.p[k0] / p_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[2].plot(m0, history.u[k0] / u_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[3].plot(m0, history.e[k0] / e_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[4].plot(m0, history.T_material[k0] if has_T_material else history.T[k0], lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[5].plot(m0, history.T[k0], lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[6].plot(m0, history.E_rad[k0], lw=2, label="simulation", color="blue")[0])
    
    # 2) Shussman reference lines (green dashed)
    sh_lines = []
    sh_lines.append(axes[0].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[1].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[2].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[3].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[4].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[5].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[6].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    
    # 3) Menahem reference lines (red dashed)
    men_lines = []
    men_lines.append(axes[0].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[1].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[2].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[3].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[4].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[5].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[6].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        
    x_mass = r"Mass coordinate $m$ [g/cm²]"
    axes[0].set_ylabel(r"$\rho$ [g/cm³]")
    axes[1].set_ylabel(r"$P$ [MBar]")
    axes[2].set_ylabel(r"$u$ [km/s]")
    axes[3].set_ylabel(r"$e_{\mathrm{mat}}$ [$10^9$ erg/g]")
    axes[4].set_ylabel(r"$T_{\mathrm{mat}}$ [K]")
    axes[5].set_ylabel(r"$T_{\mathrm{rad}}$ [K]")
    axes[6].set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    for ax in axes:
        ax.set_xlabel(x_mass)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.3)
        
    title = fig.suptitle("", fontweight="medium")
    frame_ids = np.arange(0, len(history.t), stride)
    
    def init():
        return sim_lines + sh_lines + men_lines
        
    def update(frame_idx):
        k = int(frame_ids[frame_idx])
        mk = history.m[k] if hasattr(history, "m") else history.x[k]
        t = history.t[k]
        sim_lines[0].set_data(mk, history.rho[k])
        sim_lines[1].set_data(mk, history.p[k] / p_scale)
        sim_lines[2].set_data(mk, history.u[k] / u_scale)
        sim_lines[3].set_data(mk, history.e[k] / e_scale)
        sim_lines[4].set_data(mk, history.T_material[k] if has_T_material else history.T[k])
        sim_lines[5].set_data(mk, history.T[k])
        sim_lines[6].set_data(mk, history.E_rad[k])
        
        if shussman_ref is not None:
            sh_idx = int(np.argmin(np.abs(shussman_ref.times - t)))
            mr = shussman_ref.m[sh_idx]
            sh_lines[0].set_data(mr, shussman_ref.rho[sh_idx])
            sh_lines[1].set_data(mr, shussman_ref.p[sh_idx] / p_scale)
            sh_lines[2].set_data(mr, shussman_ref.u[sh_idx] / u_scale)
            sh_lines[3].set_data(mr, shussman_ref.e[sh_idx] / e_scale)
            T_sh_K = (shussman_ref.T[sh_idx] * KELVIN_PER_HEV) if (shussman_ref.T and sh_idx < len(shussman_ref.T)) else np.array([])
            sh_lines[4].set_data(mr, T_sh_K)
            sh_lines[5].set_data(mr, T_sh_K)
            E_sh = shussman_ref.E_rad[sh_idx] if (shussman_ref.E_rad and sh_idx < len(shussman_ref.E_rad)) else np.array([])
            sh_lines[6].set_data(mr, E_sh)
            
        if menahem_ref is not None:
            men_idx = int(np.argmin(np.abs(menahem_ref.times - t)))
            mm = menahem_ref.m[men_idx]
            men_lines[0].set_data(mm, menahem_ref.rho[men_idx])
            men_lines[1].set_data(mm, menahem_ref.p[men_idx] / p_scale)
            men_lines[2].set_data(mm, menahem_ref.u[men_idx] / u_scale)
            men_lines[3].set_data(mm, menahem_ref.e[men_idx] / e_scale)
            T_men_rad_K = (menahem_ref.T[men_idx] * KELVIN_PER_HEV) if (menahem_ref.T and men_idx < len(menahem_ref.T)) else np.array([])
            if hasattr(menahem_ref, "T_material") and menahem_ref.T_material:
                T_men_mat_K = (menahem_ref.T_material[men_idx] * KELVIN_PER_HEV)
            else:
                T_men_mat_K = T_men_rad_K
            men_lines[4].set_data(mm, T_men_mat_K)
            men_lines[5].set_data(mm, T_men_rad_K)
            E_men = menahem_ref.E_rad[men_idx] if (menahem_ref.E_rad and men_idx < len(menahem_ref.E_rad)) else np.array([])
            men_lines[6].set_data(mm, E_men)
            
        case_title = case.title if hasattr(case, "title") and case.title else "Simulation"
        header = f"{case_title}\n{subtitle}" if subtitle else case_title
        if case.T0_Kelvin is not None and case.tau is not None:
            title.set_text(
                f"{header}\n"
                f"$T(0,t)=T_0 t^{{\\tau}},\\; T_0={case.T0_Kelvin},\\; \\tau={case.tau},\\; t={t:.3e}$ s"
            )
        elif getattr(case, "P0", None) is not None and getattr(case, "tau", None) is not None:
            title.set_text(
                f"{header}\n"
                f"$P(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0_Barye},\\; \\tau={case.tau},\\; t={t:.3e}$ s"
            )
        else:
            title.set_text(f"{header}\n$t={t:.3e}$ s")
            
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        return sim_lines + sh_lines + men_lines
        
    anim = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=False)
    
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved custom GIF to {gif_path}")


def save_custom_euler_evolution_gif(
    history,
    case,
    shussman_ref,
    menahem_ref,
    gif_path: str,
    fps: int = 12,
    stride: int = 1,
    subtitle: str | None = None,
    ablation_solver=None,
):
    """Generate animated 3-way GIF in shifted Eulerian coordinates with front predictions."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    p_scale, u_scale, e_scale = 1e12, 1e5, 1e9
    has_T_material = hasattr(history, "T_material") and history.T_material is not None
    
    fig, axes = _create_7panel_vertical_figure()
    k0 = 0
    
    # Grid construction for fronts
    n_cells = len(history.rho[0])
    mass_grid = _build_mass_grid(case, num_cells=n_cells)
    if ablation_solver is None:
        from ablation_solver import AblationSolver
        ablation_solver = AblationSolver(**_ablation_kwargs_from_case(case))
    
    # 1) Simulation lines (blue solid)
    # Shift simulation cell coordinates so boundary is at x=0
    shift_sim = history.x[k0, 0] - history.m[k0, 0] / (2.0 * history.rho[k0, 0])
    xk0 = history.x[k0] - shift_sim
    
    sim_lines = []
    sim_lines.append(axes[0].plot(xk0, history.rho[k0], lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[1].plot(xk0, history.p[k0] / p_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[2].plot(xk0, history.u[k0] / u_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[3].plot(xk0, history.e[k0] / e_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[4].plot(xk0, history.T_material[k0] if has_T_material else history.T[k0], lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[5].plot(xk0, history.T[k0], lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[6].plot(xk0, history.E_rad[k0], lw=2, label="simulation", color="blue")[0]               )
    
    # 2) Shussman reference lines (green dashed)
    sh_lines = []
    sh_lines.append(axes[0].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[1].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[2].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[3].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[4].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[5].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    sh_lines.append(axes[6].plot([], [], lw=1.5, color="green", linestyle="--", label="Shussman")[0])
    
    # 3) Menahem reference lines (red dashed)
    men_lines = []
    men_lines.append(axes[0].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[1].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[2].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[3].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[4].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[5].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[6].plot([], [], lw=1.5, color="red", linestyle="--", label="Menahem")[0])
    
    # 4) Front vertical lines
    bnd_vlines = []
    abl_vlines = []
    sh_vlines = []
    for i, ax in enumerate(axes):
        lbl_bnd = "Boundary Front" if i == 0 else None
        lbl_abl = "Ablation Front" if i == 0 else None
        lbl_sh = "Shock Front" if i == 0 else None
        bnd_vlines.append(ax.axvline(0.0, color="black", linestyle="-", alpha=0.5, label=lbl_bnd))
        abl_vlines.append(ax.axvline(0.0, color="purple", linestyle="--", alpha=0.7, label=lbl_abl))
        sh_vlines.append(ax.axvline(0.0, color="red", linestyle="--", alpha=0.7, label=lbl_sh))
        
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        
    x_euler = r"Eulerian coordinate $x$ [cm] (shifted)"
    axes[0].set_ylabel(r"$\rho$ [g/cm³]")
    axes[1].set_ylabel(r"$P$ [MBar]")
    axes[2].set_ylabel(r"$u$ [km/s]")
    axes[3].set_ylabel(r"$e_{\mathrm{mat}}$ [$10^9$ erg/g]")
    axes[4].set_ylabel(r"$T_{\mathrm{mat}}$ [K]")
    axes[5].set_ylabel(r"$T_{\mathrm{rad}}$ [K]")
    axes[6].set_ylabel(r"$E_{\mathrm{rad}}$ [erg/cm³]")
    for ax in axes:
        ax.set_xlabel(x_euler)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.3)
        
    title = fig.suptitle("", fontweight="medium")
    frame_ids = np.arange(0, len(history.t), stride)
    
    def init():
        return sim_lines + sh_lines + men_lines + bnd_vlines + abl_vlines + sh_vlines
        
    def update(frame_idx):
        k = int(frame_ids[frame_idx])
        t = history.t[k]
        
        # Shift simulation cell coordinates so boundary is at x=0
        shift_sim = history.x[k, 0] - history.m[k, 0] / (2.0 * history.rho[k, 0])
        xk = history.x[k] - shift_sim
        
        sim_lines[0].set_data(xk, history.rho[k])
        sim_lines[1].set_data(xk, history.p[k] / p_scale)
        sim_lines[2].set_data(xk, history.u[k] / u_scale)
        sim_lines[3].set_data(xk, history.e[k] / e_scale)
        sim_lines[4].set_data(xk, history.T_material[k] if has_T_material else history.T[k])
        sim_lines[5].set_data(xk, history.T[k])
        sim_lines[6].set_data(xk, history.E_rad[k])
        
        if shussman_ref is not None:
            sh_idx = int(np.argmin(np.abs(shussman_ref.times - t)))
            xr = shussman_ref.x[sh_idx]
            sh_lines[0].set_data(xr, shussman_ref.rho[sh_idx])
            sh_lines[1].set_data(xr, shussman_ref.p[sh_idx] / p_scale)
            sh_lines[2].set_data(xr, shussman_ref.u[sh_idx] / u_scale)
            sh_lines[3].set_data(xr, shussman_ref.e[sh_idx] / e_scale)
            T_sh_K = (shussman_ref.T[sh_idx] * KELVIN_PER_HEV) if (shussman_ref.T and sh_idx < len(shussman_ref.T)) else np.array([])
            sh_lines[4].set_data(xr, T_sh_K)
            sh_lines[5].set_data(xr, T_sh_K)
            E_sh = shussman_ref.E_rad[sh_idx] if (shussman_ref.E_rad and sh_idx < len(shussman_ref.E_rad)) else np.array([])
            sh_lines[6].set_data(xr, E_sh)
            
        if menahem_ref is not None:
            men_idx = int(np.argmin(np.abs(menahem_ref.times - t)))
            xm = menahem_ref.x[men_idx]
            men_lines[0].set_data(xm, menahem_ref.rho[men_idx])
            men_lines[1].set_data(xm, menahem_ref.p[men_idx] / p_scale)
            men_lines[2].set_data(xm, menahem_ref.u[men_idx] / u_scale)
            men_lines[3].set_data(xm, menahem_ref.e[men_idx] / e_scale)
            T_men_rad_K = (menahem_ref.T[men_idx] * KELVIN_PER_HEV) if (menahem_ref.T and men_idx < len(menahem_ref.T)) else np.array([])
            if hasattr(menahem_ref, "T_material") and menahem_ref.T_material:
                T_men_mat_K = (menahem_ref.T_material[men_idx] * KELVIN_PER_HEV)
            else:
                T_men_mat_K = T_men_rad_K
            men_lines[4].set_data(xm, T_men_mat_K)
            men_lines[5].set_data(xm, T_men_rad_K)
            E_men = menahem_ref.E_rad[men_idx] if (menahem_ref.E_rad and men_idx < len(menahem_ref.E_rad)) else np.array([])
            men_lines[6].set_data(xm, E_men)
            
        # 4) Front predictions via AblationSolver
        try:
            sol = ablation_solver.solve(mass=mass_grid, time=max(float(t), 1e-18))
            x_bnd = 0.00
            x_abl = sol["heat_position"] - sol["boundary_position"]
            x_shock = sol["shock_position"] - sol["boundary_position"]
        except Exception:
            x_bnd = 0.0
            x_abl = np.nan
            x_shock = np.nan
            
        for i, ax in enumerate(axes):
            bnd_vlines[i].set_xdata([x_bnd, x_bnd])
            abl_vlines[i].set_xdata([x_abl, x_abl])
            sh_vlines[i].set_xdata([x_shock, x_shock])
            
        case_title = case.title if hasattr(case, "title") and case.title else "Simulation"
        header = f"{case_title}\n{subtitle}" if subtitle else case_title
        if case.T0_Kelvin is not None and case.tau is not None:
            title.set_text(
                f"{header}\n"
                f"$T(0,t)=T_0 t^{{\\tau}},\\; T_0={case.T0_Kelvin},\\; \\tau={case.tau},\\; t={t:.3e}$ s"
            )
        elif getattr(case, "P0", None) is not None and getattr(case, "tau", None) is not None:
            title.set_text(
                f"{header}\n"
                f"$P(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0_Barye},\\; \\tau={case.tau},\\; t={t:.3e}$ s"
            )
        else:
            title.set_text(f"{header}\n$t={t:.3e}$ s")
            
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        return sim_lines + sh_lines + men_lines + bnd_vlines + abl_vlines + sh_vlines
        
    anim = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=False)
    
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved custom Eulerian GIF to {gif_path}")


# =============================================================================
# Main Orchestration Loop
# =============================================================================

def run_simulation_and_references(preset_name: str, case_label: str):
    """Run full simulation and build reference solvers, or load from pickle cache."""
    import pickle
    
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_label}_cache.pkl"
    
    # 1) Initialize case and set grid size N=400
    case, config = get_preset(preset_name)
    config = replace(config, N=400)
    
    if cache_path.exists():
        print(f"Loading cached simulation and reference data from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            # Unpack cached objects
            history = data["history"]
            shussman_ref = data["shussman_ref"]
            menahem_ref = data["menahem_ref"]
            ablation_solver = data["ablation_solver"]
            print("Cache loaded successfully.")
            return case, history, shussman_ref, menahem_ref, ablation_solver
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running simulation...")
    # Pre-calculate evaluation times
    eval_times_sec = np.linspace(1e-12, case.t_sec_end, 50)
    
    # 4) Build Menahem ablation piecewise reference
    print(f"Building Menahem piecewise reference...")
    menahem_times_sec = np.array([0.25, 0.5, 0.75, 1.0]) * float(case.t_sec_end)
    menahem_ref = run_menahem_piecewise_reference(case, times_sec=menahem_times_sec)
        
    # 2) Run radiation-hydrodynamics simulation
    print(f"Running simulation...")
    _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)
    
    # 3) Build Shussman piecewise reference (seconds / nanoseconds coordinate transformation)
    print(f"Building Shussman piecewise reference...")
    eval_times_ns = eval_times_sec * 1e9
    T0_HeV = float(case.T0_Kelvin) / KELVIN_PER_HEV
    shussman_ref = run_shussman_piecewise_reference(case, times_ns=eval_times_ns, T0_HeV=T0_HeV)
    
    
    # 5) Instantiate AblationSolver to reuse and cache
    print(f"Instantiating AblationSolver reference...")
    ablation_solver = AblationSolver(**_ablation_kwargs_from_case(case))
    
    # Cache everything
    cache_data = {
        "case": case,
        "history": history,
        "shussman_ref": shussman_ref,
        "menahem_ref": menahem_ref,
        "ablation_solver": ablation_solver,
    }
    
    print(f"Saving simulation and reference data cache to {cache_path}...")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Cache saved successfully.")
    except Exception as e:
        print(f"Failed to save cache: {e}")
        
    return case, history, shussman_ref, menahem_ref, ablation_solver


def generate_verification_plots(
    history,
    case,
    shussman_ref,
    menahem_ref,
    ablation_solver,
    case_label: str,
    case_title: str,
):
    """Generate all verification comparisons, plotting, fitting, and exports."""
    # Create results output directory structure
    out_dir = Path("results/ictt")
    xt_dir = out_dir / "xt"
    mh_dir = out_dir / "material_hydro"
    rh_dir = out_dir / "rad_hydro"
    ss_dir = out_dir / "self_similar"
    ev_dir = out_dir / "evolution"
    ev_euler_dir = out_dir / "evolution with euler predictions"
    
    for d in [xt_dir, mh_dir, rh_dir, ss_dir, ev_dir, ev_euler_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    xt_path = str(xt_dir / f"{case_label}_xt.png")
    material_hydro_path = str(mh_dir / f"{case_label}_material_hydro.png")
    rad_hydro_path = str(rh_dir / f"{case_label}_rad_hydro.png")
    self_similar_path = str(ss_dir / f"{case_label}_self_similar.png")
    standalone_path = str(ss_dir / f"{case_label}_velocity_fits_standalone.png")
    relative_errors_path = str(ss_dir / f"{case_label}_relative_errors.png")
    gif_path = str(ev_dir / f"{case_label}_evolution.gif")
    gif_euler_path = str(ev_euler_dir / f"{case_label}_euler_evolution.gif")
    
    # Solve subsonic similarity ODEs once (or load from cache) to share fits among all plots
    solver = get_cached_subsonic_solver(case, case_label)
    _, _, _, _, _, _, _, popt_T, popt_P, best_u, _ = perform_subsonic_fitting(solver)
    
    # 1) Generate space-time trajectories and fronts plot (reusing ablation_solver)
    plot_xt_trajectories(history, case, xt_path, case_title, ablation_solver=ablation_solver)
    
    # 2) Plot material hydrodynamics profiles comparing Sim, Solver, and Fits
    plot_material_hydro_profiles(history, solver, popt_T, popt_P, best_u, material_hydro_path, case_title)
    
    # 3) Plot radiation-hydrodynamics profiles
    plot_rad_hydro_profiles(history, shussman_ref, menahem_ref, rad_hydro_path, case_title)
    
    # 4) Plot similarity profiles and errors
    plot_and_fit_self_similar(solver, self_similar_path, standalone_path, relative_errors_path, case_title)
    
    # 5) Generate evolution GIFs (conditionally)
    if RUN_GIFS:
        print(f"Saving animated custom 3-way evolution GIF to {gif_path}...")
        save_custom_evolution_gif(
            history=history,
            case=case,
            shussman_ref=shussman_ref,
            menahem_ref=menahem_ref,
            gif_path=gif_path,
            fps=12,
            stride=max(1, len(history.t) // 60),
            subtitle="Simulation vs Shussman vs Menahem reference",
        )
        
        print(f"Saving animated custom Eulerian GIF with front predictions to {gif_euler_path}...")
        save_custom_euler_evolution_gif(
            history=history,
            case=case,
            shussman_ref=shussman_ref,
            menahem_ref=menahem_ref,
            gif_path=gif_euler_path,
            fps=12,
            stride=max(1, len(history.t) // 60),
            subtitle="Simulation vs Shussman vs Menahem reference (Eulerian coordinates)",
            ablation_solver=ablation_solver,
        )
    else:
        print("Skipping evolution GIFs generation (RUN_GIFS is set to False)...")


def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run full verification comparison pipeline for a given preset."""
    print("=" * 80)
    print(f"PROCESSING PRESET: {preset_name} -> {case_label}")
    print("=" * 80)
    
    # call run_simulation_and_references only if cache file doesn't exist
    cache_path = Path("results/ictt/cache") / f"{case_label}_cache.pkl"
    if not cache_path.exists():
        print("Pickle file doesn't exist, running simulation...")
        case, history, shussman_ref, menahem_ref, ablation_solver = run_simulation_and_references(
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
        shussman_ref = data["shussman_ref"]
        menahem_ref = data["menahem_ref"]
        ablation_solver = data["ablation_solver"]
    
    generate_verification_plots(
        history=history,
        case=case,
        shussman_ref=shussman_ref,
        menahem_ref=menahem_ref,
        ablation_solver=ablation_solver,
        case_label=case_label,
        case_title=case_title,
    )
    print(f"Preset {preset_name} processed successfully.")


def main():
    # Process constant boundary temperature (Fig 8)
    run_preset_workflow(
        PRESET_FIG_8_CONSTANT_TEMPERATURE, 
        "constant_boundary_temperature_tau_0", 
        "Constant Boundary Temperature (tau=0)"
    )
    # run_preset_workflow(
    #     PRESET_FIG_9_CONSTANT_FLUX, 
    #     "constant_flux_temperature_tau_0_123", 
    #     "Constant Flux Temperature (tau=0.123)"
    # )
    print("\nAll custom simulations, reference comparisons, plotting, fitting, and exports completed successfully!")


if __name__ == "__main__":
    main()

# ictt_other_presets/full_fitting_eulerian.py
"""
Unified patched ablation and shock fitting verification script in Eulerian coordinates.

Combines:
1. 1D Rad-Hydro Simulation (cell boundaries mapped to cell-center positions).
2. AblationSolver patched reference solver (heat wave + piston shock) in Eulerian frame.
3. Patched self-similar analytical fits dynamically mapped to Eulerian coordinate.
"""
from __future__ import annotations

import os
import sys
sys.setrecursionlimit(10000)
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

_ICTT_OTHER_DIR = Path(__file__).resolve().parent
if str(_ICTT_OTHER_DIR) not in sys.path:
    sys.path.insert(0, str(_ICTT_OTHER_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_9_CONSTANT_FLUX,
    PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
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
    
    # Scale mass_grid for Fig 10 compatibility
    m_f_max = ablation_solver.heat_solver.ablated_mass(time=times[-1])
    m_max_val = max(mass_grid[-1], 2.0 * m_f_max)
    mass_grid = mass_grid * (m_max_val / mass_grid[-1])
    
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
    ax.plot(times[1:] * 1e9, x_shock_sim, lw=2.5, c="red", label="Shock (simulation)")
    ax.plot(times_model * 1e9, shock_position, lw=2.0, ls="--", c="darkred", label="Shock (Menahem)")
    ax.plot(times_model * 1e9, piston_position, lw=2.0, ls="--", c="green", label="Piston (Menahem)")
    ax.plot(times_model * 1e9, heat_position, lw=2.0, ls="--", c="purple", label="Heat Wave (Menahem)")
    
    ax.set_xlabel(r"$t$ [ns]", fontsize=12)
    ax.set_ylabel(r"$x$ [cm]", fontsize=12)
    ax.set_title(f"Space-Time (xt) Trajectories and Fronts\n{case_title}", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_ylim(0, float(case.x_max) * 1.1)
    
    fig.tight_layout()
    fig.savefig(xt_path, dpi=200)
    plt.close(fig)


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

def get_data(preset_name: str, case_label: str):
    """Run full simulation and build AblationSolver reference solver, or load from cache."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_label}_full_fitting_cache.pkl"

    case, config = get_preset(preset_name)

    if USE_CACHE and cache_path.exists():
        print(f"Loading cached simulation history and solver from {cache_path}...")
        try:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if 'numpy._core' in module:
                        try:
                            return super().find_class(module, name)
                        except (ModuleNotFoundError, ImportError):
                            module = module.replace('numpy._core', 'numpy.core')
                    elif 'numpy.core' in module:
                        try:
                            return super().find_class(module, name)
                        except (ModuleNotFoundError, ImportError):
                            module = module.replace('numpy.core', 'numpy._core')
                    return super().find_class(module, name)
            with open(cache_path, "rb") as f:
                data = CustomUnpickler(f).load()
            solver = data["solver"]
            if solver is not None:
                # Re-bind ODE solvers containing method callbacks
                if hasattr(solver.heat_solver, "fode"):
                    solver.heat_solver.ode_solver = scipy.integrate.ode(solver.heat_solver.fode).set_integrator(solver.heat_solver.ode_scheme)
                if hasattr(solver.shock_solver, "fode"):
                    solver.shock_solver.ode_solver = scipy.integrate.ode(solver.shock_solver.fode).set_integrator(solver.shock_solver.ode_scheme)
            print("Loaded successfully from cache.")
            return data["case"], data["history"], solver
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running simulation...")

    print("Running 1D Rad-Hydro Simulation...")
    config = replace(config, show_plot=False, show_slider=False)
    _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)

    print("Instantiating AblationSolver reference solver...")
    ablation_solver = AblationSolver(**_ablation_kwargs_from_case(case))

    cache_data = {"case": case, "history": history, "solver": ablation_solver}
    # Detach un-picklable ODE solver components temporarily
    heat_ode = getattr(ablation_solver.heat_solver, "ode_solver", None)
    shock_ode = getattr(ablation_solver.shock_solver, "ode_solver", None)
    try:
        if hasattr(ablation_solver.heat_solver, "ode_solver"):
            del ablation_solver.heat_solver.ode_solver
        if hasattr(ablation_solver.shock_solver, "ode_solver"):
            del ablation_solver.shock_solver.ode_solver

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Failed to save simulation cache: {e}")
    finally:
        if heat_ode is not None:
            ablation_solver.heat_solver.ode_solver = heat_ode
        if shock_ode is not None:
            ablation_solver.shock_solver.ode_solver = shock_ode

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

        m_f = hs.ablated_mass(time=t_actual)
        m_s = ss.shocked_mass(time=t_actual)

        # Define grids (natively sized 1000) with quadratic clustering near m=0
        mass_sub = 1e-12 + (m_f - 1e-12) * (np.linspace(0.0, 1.0, 1000) ** 2.0)
        mass_shock = np.linspace(1e-12, max(m_sim[-1], 2.0 * m_f), 1000)

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

        # Analytical subsonic & shock fits
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

        # Plot Simulation (entire domain)
        ax_rho.plot(x_sim_center_um, sim_rho, '-', color='#1a5fb4', lw=2.2)
        ax_p.plot(x_sim_center_um, sim_p, '-', color='#1a5fb4', lw=2.2)
        ax_u.plot(x_sim_center_um, sim_u, '-', color='#1a5fb4', lw=2.2)
        ax_T.plot(x_sim_center_um, sim_T, '-', color='#1a5fb4', lw=2.2)

        # Plot Subsonic exact (solid black)
        ax_rho.plot(exact_sub_x_um, exact_sub_rho, '-', color='#333333', lw=2.0)
        ax_p.plot(exact_sub_x_um, exact_sub_p, '-', color='#333333', lw=2.0)
        ax_u.plot(exact_sub_x_um, exact_sub_u, '-', color='#333333', lw=2.0)
        ax_T.plot(exact_sub_x_um, exact_sub_T, '-', color='#333333', lw=2.0)

        # Plot Subsonic fits (dotted green)
        ax_rho.plot(fit_sub_x_um, fit_sub_rho, ':', color='#26a269', lw=2.0)
        ax_p.plot(fit_sub_x_um, fit_sub_p, ':', color='#26a269', lw=2.0)
        ax_u.plot(fit_sub_x_um, fit_sub_u, ':', color='#26a269', lw=2.0)
        ax_T.plot(fit_sub_x_um, fit_sub_T, ':', color='#26a269', lw=2.0)

        # Plot Shock exact (dashed black)
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
        rhok_smooth = _rolling_mean(sim_rho, 5)
        ishock, _ = find_shock_front(rhok_smooth, m_sim, rho_unshocked=float(case.rho0), gamma=float(case.r) + 1.0)
        x_s_sim = x_sim_center_um[ishock] if ishock >= 1 else np.nan

        # Draw vertical lines for the fronts on all subplots
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
        Line2D([0], [0], color='#1a5fb4', lw=2.5, linestyle='-', label=f'Simulation ({t_target*1e9:.3f} ns)'),
        Line2D([0], [0], color='#333333', lw=2.0, linestyle='-', label='Subsonic Solver'),
        Line2D([0], [0], color='#26a269', lw=2.0, linestyle=':', label='Subsonic Fit'),
        Line2D([0], [0], color='#333333', lw=1.8, linestyle='--', label='Shock Solver'),
        Line2D([0], [0], color='#e67e22', lw=1.8, linestyle='-.', label='Shock Fit'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle='-', label='Simulation Fronts'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle='--', label='Solver Fronts'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle=':', label='Fit Fronts'),
    ]
    ax_rho.legend(handles=legend_handles, loc="best", fontsize=9.5)

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

    labels = ["Density [g/cm³]", "Pressure [MBar]", "Velocity [km/s]", "Temperature [HeV]"]
    for j, ax in enumerate([ax_rho, ax_p, ax_u, ax_T]):
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels[j], fontsize=12)
        ax.set_xlabel("Eulerian Position Coordinate $x$ [$\mu$m]", fontsize=12)

    plt.suptitle(f"Unified Patched Ablation & Shock Verification in Eulerian Coordinate\n{case_title}", fontsize=14, fontweight='bold')
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
):
    """
    Plots a unified 2x2 comparison showing the fully patched (seamless) profiles vs Eulerian coordinate x.
    Compares Rad-Hydro Simulation, AblationSolver (exact patched), and fully patched Fits.
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

        m_f = hs.ablated_mass(time=t_actual)
        m_max_val = max(m_sim[-1], 2.0 * m_f)
        mass_solver = 1e-12 + (m_max_val - 1e-12) * (np.linspace(0.0, 1.0, 1000) ** 2.0)

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

        # Plot Simulation (entire domain)
        ax_rho.plot(x_sim_center_um, sim_rho, '-', color='#1a5fb4', lw=2.2)
        ax_p.plot(x_sim_center_um, sim_p, '-', color='#1a5fb4', lw=2.2)
        ax_u.plot(x_sim_center_um, sim_u, '-', color='#1a5fb4', lw=2.2)
        ax_T.plot(x_sim_center_um, sim_T, '-', color='#1a5fb4', lw=2.2)

        # Plot Exact patched solver (solid black)
        ax_rho.plot(exact_x_um, exact_rho, '-', color='#333333', lw=2.0)
        ax_p.plot(exact_x_um, exact_p, '-', color='#333333', lw=2.0)
        ax_u.plot(exact_x_um, exact_u, '-', color='#333333', lw=2.0)
        ax_T.plot(exact_x_um, exact_T, '-', color='#333333', lw=2.0)

        # Plot Patched fits (dotted orange)
        ax_rho.plot(fit_x_um, fit_rho, ':', color='#e67e22', lw=2.0)
        ax_p.plot(fit_x_um, fit_p, ':', color='#e67e22', lw=2.0)
        ax_u.plot(fit_x_um, fit_u, ':', color='#e67e22', lw=2.0)
        ax_T.plot(fit_x_um, fit_T, ':', color='#e67e22', lw=2.0)

        # Front positions
        heat_position = sol_exact["heat_position"]
        x_b_sol = (sol_exact["boundary_position"] + heat_position) * 1e4
        x_af_sol = heat_position * 1e4
        x_s_sol = sol_exact["shock_position"] * 1e4
        
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
        rhok_smooth = _rolling_mean(sim_rho, 5)
        ishock, _ = find_shock_front(rhok_smooth, m_sim, rho_unshocked=float(case.rho0), gamma=float(case.r) + 1.0)
        x_s_sim = x_sim_center_um[ishock] if ishock >= 1 else np.nan

        # Draw vertical lines for the fronts on all subplots
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
        Line2D([0], [0], color='#1a5fb4', lw=2.5, linestyle='-', label=f'Simulation ({t_target*1e9:.3f} ns)'),
        Line2D([0], [0], color='#333333', lw=2.0, linestyle='-', label='Exact Patched Solver'),
        Line2D([0], [0], color='#e67e22', lw=2.0, linestyle=':', label='Patched Fit'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle='-', label='Simulation Fronts'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle='--', label='Solver Fronts'),
        Line2D([0], [0], color='grey', lw=1.2, linestyle=':', label='Fit Fronts'),
    ]
    ax_rho.legend(handles=legend_handles, loc="best", fontsize=9.5)

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

    labels = ["Density [g/cm³]", "Pressure [MBar]", "Velocity [km/s]", "Temperature [HeV]"]
    for j, ax in enumerate([ax_rho, ax_p, ax_u, ax_T]):
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels[j], fontsize=12)
        ax.set_xlabel("Eulerian Position Coordinate $x$ [$\mu$m]", fontsize=12)

    plt.suptitle(f"Unified Patched Ablation & Shock Verification (Fully Patched) in Eulerian Coordinate\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved fully patched comparison plot in Eulerian to {plot_path}")
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
    
    n_cells = x_sim.shape[1] - 1
    mass_grid = _build_mass_grid(case, num_cells=n_cells)
    
    times_model = _get_equally_spaced_elements(times, 200)
    
    x_shock_sim = np.full(times.size, np.nan, dtype=float)
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
            x_shock_sim[k] = float(x_sim[k, ishock]) * 1e4
            
    x_shock_sim = _rolling_mean(x_shock_sim, 3)
    x_boundary_sim = x_sim[:, 0] * 1e4
    
    x_boundary_sol = np.zeros_like(times_model)
    x_piston_sol = np.zeros_like(times_model)
    x_ablation_front_sol = np.zeros_like(times_model)
    x_shock_sol = np.zeros_like(times_model)
    
    x_boundary_fit = np.zeros_like(times_model)
    x_piston_fit = np.zeros_like(times_model)
    x_ablation_front_fit = np.zeros_like(times_model)
    x_shock_fit = np.zeros_like(times_model)
    
    hs = ablation_solver.heat_solver
    ss = ablation_solver.shock_solver
    q1 = 1.0 - ss.omega
    q2 = (2.0 - ss.omega) / (ss.tau + 2.0)
    
    m_f_max = hs.ablated_mass(time=times_model[-1])
    m_max_val = max(mass_grid[-1], 2.0 * m_f_max)
    mass_grid = mass_grid * (m_max_val / mass_grid[-1])
    
    for i, t in enumerate(times_model):
        t_val = max(float(t), 1e-18)
        sol = ablation_solver.solve(mass=mass_grid, time=t_val)
        
        # Solver positions
        x_boundary_sol[i] = (sol["boundary_position"] + sol["heat_position"]) * 1e4
        x_piston_sol[i] = sol["piston_position"] * 1e4
        x_ablation_front_sol[i] = sol["heat_position"] * 1e4
        x_shock_sol[i] = sol["shock_position"] * 1e4
        
        # Fit positions
        m_f = hs.ablated_mass(time=t_val)
        xsi_mf = m_f * ss.xsi_over_m(time=t_val)
        y_mf = xsi_mf / ss.xsi_s
        _, _, U_fit_sh_mf, rho_fit_sh_mf = shock_fit_by_params(np.array([y_mf]), shock_params)
        V_fit_sh_mf = 1.0 / rho_fit_sh_mf[0]
        
        pos_scale = ss._position_temporal_factor(time=t_val)
        
        # front position (heat wave interface) for fit
        x_af_val = pos_scale * (q1 * xsi_mf * V_fit_sh_mf + q2 * U_fit_sh_mf[0])
        x_ablation_front_fit[i] = x_af_val * 1e4
        
        # shock piston position for fit
        x_piston_fit[i] = pos_scale * q2 * ss.U0 * 1e4
        
        # boundary position for fit
        x_boundary_fit[i] = (hs.boundary_position(time=t_val) + x_af_val) * 1e4
        
        # shock front position for fit
        x_shock_fit[i] = ss.shock_position(time=t_val) * 1e4
        
    fig, ax = plt.subplots(figsize=(10, 7))
    t_ns = times * 1e9
    t_ns_model = times_model * 1e9
    
    # Plot Simulation
    ax.plot(t_ns, x_boundary_sim, '-', color='black', lw=2.5, label=r"$x_{\rm boundary}$ (simulation)")
    ax.plot(t_ns, x_shock_sim, '-', color='red', lw=2.5, label=r"$x_{\rm shock\ front}$ (simulation)")
    
    # Plot Solver
    ax.plot(t_ns_model, x_boundary_sol, '--', color='grey', lw=2.0, label=r"$x_{\rm boundary}$ (solver)")
    ax.plot(t_ns_model, x_piston_sol, '--', color='blue', lw=2.0, label=r"$x_{\rm piston}$ (solver)")
    ax.plot(t_ns_model, x_ablation_front_sol, '--', color='fuchsia', lw=2.0, label=r"$x_{\rm ablation\ front}$ (solver)")
    ax.plot(t_ns_model, x_shock_sol, '--', color='darkred', lw=2.0, label=r"$x_{\rm shock\ front}$ (solver)")
    
    # Plot Fit
    ax.plot(t_ns_model, x_boundary_fit, ':', color='black', lw=2.2, label=r"$x_{\rm boundary}$ (fit)")
    ax.plot(t_ns_model, x_piston_fit, ':', color='cyan', lw=2.2, label=r"$x_{\rm piston}$ (fit)")
    ax.plot(t_ns_model, x_ablation_front_fit, ':', color='purple', lw=2.2, label=r"$x_{\rm ablation\ front}$ (fit)")
    ax.plot(t_ns_model, x_shock_fit, ':', color='orange', lw=2.2, label=r"$x_{\rm shock\ front}$ (fit)")
    
    ax.set_xlabel(r"$t$ [ns]", fontsize=13)
    ax.set_ylabel(r"$x$ [$\mu$m]", fontsize=13)
    ax.set_title(f"Front Trajectory Evolution (Simulation vs Solver vs Fit)\n{case_title}", fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim(0, times[-1] * 1e9)
    
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved front trajectories plot in Eulerian to {plot_path}")
    plt.close(fig)


# =============================================================================
# Main Execution Workflow
# =============================================================================

def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run simulation, load reference solvers, run fitting pipelines, and plot eulerian profiles and trajectories."""
    case, history, ablation_solver = get_data(preset_name, case_label)

    # 1) Run Subsonic fitting pipeline using ablation_solver.heat_solver
    print("--- Running Subsonic Ablation Fitting ---")
    sub_solver = ablation_solver.heat_solver
    sub_params = perform_subsonic_fitting(sub_solver)

    # 2) Run Piston Shock fitting pipeline using ablation_solver.shock_solver
    print("--- Running Piston Shock Fitting ---")
    shock_solver = ablation_solver.shock_solver
    shock_params = perform_shock_fitting(shock_solver)

    case_dir = Path("results/ictt") / case_label
    eulerian_ver_dir = case_dir / "eulerian_verification"
    eulerian_ver_dir.mkdir(parents=True, exist_ok=True)

    # 3) Plot 1: xt space-time trajectories
    plot_path_xt = eulerian_ver_dir / f"{case_label}_xt.png"
    plot_xt_trajectories(
        history=history,
        case=case,
        xt_path=str(plot_path_xt),
        case_title=case_title,
        ablation_solver=ablation_solver,
    )

    # 4) Plot 2: Fully Patched profiles in Eulerian coordinate
    plot_path_patched = eulerian_ver_dir / f"{case_label}_eulerian_comparison.png"
    plot_fully_patched_comparison_eulerian(
        history=history,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
        case=case,
        plot_path=str(plot_path_patched),
        case_title=case_title,
    )
    
    # 5) Plot 3: Time-dependent front trajectories in Eulerian coordinate
    plot_path_trajectories = eulerian_ver_dir / f"{case_label}_front_trajectories.png"
    plot_front_trajectories_eulerian(
        history=history,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
        case=case,
        plot_path=str(plot_path_trajectories),
        case_title=case_title,
    )


def main():
    from project3_code.rad_hydro_sim.problems.presets_config import PRESET_FIG_8_CONSTANT_TEMPERATURE
    run_preset_workflow(
        PRESET_FIG_8_CONSTANT_TEMPERATURE,
        "const_T",
        "Fig 8 Constant Temperature Drive"
    )
    run_preset_workflow(
        PRESET_FIG_9_CONSTANT_FLUX,
        "const_S",
        "Fig 9 Constant Flux Drive"
    )
    run_preset_workflow(
        PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
        "const_P_shock",
        "Fig 10 Constant Ablation Pressure Drive"
    )
    print("\nPatched eulerian ablation and shock simulations, fittings, comparisons, and plots generated successfully!")


if __name__ == "__main__":
    main()

# ictt_other_presets/full_fitting.py
"""
Unified patched ablation and shock fitting verification script.

Combines:
1. 1D Rad-Hydro Simulation.
2. AblationSolver patched reference solver (heat wave + piston shock).
3. Patched self-similar analytical fits dynamically optimized for both regions.
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

# Monkeypatch scipy.integrate.simps and numpy.trapz for compatibility
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None) or scipy.integrate.trapz

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

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset as _orig_get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_9_CONSTANT_FLUX,
    PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
)
def get_preset(preset_name):
    case, config = _orig_get_preset(preset_name)
    if preset_name == PRESET_FIG_9_CONSTANT_FLUX:
        case = replace(case, t_sec_end=1.5e-9, times_for_png=np.array([0.5e-9, 1.0e-9, 1.5e-9], dtype=float))
    elif preset_name == PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE:
        case = replace(case, t_sec_end=1.5e-7, times_for_png=np.array([50e-9, 100e-9, 150e-9], dtype=float))
    return case, config
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _ablation_kwargs_from_case,
    _build_mass_grid,
)
from project3_code.menahem_new.ablation_solver_og import AblationSolver

# Import fitting and dimensional scaling functions from shock and sub fitting scripts
from sub_fitting import perform_subsonic_fitting, calculate_dimensional_fits as calculate_dimensional_fits_sub
from shock_fitting import perform_shock_fitting, calculate_dimensional_fits as calculate_dimensional_fits_shock

USE_CACHE = True  # Set to True to use pre-saved pickle files, False to run again


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
                    import scipy.integrate
                    solver.heat_solver.ode_solver = scipy.integrate.ode(solver.heat_solver.fode).set_integrator(solver.heat_solver.ode_scheme)
                if hasattr(solver.shock_solver, "fode"):
                    import scipy.integrate
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
    print(f"Saving simulation cache to {cache_path}...")
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
    """
    Constructs fully patched, seamless analytical physical profiles.
    For mass <= ablated_mass (m_f), it evaluates the subsonic fits.
    For mass > ablated_mass (m_f), it evaluates the shock fits.
    """
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


def plot_patched_dimensional_fit_comparison(
    history,
    ablation_solver,
    sub_params,
    shock_params,
    case,
    plot_path,
    case_title,
):
    """Plots unified 2x2 CGS overlays showing individual subsonic and shock region profiles."""
    print(f"Generating physical patched profiles comparison for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_max = max(history.t)
    target_times = [0.5 * t_max, 0.75 * t_max, t_max]
    sim_colors = ["royalblue", "darkorange", "crimson"]

    ax_rho = axes[0, 0]
    ax_p = axes[0, 1]
    ax_u = axes[1, 0]
    ax_T = axes[1, 1]

    p_scale = 1e12
    u_scale = 1e5
    T_scale = 1.160451812e6

    for i, (t_target, sim_color) in enumerate(zip(target_times, sim_colors)):
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]

        sim_rho = history.rho[idx_sim]
        sim_p = history.p[idx_sim] / p_scale
        sim_u = history.u[idx_sim] / u_scale
        sim_T = history.T[idx_sim] / T_scale

        m_f = ablation_solver.heat_solver.ablated_mass(time=t_actual)

        mass_sub = np.linspace(1e-12, m_f, 1000)
        mass_shock = np.linspace(1e-12, max(m_sim[-1], 2.0 * m_f), 1000)

        # Solver subsonic exact profiles
        sol_sub = ablation_solver.heat_solver.solve(mass=mass_sub, time=t_actual)
        exact_sub_rho = sol_sub["density"]
        exact_sub_p = sol_sub["pressure"] / p_scale
        exact_sub_u = sol_sub["velocity"] / u_scale
        exact_sub_T = sol_sub["temperature"] / T_scale

        # Solver shock exact profiles
        sol_shock = ablation_solver.shock_solver.solve(mass=mass_shock, time=t_actual)
        exact_shock_rho = sol_shock["density"]
        exact_shock_p = sol_shock["pressure"] / p_scale
        exact_shock_u = sol_shock["velocity"] / u_scale
        exact_shock_T_kelvin = ((sol_shock["pressure"] * sol_shock["density"]**(ablation_solver.mu_shock - 1.0)) / (ablation_solver.shock_solver.r * ablation_solver.f_shock))**(1.0 / ablation_solver.beta_shock)
        exact_shock_T = exact_shock_T_kelvin / T_scale

        # Subsonic Analytical fits mapped to CGS
        fits_sub = calculate_dimensional_fits_sub(mass_sub, t_actual, ablation_solver.heat_solver, sub_params)
        fit_sub_rho = fits_sub["density"]
        fit_sub_p = fits_sub["pressure"] / p_scale
        fit_sub_u = fits_sub["velocity"] / u_scale
        fit_sub_T = fits_sub["temperature"] / T_scale

        # Shock Analytical fits mapped to CGS
        fits_shock = calculate_dimensional_fits_shock(mass_shock, t_actual, ablation_solver.shock_solver, shock_params)
        fit_shock_rho = fits_shock["density"]
        fit_shock_p = fits_shock["pressure"] / p_scale
        fit_shock_u = fits_shock["velocity"] / u_scale
        fit_shock_T = fits_shock["temperature"] / T_scale

        show_label = i == 0

        # Plot Simulation (entire domain)
        ax_rho.plot(m_sim * 1e3, sim_rho, '-', color=sim_color, alpha=0.8, label=f"Simulation ({t_target*1e9:.3f} ns)" if show_label else None)
        ax_p.plot(m_sim * 1e3, sim_p, '-', color=sim_color, alpha=0.8)
        ax_u.plot(m_sim * 1e3, sim_u, '-', color=sim_color, alpha=0.8)
        ax_T.plot(m_sim * 1e3, sim_T, '-', color=sim_color, alpha=0.8)

        # Plot Subsonic exact (solid black)
        ax_rho.plot(mass_sub * 1e3, exact_sub_rho, '-', color='black', lw=2.0, label="Subsonic Solver" if show_label else None)
        ax_p.plot(mass_sub * 1e3, exact_sub_p, '-', color='black', lw=2.0)
        ax_u.plot(mass_sub * 1e3, exact_sub_u, '-', color='black', lw=2.0)
        ax_T.plot(mass_sub * 1e3, exact_sub_T, '-', color='black', lw=2.0)

        # Plot Subsonic fits (solid forestgreen)
        ax_rho.plot(mass_sub * 1e3, fit_sub_rho, '.', color='forestgreen', markersize=2, alpha=0.4, label="Subsonic Fit" if show_label else None)
        ax_p.plot(mass_sub * 1e3, fit_sub_p, '.', color='forestgreen', markersize=2, alpha=0.4)
        ax_u.plot(mass_sub * 1e3, fit_sub_u, '.', color='forestgreen', markersize=2, alpha=0.4)
        ax_T.plot(mass_sub * 1e3, fit_sub_T, '.', color='forestgreen', markersize=2, alpha=0.4)

        # Plot Shock exact (dashed black)
        ax_rho.plot(mass_shock * 1e3, exact_shock_rho, '--', color='black', lw=1.8, label="Shock Solver" if show_label else None)
        ax_p.plot(mass_shock * 1e3, exact_shock_p, '--', color='black', lw=1.8)
        ax_u.plot(mass_shock * 1e3, exact_shock_u, '--', color='black', lw=1.8)
        ax_T.plot(mass_shock * 1e3, exact_shock_T, '--', color='black', lw=1.8)

        # Plot Shock fits (dashed forestgreen)
        ax_rho.plot(mass_shock * 1e3, fit_shock_rho, ':', color='forestgreen', lw=1.5, alpha=0.7, label="Shock Fit" if show_label else None)
        ax_p.plot(mass_shock * 1e3, fit_shock_p, ':', color='forestgreen', lw=1.5, alpha=0.7)
        ax_u.plot(mass_shock * 1e3, fit_shock_u, ':', color='forestgreen', lw=1.5, alpha=0.7)
        ax_T.plot(mass_shock * 1e3, fit_shock_T, ':', color='forestgreen', lw=1.5, alpha=0.7)

    # Build time legend entries
    time_handles = [
        Line2D([0], [0], color=sim_colors[k], lw=2, label=f"{target_times[k]*1e9:.3f} ns")
        for k in range(len(target_times))
    ]
    style_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Subsonic Solver'),
        Line2D([0], [0], color='forestgreen', lw=2, linestyle=':', label='Subsonic Fit'),
        Line2D([0], [0], color='black', lw=1.8, linestyle='--', label='Shock Solver'),
        Line2D([0], [0], color='forestgreen', lw=1.8, linestyle='--', label='Shock Fit'),
    ]
    ax_rho.legend(handles=time_handles + style_handles, loc="best", fontsize=9.5)

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
        ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm²]", fontsize=12)

    plt.suptitle(f"Unified Patched Ablation & Shock Verification (Region Overlays)\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved region overlays plot to {plot_path}")
    plt.close(fig)


def plot_fully_patched_comparison(
    history,
    ablation_solver,
    sub_params,
    shock_params,
    case,
    plot_path,
    case_title,
):
    """
    Plots a unified 2x2 comparison showing the fully patched (seamless) profiles.
    Compares Rad-Hydro Simulation, AblationSolver (exact patched), and fully patched Fits.
    """
    print(f"Generating physical fully patched comparison for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_max = max(history.t)
    target_times = [0.5 * t_max, 0.75 * t_max, t_max]
    sim_colors = ["royalblue", "darkorange", "crimson"]

    ax_rho = axes[0, 0]
    ax_p = axes[0, 1]
    ax_u = axes[1, 0]
    ax_T = axes[1, 1]

    p_scale = 1e12
    u_scale = 1e5
    T_scale = 1.160451812e6

    for i, (t_target, sim_color) in enumerate(zip(target_times, sim_colors)):
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]

        sim_rho = history.rho[idx_sim]
        sim_p = history.p[idx_sim] / p_scale
        sim_u = history.u[idx_sim] / u_scale
        sim_T = history.T[idx_sim] / T_scale
        m_f = ablation_solver.heat_solver.ablated_mass(time=t_actual)
        mass_solver = np.linspace(1e-12, max(m_sim[-1], 2.0 * m_f), 1000)

        # Solve fully patched exact profiles from AblationSolver
        sol_exact = ablation_solver.solve(mass=mass_solver, time=t_actual)
        exact_rho = sol_exact["density"]
        exact_p = sol_exact["pressure"] / p_scale
        exact_u = sol_exact["velocity"] / u_scale
        exact_T = sol_exact["temperature"] / T_scale

        # Solve fully patched Fits profiles
        fits = calculate_patched_dimensional_fits(mass_solver, t_actual, ablation_solver, sub_params, shock_params)
        fit_rho = fits["density"]
        fit_p = fits["pressure"] / p_scale
        fit_u = fits["velocity"] / u_scale
        fit_T = fits["temperature"] / T_scale

        show_label = i == 0

        # Plot Simulation (entire domain)
        ax_rho.plot(m_sim * 1e3, sim_rho, '-', color=sim_color, alpha=0.8, label=f"Simulation ({t_target*1e9:.3f} ns)" if show_label else None)
        ax_p.plot(m_sim * 1e3, sim_p, '-', color=sim_color, alpha=0.8)
        ax_u.plot(m_sim * 1e3, sim_u, '-', color=sim_color, alpha=0.8)
        ax_T.plot(m_sim * 1e3, sim_T, '-', color=sim_color, alpha=0.8)

        # Plot Exact patched solver (solid black)
        ax_rho.plot(mass_solver * 1e3, exact_rho, '-', color='black', lw=2.0, label="Exact Patched Solver" if show_label else None)
        ax_p.plot(mass_solver * 1e3, exact_p, '-', color='black', lw=2.0)
        ax_u.plot(mass_solver * 1e3, exact_u, '-', color='black', lw=2.0)
        ax_T.plot(mass_solver * 1e3, exact_T, '-', color='black', lw=2.0)

        # Plot Patched fits (dotted forestgreen)
        ax_rho.plot(mass_solver * 1e3, fit_rho, ':', color='forestgreen', lw=1.8, label="Patched Fit" if show_label else None)
        ax_p.plot(mass_solver * 1e3, fit_p, ':', color='forestgreen', lw=1.8)
        ax_u.plot(mass_solver * 1e3, fit_u, ':', color='forestgreen', lw=1.8)
        ax_T.plot(mass_solver * 1e3, fit_T, ':', color='forestgreen', lw=1.8)

    # Build time legend entries
    time_handles = [
        Line2D([0], [0], color=sim_colors[k], lw=2, label=f"{target_times[k]*1e9:.3f} ns")
        for k in range(len(target_times))
    ]
    style_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Exact Patched Solver'),
        Line2D([0], [0], color='forestgreen', lw=1.8, linestyle=':', label='Patched Fit'),
    ]
    ax_rho.legend(handles=time_handles + style_handles, loc="best", fontsize=9.5)

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
        ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm²]", fontsize=12)

    plt.suptitle(f"Unified Patched Ablation & Shock Verification (Fully Patched)\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved fully patched comparison plot to {plot_path}")
    plt.close(fig)


def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run simulation, solve patched similarity models, run fitting pipelines, and plot both formats."""
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
    dv_dir = case_dir / "dimensional_verification"
    dv_dir.mkdir(parents=True, exist_ok=True)

    # Plot: Fully Patched seamless profiles compared to AblationSolver
    plot_path_patched = dv_dir / f"{case_label}_fully_patched_comparison.png"
    plot_fully_patched_comparison(
        history=history,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
        case=case,
        plot_path=str(plot_path_patched),
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
    print("\nPatched ablation and shock simulations, fittings, comparisons, and plots generated successfully!")


if __name__ == "__main__":
    main()

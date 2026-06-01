# ictt29/full_fitting.py
"""
Unified patched ablation and shock fitting verification script.

Combines:
1. 1D Rad-Hydro Simulation.
2. AblationSolver patched reference solver (heat wave + piston shock).
3. Patched self-similar analytical fits dynamically optimized for both regions.

Produces a single, unified CGS physical comparison plot covering both subsonic and shock regions.
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
from project3_code.ictt29.sub_fitting import perform_subsonic_fitting, calculate_dimensional_fits as calculate_dimensional_fits_sub
from project3_code.ictt29.shock_fitting import perform_shock_fitting, calculate_dimensional_fits as calculate_dimensional_fits_shock

USE_CACHE = False  # Set to True to use pre-saved pickle files, False to run again


def get_data():
    """Run full simulation and build AblationSolver reference solver, or load from cache."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "full_fitting_cache.pkl"

    case, config = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)

    if USE_CACHE and cache_path.exists():
        print(f"Loading cached simulation history and solver from {cache_path}...")
        try:
            import sys
            import numpy.core
            sys.modules['numpy._core'] = sys.modules.get('numpy.core')
            sys.modules['numpy._core.numeric'] = sys.modules.get('numpy.core.numeric')
            sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray')
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
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
    try:
        # Detach un-picklable ODE solver components temporarily
        heat_ode = getattr(ablation_solver.heat_solver, "ode_solver", None)
        shock_ode = getattr(ablation_solver.shock_solver, "ode_solver", None)
        if hasattr(ablation_solver.heat_solver, "ode_solver"):
            del ablation_solver.heat_solver.ode_solver
        if hasattr(ablation_solver.shock_solver, "ode_solver"):
            del ablation_solver.shock_solver.ode_solver

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if heat_ode is not None:
            ablation_solver.heat_solver.ode_solver = heat_ode
        if shock_ode is not None:
            ablation_solver.shock_solver.ode_solver = shock_ode
    except Exception as e:
        print(f"Failed to save simulation cache: {e}")

    return case, history, ablation_solver


def plot_patched_dimensional_fit_comparison(
    history,
    ablation_solver,
    sub_params,
    shock_params,
    case,
    plot_path,
    case_title,
):
    """
    Plots a unified 2x2 CGS physical overlay comparison showing Density, Pressure, Velocity, and Temperature.
    Displays Rad-Hydro Simulation, Exact Subsonic & Shock Solver, and dynamically fitted analytical curves.
    """
    print(f"Generating physical patched profiles comparison for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    target_times = [1.0e-9, 1.5e-9, 2.0e-9]
    plasma = plt.get_cmap("plasma")
    sim_colors = [plasma(v) for v in np.linspace(0, 1, len(target_times))]

    ax_rho = axes[0, 0]
    ax_p = axes[0, 1]
    ax_u = axes[1, 0]
    ax_T = axes[1, 1]

    p_scale = 1e12  # 1 Mbar = 1e12 Barye
    u_scale = 1e5   # 1 km/s = 1e5 cm/s

    for i, (t_target, sim_color) in enumerate(zip(target_times, sim_colors)):
        # 1) Find simulation actual data at nearest target time
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]

        sim_rho = history.rho[idx_sim]
        sim_p = history.p[idx_sim] / p_scale
        sim_u = history.u[idx_sim] / u_scale
        sim_T = history.T[idx_sim]

        # Determine fronts at this actual time
        m_f = ablation_solver.heat_solver.ablated_mass(time=t_actual)
        m_s = m_f + ablation_solver.shock_solver.shocked_mass(time=t_actual)

        # Define grids
        mass_sub = np.linspace(1e-12, m_f, 500)
        mass_shock = np.linspace(1e-12, m_sim[-1], 1000)

        # 2) Solve Subsonic region exact profiles
        sol_sub = ablation_solver.heat_solver.solve(mass=mass_sub, time=t_actual)
        exact_sub_rho = sol_sub["density"]
        exact_sub_p = sol_sub["pressure"] / p_scale
        exact_sub_u = sol_sub["velocity"] / u_scale
        exact_sub_T = sol_sub["temperature"]

        # 3) Solve Shock region exact profiles (covers shock region and cold region up to m_sim[-1])
        sol_shock = ablation_solver.shock_solver.solve(mass=mass_shock, time=t_actual)
        exact_shock_rho = sol_shock["density"]
        exact_shock_p = sol_shock["pressure"] / p_scale
        exact_shock_u = sol_shock["velocity"] / u_scale
        # shock solver doesn't solve T natively, so compute it using the shock EOS
        exact_shock_T = ((sol_shock["pressure"] * sol_shock["density"]**(ablation_solver.mu_shock - 1.0)) / (ablation_solver.shock_solver.r * ablation_solver.f_shock))**(1.0 / ablation_solver.beta_shock)

        # 4) Subsonic Analytical fits mapped to CGS
        fits_sub = calculate_dimensional_fits_sub(mass_sub, t_actual, ablation_solver.heat_solver, sub_params)
        fit_sub_rho = fits_sub["density"]
        fit_sub_p = fits_sub["pressure"] / p_scale
        fit_sub_u = fits_sub["velocity"] / u_scale
        fit_sub_T = fits_sub["temperature"]

        # 5) Shock Analytical fits mapped to CGS (covers both shock and cold regions)
        fits_shock = calculate_dimensional_fits_shock(mass_shock, t_actual, ablation_solver.shock_solver, shock_params)
        fit_shock_rho = fits_shock["density"]
        fit_shock_p = fits_shock["pressure"] / p_scale
        fit_shock_u = fits_shock["velocity"] / u_scale
        fit_shock_T = fits_shock["temperature"]

        show_label = i == 0

        # Plot Simulation (entire domain)
        ax_rho.plot(m_sim * 1e3, sim_rho, '-', color=sim_color, markersize=3, alpha=0.7, label=f"Simulation ({t_target*1e9:.1f} ns)" if show_label else None)
        ax_p.plot(m_sim * 1e3, sim_p, '-', color=sim_color, markersize=3, alpha=0.7)
        ax_u.plot(m_sim * 1e3, sim_u, '-', color=sim_color, markersize=3, alpha=0.7)
        ax_T.plot(m_sim * 1e3, sim_T, '-', color=sim_color, markersize=3, alpha=0.7)

        # Plot Subsonic exact solutions (solid black)
        ax_rho.plot(mass_sub * 1e3, exact_sub_rho, '-', color='black', lw=2.0, label="Subsonic Solver" if show_label else None)
        ax_p.plot(mass_sub * 1e3, exact_sub_p, '-', color='black', lw=2.0)
        ax_u.plot(mass_sub * 1e3, exact_sub_u, '-', color='black', lw=2.0)
        ax_T.plot(mass_sub * 1e3, exact_sub_T, '-', color='black', lw=2.0)

        # Plot Subsonic fits (solid green)
        ax_rho.plot(mass_sub * 1e3, fit_sub_rho, '.', color='green', markersize=2, alpha=0.4, label="Subsonic Fit" if show_label else None)
        ax_p.plot(mass_sub * 1e3, fit_sub_p, '.', color='green', markersize=2, alpha=0.4)
        ax_u.plot(mass_sub * 1e3, fit_sub_u, '.', color='green', markersize=2, alpha=0.4)
        ax_T.plot(mass_sub * 1e3, fit_sub_T, '.', color='green', markersize=2, alpha=0.4)

        # Plot Shock exact solutions (dashed black)
        ax_rho.plot(mass_shock * 1e3, exact_shock_rho, '--', color='black', lw=1.8, label="Shock Solver" if show_label else None)
        ax_p.plot(mass_shock * 1e3, exact_shock_p, '--', color='black', lw=1.8)
        ax_u.plot(mass_shock * 1e3, exact_shock_u, '--', color='black', lw=1.8)
        ax_T.plot(mass_shock * 1e3, exact_shock_T, '--', color='black', lw=1.8)

        # Plot Shock fits (dashed green, covering shock + cold region)
        ax_rho.plot(mass_shock * 1e3, fit_shock_rho, ':', color='green', lw=1.5, alpha=0.7, label="Shock Fit" if show_label else None)
        ax_p.plot(mass_shock * 1e3, fit_shock_p, ':', color='green', lw=1.5, alpha=0.7)
        ax_u.plot(mass_shock * 1e3, fit_shock_u, ':', color='green', lw=1.5, alpha=0.7)
        ax_T.plot(mass_shock * 1e3, fit_shock_T, ':', color='green', lw=1.5, alpha=0.7)

    # Build time legend entries using plasma colors
    time_handles = [
        Line2D([0], [0], color=sim_colors[k], lw=2, label=f"{target_times[k]*1e9:.1f} ns")
        for k in range(len(target_times))
    ]
    style_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Subsonic Solver'),
        Line2D([0], [0], color='green', lw=2, linestyle=':', label='Subsonic Fit'),
        Line2D([0], [0], color='black', lw=1.8, linestyle='--', label='Shock Solver'),
        Line2D([0], [0], color='green', lw=1.8, linestyle='--', label='Shock Fit'),
    ]
    ax_rho.legend(handles=time_handles + style_handles, loc="best", fontsize=9.5)

    # Styling
    labels = ["Density [g/cm³]", "Pressure [MBar]", "Velocity [km/s]", "Temperature [Kelvin]"]
    for j, ax in enumerate([ax_rho, ax_p, ax_u, ax_T]):
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels[j], fontsize=12)
        ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm²]", fontsize=12)

    plt.suptitle(f"Unified Patched Ablation & Shock Verification\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved patched unified dimensional comparison plot to {plot_path}")
    plt.close(fig)


def run_preset_workflow():
    """Run simulation, solve patched similarity models, run fitting pipelines, and plot."""
    case, history, ablation_solver = get_data()

    # 1) Run Subsonic fitting pipeline using ablation_solver.heat_solver
    print("--- Running Subsonic Ablation Fitting ---")
    sub_solver = ablation_solver.heat_solver
    sub_params = perform_subsonic_fitting(sub_solver)

    # 2) Run Piston Shock fitting pipeline using ablation_solver.shock_solver
    print("--- Running Piston Shock Fitting ---")
    shock_solver = ablation_solver.shock_solver
    shock_params = perform_shock_fitting(shock_solver)

    # 3) Plot single patched dimensional comparison
    dv_dir = _REPO_ROOT / "results" / "ictt" / "dimensional_verification"
    dv_dir.mkdir(parents=True, exist_ok=True)
    plot_path = dv_dir / "fig_8_patched_fit_comparison.png"

    plot_patched_dimensional_fit_comparison(
        history=history,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
        case=case,
        plot_path=str(plot_path),
        case_title="Fig 8 Constant Temperature Drive (tau=0)",
    )


def main():
    run_preset_workflow()
    print("\nPatched ablation and shock simulations, fittings, comparisons, and plots generated successfully!")


if __name__ == "__main__":
    main()

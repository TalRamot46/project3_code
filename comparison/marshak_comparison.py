"""
Compare three solutions for the Marshak radiation diffusion problem:
1. Shussman supersonic (piston-shock) solver
2. 1D self-similar diffusion (subsonic heat wave) simulation
3. Radiation-only rad_hydro simulation with improved T_bath calculation

This script overlays temperature or other profiles vs spatial/self-similar coordinates.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from project3_code.rad_hydro_sim.verification.compare_radiation_plots import plot_radiation_comparison_slider

_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
_MENAHEM_SOLVERS_DIR = Path(__file__).resolve().parents[1] / "menahem_new"
if str(_MENAHEM_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_SOLVERS_DIR))

from project3_code.menahem_new.subsonic_heat_wave import SubsonicHeatWave
from project3_code.rad_hydro_sim.output_paths import get_menahem_reproduction_figures_dir
from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.simulation.iterator import (
    simulate_rad_hydro,
    get_T_bath,
)
from project3_code.rad_hydro_sim.verification.run_comparison import run_supersonic_solver_reference
from project3_code.rad_hydro_sim.verification.run_diffusion_1d import run_diffusion_1d


def _ns_amplitude_rescale(amp: float, tau: float) -> float:
    """Convert drive amplitude defined with t[ns] into amplitude for t[s]."""
    return float(amp) * (1.0e9 ** float(tau))


def _build_mass_grid_uniform(case, omega: float, num_cells: int) -> np.ndarray:
    """Build uniform Lagrangian mass grid."""
    coordinate = np.array(list(sorted(set(
        list(np.linspace(0., float(case.x_max), num_cells+1))
    ))))
    dx = coordinate[1:] - coordinate[:-1]
    density = case.rho0 / (1.-omega) * (coordinate[1:]**(1.-omega) - coordinate[:-1]**(1.-omega))/(coordinate[1:] - coordinate[:-1])
    mass_cells = density * dx
    mass = np.cumsum(mass_cells)
    mass = np.array([1e-30, 1e-7*mass[0]] + list(mass))
    return mass

def main() -> None:
    """Main comparison function."""
    from project3_code.rad_hydro_sim.problems.presets_config import (
        PRESET_CONSTANT_T_RADIATION_ONLY,
    )
    from project3_code.rad_hydro_sim.verification.radiation_data import (
        diffusion_output_to_radiation_data,
        rad_hydro_history_to_radiation_data,
    )
    
    # Get Marshak radiation-only preset
    case_base, config_base = get_preset(PRESET_CONSTANT_T_RADIATION_ONLY)
    case = case_base
    config = replace(
        config_base,
        N=150,  # number of cells
        store_every=1,
        show_plot=False,
        show_slider=False,
    )
    
    print(f"Case: {case.title}")
    print(f"T0_Kelvin: {case.T0_Kelvin}")
    print(f"tau: {case.tau}")
    print(f"t_end: {case.t_sec_end}")
    print(f"bc_type: {case.bc_type}")
    
    # Evaluation time (middle of simulation)
    eval_time = case.t_sec_end * 0.5
    
    print(f"\nEvaluating all three solvers at t = {eval_time:.3e} s")
    
    # 1. Shussman supersonic solver (manager_super)
    print("Running Shussman supersonic solver (manager_super)...")
    shussman_super_data = run_supersonic_solver_reference(case, n_times=100)

    # 2. Radiation-only rad_hydro (Marshak BC). Ensure Marshak so rad_hydro
    #    will use the subsonic-derived T_bath internally; also compute and
    #    print the T_bath we would obtain from menahem's solver for reference.
    t_bath = get_T_bath(case, time=eval_time)
    print(f"Computed T_bath from SubsonicHeatWave (for Marshak drive): {t_bath:.6e} K")
    x_cells, state, meta, history = simulate_rad_hydro(
        rad_hydro_case=case,
        simulation_config=config,
    )
    rad_hydro_data = rad_hydro_history_to_radiation_data(history)
    
    # 3. 1D diffusion simulation (1D self similar diffusion simulation)
    times_sec, z, T_list, E_rad_list = run_diffusion_1d(
    x_max=float(case.x_max),
    t_end=float(case.t_sec_end),
    T_bath_Kelvin=float(case.T0_Kelvin),
    rho0=float(case.rho0),
    n_times=40,
    Nz=config.N,
    f_Kelvin=float(case.f_Kelvin),
    g_Kelvin=float(case.g_Kelvin),
    T_right_Kelvin=float(case.T_right_Kelvin or 0.0),
    marshak_boundary=True,
    )
    # Create comparison plots
    diffusion_data = diffusion_output_to_radiation_data(times_sec, z, T_list, E_rad_list)

    sim_data = rad_hydro_data
    ref_data = diffusion_data
    extra_refs = [shussman_super_data] if shussman_super_data is not None else []
    title = f"Marshak radiation diffusion comparison at t={eval_time:.3e}"
    plot_radiation_comparison_slider(
        sim_data, ref_data,
        extra_ref_data=extra_refs,
        show=False,
        title=title,
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fig. 8 Preset Comparison Script: chi=1000 vs chi=1 vs Menahem's Solver.

This script visually compares the PRESET_FIG_8_CONSTANT_TEMPERATURE under:
1. Strong radiation-material coupling (chi = 1000, 1T limit)
2. Weak radiation-material coupling (chi = 1, 2T limit)
3. Both compared to Menahem's analytical solver (local thermodynamic equilibrium).

The results are presented in a multi-panel interactive Matplotlib slider.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Ensure parent of project3_code is on sys.path for absolute package imports
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

# Ensure menahem_new is on sys.path for Menahem's analytical solvers
_MENAHEM_DIR = Path(__file__).resolve().parents[1] / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

# Import rad_hydro_sim components
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    KELVIN_PER_HEV,
)
from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.hydro_sim.verification.compare_shock_plots import (
    load_rad_hydro_history,
    plot_comparison_slider,
)
from project3_code.rad_hydro_sim.verification.hydro_data import RadHydroSimData
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    run_menahem_piecewise_reference,
)


def to_hev_data(sim_data: RadHydroSimData, label: str, color: str, linestyle: str = "-") -> RadHydroSimData:
    """Convert temperatures in RadHydroSimData from Kelvin to HeV for display."""
    T_hev = [t / KELVIN_PER_HEV for t in sim_data.T] if sim_data.T else []
    Tm_hev = [tm / KELVIN_PER_HEV for tm in sim_data.T_material] if sim_data.T_material else []
    return RadHydroSimData(
        times=sim_data.times,
        m=sim_data.m,
        x=sim_data.x,
        rho=sim_data.rho,
        p=sim_data.p,
        u=sim_data.u,
        e=sim_data.e,
        T=T_hev,
        E_rad=sim_data.E_rad,
        T_material=Tm_hev,
        label=label,
        color=color,
        linestyle=linestyle,
    )


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    N_CELLS = 1000       # Grid cell count. 500 is very fast and snappy.
    X_AXIS = "m"        # "m" for Lagrangian mass coordinate, "x" for spatial position

    print("================================================================")
    print(" Fig. 8 Coupling Factor Comparison (chi=1000 vs chi=1 vs Menahem)")
    print("================================================================")

    # 1. Load default Fig. 8 case and config
    case_base, config_base = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    
    # Configure simulation numerical settings locally (silent execution)
    config = replace(
        config_base,
        N=N_CELLS,
        show_plot=False,
        show_slider=False,
        gif_path=None,
        save_path=None,
    )

    # 2. Setup simulation cases
    # Case 1: strong coupling (chi = 1000)
    case_chi_1000 = case_base 
    
    # Case 2: weak coupling (chi = 1)
    case_chi_1 = replace(
        case_base,
        chi=1.0,
        title=r"Fig 8 comparison ($\chi = 1$)",
    )

    # 3. Execute simulations
    print(f"\n[1/3] Running Rad-Hydro simulation for chi=1000 (N={N_CELLS})...")
    _, _, _, history_1000 = simulate_rad_hydro(case_chi_1000, config)
    sim_data_1000 = load_rad_hydro_history(history_1000, label="Rad-Hydro (chi=1000)")
    # Convert temperatures to HeV
    sim_data_1000_hev = to_hev_data(sim_data_1000, label=r"Rad-Hydro ($\chi=1000$)", color="black", linestyle="-")

    print(f"\n[2/3] Running Rad-Hydro simulation for chi=1 (N={N_CELLS})...")
    _, _, _, history_1 = simulate_rad_hydro(case_chi_1, config)
    sim_data_1 = load_rad_hydro_history(history_1, label="Rad-Hydro (chi=1)")
    # Convert temperatures to HeV
    sim_data_1_hev = to_hev_data(sim_data_1, label=r"Rad-Hydro ($\chi=1$)", color="red", linestyle="--")

    # 4. Run Menahem's analytical solver
    print("\n[3/3] Running Menahem's analytical ablation solver...")
    # Use exact same positive time stamps from simulation
    times_sec = sim_data_1000.times
    # Menahem's solver is evaluated on case parameters (independent of chi)
    menahem_ref = run_menahem_piecewise_reference(
        case=case_chi_1000,
        times_sec=times_sec,
        num_cells=N_CELLS,
        label="Menahem (Ablation Solver)",
        color="magenta",
        linestyle="--",
    )

    if menahem_ref is None:
        print("Error: Could not execute Menahem's solver reference.")
        return

    # 5. Launch interactive comparison slider
    print("\nLaunching interactive comparison slider plot...")
    title_fig = rf"Fig. 8 Preset: Radiation-Material Coupling Comparison (N={N_CELLS})"
    
    # We pass Case 1 as sim_data, Case 2 as ref_data, and Menahem as extra_data
    plot_comparison_slider(
        sim_data=sim_data_1000_hev,
        ref_data=sim_data_1_hev,
        xaxis=X_AXIS,
        show=True,
        title=title_fig,
        extra_data=[menahem_ref],
    )


if __name__ == "__main__":
    main()

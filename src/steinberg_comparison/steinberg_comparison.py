#!/usr/bin/env python3
"""
Steinberg comparison module.
Includes helper functions for running rad_hydro simulations, fetching analytical solver results,
parsing Steinberg's original simulation (diffusion) and IMC data, and plotting comparisons.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.integrate

if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(scipy.integrate, "cumtrapz"):
    scipy.integrate.cumtrapz = scipy.integrate.cumulative_trapezoid
if not hasattr(np, "trapz"):
    if hasattr(scipy.integrate, "trapezoid"):
        np.trapz = scipy.integrate.trapezoid
    elif hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid

import matplotlib.pyplot as plt

# Ensure project3_code parent is on sys.path for absolute package imports
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
)
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    run_menahem_piecewise_reference,
)

from project3_code.rad_hydro_sim.verification.hydro_data import RadHydroSimData

def get_sim_results_for_mutiplier_and_num_cells(chi: float, num_cells: int, preset_name: str = PRESET_FIG_8_CONSTANT_TEMPERATURE) -> RadHydroSimData:
    """
    Run the rad_hydro simulation with the specified chi and number of cells.
    Uses the specified preset_name as the base preset.
    """
    case_base, config_base = get_preset(preset_name)
    
    # Configure simulation numerical settings locally (silent execution)
    config = replace(
        config_base,
        N=num_cells,
        show_plot=False,
        show_slider=False,
        gif_path=None,
        save_path=None,
    )
    case_chi = replace(
        case_base,
        chi=chi,
    )
    
    _, _, _, history = simulate_rad_hydro(case_chi, config)
    sim_data = load_rad_hydro_history(history, label=f"Rad-Hydro (chi={chi})")
    return sim_data


def get_solver_results_for_multiplier_and_num_cells(chi: float, num_cells: int, preset_name: str = PRESET_FIG_8_CONSTANT_TEMPERATURE, sim_data: RadHydroSimData=None):
    """
    Run the analytical piecewise solver (Menahem's Ablation Solver) for the preset,
    evaluated at the time grid of the simulation with the same preset parameters.
    """
    # 1. First get the simulation data to extract the correct time stamps
    times_sec = sim_data.times[::(len(sim_data.times)//200)]
    
    case_base, _ = get_preset(preset_name)
    case_chi = replace(case_base, chi=chi)
    
    solver_data = run_menahem_piecewise_reference(
        case=case_chi,
        times_sec=times_sec,
        num_cells=num_cells,
        label="Menahem (Ablation Solver)",
    )
    return solver_data


def extract_steinberg_data() -> dict[str, dict[str, np.ndarray]]:
    """
    Reads the content of all .txt files inside the shteinberg_comparison folder
    and parses their columns into a dictionary of dictionaries.
    Outer dictionary keys: filenames (e.g., 'profile_diff_1024.txt').
    Inner dictionary keys: column names (e.g., 'x', 'temperature', 'mass_coord').
    """
    current_dir = Path(__file__).resolve().parent
    results = {}
    for filename in os.listdir(current_dir):
        if filename.endswith(".txt"):
            filepath = current_dir / filename
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Extract headers from comment line
            headers = []
            for line in lines:
                if line.startswith("# x") or line.startswith("#  x"):
                    headers = line.lstrip("#").strip().split()
                    break
            if not headers and len(lines) > 1:
                headers = lines[1].lstrip("#").strip().split()
            
            # Read numerical data rows
            data_rows = []
            for line in lines:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                data_rows.append([float(val) for val in line_str.split()])
            
            if data_rows:
                data_matrix = np.array(data_rows)
                inner_dict = {}
                for idx, header in enumerate(headers):
                    if idx < data_matrix.shape[1]:
                        inner_dict[header] = data_matrix[:, idx]
                results[filename] = inner_dict
    return results


def extract_steinberg_sim_for_multiplier_and_num_cells(multiplier: float, num_cells: int) -> dict[str, np.ndarray] | None:
    """
    Extracts the inner dictionary data for the diffusion simulation txt file matching
    the specified multiplier and cell count.
    Note: if multiplier is 1, it matches filenames like 'profile_diff_1024.txt' (no p suffix).
    """
    data = extract_steinberg_data()
    for filename, inner_dict in data.items():
        if "diff" in filename:
            parts = filename.replace(".txt", "").split("_")
            if len(parts) >= 3:
                try:
                    f_num_cells = int(parts[2])
                    if len(parts) >= 4 and parts[3].startswith('p'):
                        f_multiplier = int(parts[3][1:])
                    else:
                        f_multiplier = 1
                    
                    if int(f_num_cells) == int(num_cells) and int(f_multiplier) == int(multiplier):
                        return inner_dict
                except ValueError:
                    continue
    return None


def extract_steinberg_IMC_for_num_cells(num_cells: int) -> dict[str, np.ndarray] | None:
    """
    Extracts the inner dictionary data for the IMC simulation txt file matching
    the specified cell count (e.g., 'profile_imc_512.txt').
    """
    data = extract_steinberg_data()
    for filename, inner_dict in data.items():
        if "imc" in filename:
            parts = filename.replace(".txt", "").split("_")
            if len(parts) >= 3:
                try:
                    f_num_cells = int(parts[2])
                    if int(f_num_cells) == int(num_cells):
                        return inner_dict
                except ValueError:
                    continue
    return None


def plot_steinberg_comparison_for_multiplier_and_num_cells(
    multiplier: float,
    num_cells: int,
    preset_name: str = PRESET_FIG_8_CONSTANT_TEMPERATURE,
    save_path: str | Path | None = None,
    show: bool = True
):
    """
    Plots a comparison of:
    1. Simulation data (current rad_hydro implementation)
    2. Solver data (Menahem's analytical solver)
    3. Steinberg's original simulation data (diffusion)
    4. Steinberg's IMC data
    
    Plotted at the end of simulation (t ≈ 2.0 ns).
    """
    # 1. Fetch simulation and solver data
    sim_data = get_sim_results_for_mutiplier_and_num_cells(multiplier, num_cells, preset_name=preset_name)
    solver_data = get_solver_results_for_multiplier_and_num_cells(multiplier, num_cells, preset_name=preset_name, sim_data=sim_data)
    
    # 2. Fetch Steinberg parsed data
    st_sim = extract_steinberg_sim_for_multiplier_and_num_cells(multiplier, num_cells)
    st_imc = extract_steinberg_IMC_for_num_cells(num_cells)
    
    # 3. Find the closest time index to 2.0 ns (2.0e-9 s)
    sim_idx = np.argmin(np.abs(sim_data.times - 2.0e-9))
    sol_idx = np.argmin(np.abs(solver_data.times - 2.0e-9))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax_T, ax_rho = axes[0, 0], axes[0, 1]
    ax_p, ax_u = axes[1, 0], axes[1, 1]
    
    # Plot current simulation (convert Kelvin to HeV for temperature)
    sim_m = sim_data.m[sim_idx]
    sim_Tmat = sim_data.T_material[sim_idx] / KELVIN_PER_HEV
    sim_Trad = sim_data.T[sim_idx] / KELVIN_PER_HEV
    ax_T.plot(sim_m, sim_Tmat, label="Simulation T_mat", color="blue", linestyle="-", linewidth=2.5)
    # ax_T.plot(sim_m, sim_Trad, label="Simulation T_rad", color="blue", linestyle="--", linewidth=1.8)
    
    # Plot analytical solver (already in HeV)
    sol_m = solver_data.m[sol_idx]
    sol_T = solver_data.T_material[sol_idx]
    ax_T.plot(sol_m, sol_T, label="Analytical Solver T", color="red", linestyle="--", linewidth=2)
    
    # Plot Steinberg Sim Temperature
    if st_sim is not None:
        st_sim_m = st_sim["mass_coord"]
        st_sim_Tmat = st_sim["temperature"] / KELVIN_PER_HEV
        st_sim_Trad = st_sim["T_rad"] / KELVIN_PER_HEV
        ax_T.plot(st_sim_m, st_sim_Tmat, label="Steinberg Sim T_mat", color="green", linestyle="-.", linewidth=2.5)
        # ax_T.plot(st_sim_m, st_sim_Trad, label="Steinberg Sim T_rad", color="green", linestyle="--", linewidth=1.8)
    else:
        print(f"Warning: Steinberg simulation temperature data for multiplier={multiplier}, N={num_cells} not found.")
        
    # Plot Steinberg IMC Temperature
    # if st_imc is not None:
    #     st_imc_m = st_imc["mass_coord"]
    #     st_imc_Tmat = st_imc["temperature"] / KELVIN_PER_HEV
    #     st_imc_Trad = st_imc["T_rad"] / KELVIN_PER_HEV
    #     ax_T.plot(st_imc_m, st_imc_Tmat, label="Steinberg IMC T_mat", color="red", linestyle="--", linewidth=2.5)
    #     # ax_T.plot(st_imc_m, st_imc_Trad, label="St/einberg IMC T_rad", color="#ff7f0e", linestyle="--", linewidth=1.8)
    # else:
    #     print(f"Warning: Steinberg IMC temperature data for N={num_cells} not found.")
        
    ax_T.set_ylabel("Temperature $T$ [HeV]", fontsize=12)
    ax_T.grid(True, alpha=0.3, linestyle=":")
    ax_T.legend(fontsize=9, loc="upper right")

    # --- Density Plot ---
    ax_rho.plot(sim_m, sim_data.rho[sim_idx], label="Simulation", color="blue", linestyle="-", linewidth=2.5)
    ax_rho.plot(sol_m, solver_data.rho[sol_idx], label="Analytical Solver", color="red", linestyle="--", linewidth=2)
    if st_sim is not None:
        ax_rho.plot(st_sim_m, st_sim["density"], label="Steinberg Sim", color="green", linestyle="-.", linewidth=2.5)
    # if st_imc is not None:
    #     ax_rho.plot(st_imc_m, st_imc["density"], label="Steinberg IMC", color="#ff7f0e", linestyle="-", linewidth=2.5)
    ax_rho.set_ylabel("Density $\\rho$ [g/cm³]", fontsize=12)
    ax_rho.grid(True, alpha=0.3, linestyle=":")
    ax_rho.legend(fontsize=9, loc="upper right")

    # --- Pressure Plot ---
    ax_p.plot(sim_m, sim_data.p[sim_idx] / 1e12, label="Simulation", color="blue", linestyle="-", linewidth=2.5)
    ax_p.plot(sol_m, solver_data.p[sol_idx] / 1e12, label="Analytical Solver", color="red", linestyle="--", linewidth=2)
    if st_sim is not None:
        ax_p.plot(st_sim_m, st_sim["pressure"] / 1e12, label="Steinberg Sim", color="green", linestyle="-.", linewidth=2.5)
    # if st_imc is not None:
    #     ax_p.plot(st_imc_m, st_imc["pressure"] / 1e12, label="Steinberg IMC", color="#ff7f0e", linestyle="-", linewidth=2.5)
    ax_p.set_xlabel("Mass coordinate $m$ [g/cm²]", fontsize=12)
    ax_p.set_ylabel("Pressure $P$ [MBar]", fontsize=12)
    ax_p.grid(True, alpha=0.3, linestyle=":")
    ax_p.legend(fontsize=9, loc="upper right")

    # --- Velocity Plot ---
    ax_u.plot(sim_m, sim_data.u[sim_idx] / 1e5, label="Simulation", color="blue", linestyle="-", linewidth=2.5)
    ax_u.plot(sol_m, solver_data.u[sol_idx] / 1e5, label="Analytical Solver", color="red", linestyle="--", linewidth=2)
    if st_sim is not None:
        ax_u.plot(st_sim_m, st_sim["velocity_x"] / 1e5, label="Steinberg Sim", color="green", linestyle="-.", linewidth=2.5)
    # if st_imc is not None:
    #     ax_u.plot(st_imc_m, st_imc["velocity_x"] / 1e5, label="Steinberg IMC", color="#ff7f0e", linestyle="-", linewidth=2.5)
    ax_u.set_xlabel("Mass coordinate $m$ [g/cm²]", fontsize=12)
    ax_u.set_ylabel("Velocity $u$ [km/s]", fontsize=12)
    ax_u.grid(True, alpha=0.3, linestyle=":")
    ax_u.legend(fontsize=9, loc="upper right")

    fig.suptitle(f"Steinberg Comparison: chi={multiplier}, N={num_cells} (t = 2.0 ns)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(sp), dpi=200, bbox_inches="tight")
        print(f"Saved comparison plot to {sp}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # Smoke test: Plot the comparison for multiplier=100 and N=512
    print("Running smoke test comparison for N=512, chi=100...")
    plot_steinberg_comparison_for_multiplier_and_num_cells(
        multiplier=100,
        num_cells=512,
        save_path=Path(__file__).resolve().parent / "steinberg_comparison_diff_512_p_100.png",
        show=False
    )
    print("Running smoke test comparison for N=512, chi=1000...")
    plot_steinberg_comparison_for_multiplier_and_num_cells(
        multiplier=1000,
        num_cells=512,
        save_path=Path(__file__).resolve().parent / "steinberg_comparison_diff_512_p_1000.png",
        show=False
    )
    print("Running smoke test comparison for N=1024, chi=100...")
    plot_steinberg_comparison_for_multiplier_and_num_cells(
        multiplier=100,
        num_cells=1024,
        save_path=Path(__file__).resolve().parent / "steinberg_comparison_diff_1024_p_100.png",
        show=False
    )
    print("Smoke test finished.")

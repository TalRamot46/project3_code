# ictt29/plot_xt.py
"""
Space-time (x-t) trajectory and front plotting.

Runs the 1D Rad-Hydro simulation (or loads from cache) and plots
cell boundaries x(t) with diagnosed fronts (Simulation vs Analytic).
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
)
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.verification.shussman_comparison import run_shussman_piecewise_reference
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    run_menahem_piecewise_reference,
    _ablation_kwargs_from_case,
    _build_mass_grid,
)
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

from ablation_solver_og import AblationSolver


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
    
    ax.set_xlabel(r"$t$ [ns]", fontsize=12)
    ax.set_ylabel(r"$x$ [cm]", fontsize=12)
    ax.set_title(f"Space-Time (xt) Trajectories and Fronts\n{case_title}", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_ylim(0, 0.0016)
    
    fig.tight_layout()
    fig.savefig(xt_path, dpi=200)
    plt.close(fig)


# =============================================================================
# Main Orchestration
# =============================================================================

def run_simulation_and_references(preset_name: str, case_label: str):
    """Run full simulation and build reference solvers, or load from pickle cache."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_label}_cache.pkl"
    
    # 1) Initialize case and set grid size N=400
    case, config = get_preset(preset_name)
    config = replace(config, N=400)
    
    if cache_path.exists():
        print(f"Loading cached simulation and reference data from {cache_path}...")
        try:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if 'numpy._core' in module:
                        module = module.replace('numpy._core', 'numpy.core')
                    return super().find_class(module, name)
            with open(cache_path, "rb") as f:
                data = CustomUnpickler(f).load()
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
    
    # Build Menahem ablation piecewise reference
    print(f"Building Menahem piecewise reference...")
    menahem_times_sec = np.array([0.25, 0.5, 0.75, 1.0]) * float(case.t_sec_end)
    menahem_ref = run_menahem_piecewise_reference(case, times_sec=menahem_times_sec)
        
    # Run radiation-hydrodynamics simulation
    print(f"Running simulation...")
    _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)
    
    # Build Shussman piecewise reference
    print(f"Building Shussman piecewise reference...")
    eval_times_ns = eval_times_sec * 1e9
    T0_HeV = float(case.T0_Kelvin) / KELVIN_PER_HEV
    shussman_ref = run_shussman_piecewise_reference(case, times_ns=eval_times_ns, T0_HeV=T0_HeV)
    
    # Instantiate AblationSolver
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


def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run full verification comparison pipeline for a given preset."""
    print("=" * 80)
    print(f"PROCESSING PRESET: {preset_name} -> {case_label}")
    print("=" * 80)
    
    cache_path = Path("results/ictt/cache") / f"{case_label}_cache.pkl"
    if not cache_path.exists():
        print("Pickle file doesn't exist, running simulation...")
        case, history, shussman_ref, menahem_ref, ablation_solver = run_simulation_and_references(
            preset_name, case_label
        )
    else:
        print("Pickle file exists, loading simulation...")
        try:
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if 'numpy._core' in module:
                        module = module.replace('numpy._core', 'numpy.core')
                    return super().find_class(module, name)
            with open(cache_path, "rb") as f:
                data = CustomUnpickler(f).load()
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running simulation...")
            case, history, shussman_ref, menahem_ref, ablation_solver = run_simulation_and_references(
                preset_name, case_label
            )
            data = {"case": case, "history": history, "ablation_solver": ablation_solver}
        case = data.get("case")
        if case is None:
            case, _ = get_preset(preset_name)
        history = data["history"]
        ablation_solver = data["ablation_solver"]
    
    # Generate x-t plot
    out_dir = Path("results/ictt/xt")
    out_dir.mkdir(parents=True, exist_ok=True)
    xt_path = str(out_dir / f"{case_label}_xt.png")
    plot_xt_trajectories(history, case, xt_path, case_title, ablation_solver=ablation_solver)
    
    print(f"Preset {preset_name} x-t plot generated successfully.")


def main():
    run_preset_workflow(
        PRESET_FIG_8_CONSTANT_TEMPERATURE, 
        "constant_boundary_temperature_tau_0", 
        "Constant Boundary Temperature (tau=0)"
    )
    print("\nAll x-t trajectory plots completed successfully!")


if __name__ == "__main__":
    main()

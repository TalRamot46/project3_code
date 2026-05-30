# ictt29/final_comparison.py
"""
Final Comparison and Verification Script.
Hardcodes the exact Level 3 CGS equations directly from the compiled LaTeX file
(substituted at t = 1 ns, as functions of m and t/ns) and overlays them against
the 1D Rad-Hydro simulation and the patched AblationSolver.

Outputs:
- results/ictt/evolution/fig_8_final_comparison.gif
"""
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from dataclasses import replace
import numpy as np
# Dynamic monkeypatch of numpy.trapezoid to numpy.trapz for compatibility with NumPy 1.x
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter

# Ensure proper project imports
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_REPO_ROOT = _REPO_PARENT / "project3_code"
_MENAHEM_DIR = _REPO_ROOT / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import PRESET_FIG_8_CONSTANT_TEMPERATURE
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _ablation_kwargs_from_case,
    _build_mass_grid,
)
from ablation_solver import AblationSolver

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
# Level 3 CGS Hardcoded Curve Fit Evaluation
# =============================================================================

def evaluate_level3_fits(mass_grid: np.ndarray, t_sec: float) -> dict:
    """
    Computes the explicit Level 3 analytical fits directly from the compiled LaTeX file.
    All coefficients are hardcoded and expressed as explicit functions of m and t/ns.
    Returns profiles for p, u, rho, T in CGS units.
    """
    t_ns = t_sec * 1e9
    
    # 1. Front Positions (derived from the coordinate shooting solver in print_fitted_formulas.py)
    # y = m/m_f(t) = 981.45247 * m * (t/ns)^(-33/64)
    # y_s = m/m_s(t) = 82.68924 * m * (t/ns)^(-149/192)
    m_f = (t_ns ** (33.0 / 64.0)) / 981.45247
    m_s = (t_ns ** (149.0 / 192.0)) / 82.68924
    
    n = len(mass_grid)
    p = np.zeros(n)
    u = np.zeros(n)
    rho = np.zeros(n)
    T = np.zeros(n)
    
    for i, m in enumerate(mass_grid):
        if m < m_f:
            # --- Subsonic Ablation Regime ---
            y = 981.45247 * m * (t_ns ** (-33.0 / 64.0))
            
            # Fluid velocity u [cm/s] (LaTeX: km/s -> multiply by 1e5 to get cm/s)
            u_kms = -200.39472 * (1.0 - y) / (1.0 + 5.24332440 * 981.45247 * m * (t_ns ** (-33.0 / 64.0))) * (t_ns ** (7.0 / 192.0))
            # Wait, 1.0 + b_u * y = 1.0 + denom_coeff_u_h * m * ...
            # The compiled LaTeX: 1 + 5243.32440 * m * (t/ns)^-33/64
            u_kms = -200.39472 * (1.0 - 981.45247 * m * (t_ns ** (-33.0 / 64.0))) / (1.0 + 5243.32440 * m * (t_ns ** (-33.0 / 64.0))) * (t_ns ** (7.0 / 192.0))
            u[i] = u_kms * 1e5
            
            # Specific volume v [cm^3/g]
            v_val = 600.66640 * ((1.0 - 981.45247 * m * (t_ns ** (-33.0 / 64.0))) ** 39.54884) * (t_ns ** (25.0 / 48.0))
            rho[i] = 1.0 / v_val
            
            # Pressure p [Barye] (LaTeX: MBar -> multiply by 1e12 to get Barye)
            p_mbar = (2.45709 * (y ** 0.87633) + 0.19901 * (y ** 19.97029)) * (t_ns ** (-43.0 / 96.0))
            p[i] = p_mbar * 1e12
            
            # Temperature T [Kelvin] (LaTeX: HeV -> multiply by KELVIN_PER_HEV to get Kelvin)
            T_hev = 1.00000 * (((1.0 - y) * (1.0 + 199.10933 * m * (t_ns ** (-33.0 / 64.0)))) ** (10.0 / 39.0))
            T[i] = T_hev * KELVIN_PER_HEV
            
        elif m < m_s:
            # --- Shock Compressed Regime ---
            # Fluid velocity u [cm/s] (LaTeX: km/s -> multiply by 1e5)
            u_kms = (3.65295 + 11.10002 * (m ** 0.63757) * (t_ns ** -0.49478)) * (t_ns ** (-43.0 / 192.0))
            u[i] = u_kms * 1e5
            
            # Density rho [g/cm^3] (LaTeX: g/cm^3)
            rho[i] = 3057.82924 * (m ** 0.64939) * (t_ns ** -0.50395)
            
            # Pressure p [Barye] (LaTeX: MBar -> multiply by 1e12)
            p_mbar = (2.71000 + 199.81724 * (m ** 1.13315) * (t_ns ** -0.87937)) * (t_ns ** (-43.0 / 96.0))
            p[i] = p_mbar * 1e12
            
            # Temperature T [Kelvin] (LaTeX: HeV -> multiply by KELVIN_PER_HEV)
            # T_shock = 0.00161 * [ m^-0.74030 * t^0.57450 + 73.73330 * m^0.39285 * t^-0.30487 ]^0.625 * t^(-215/768)
            T_hev = 0.00161 * (
                (m ** -0.74030) * (t_ns ** 0.57450) + 73.73330 * (m ** 0.39285) * (t_ns ** -0.30487)
            ) ** 0.625 * (t_ns ** (-215.0 / 768.0))
            T[i] = T_hev * KELVIN_PER_HEV
            
        else:
            # --- Unshocked Cold Material ---
            rho[i] = 19.32
            p[i] = 1e-6
            u[i] = 0.0
            T[i] = 300.0
            
    return {"density": rho, "pressure": p, "velocity": u, "temperature": T, "m_f": m_f, "m_s": m_s}

# =============================================================================
# Data Orchestration
# =============================================================================

def get_data():
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "interactive_ablation_fits_cache.pkl"
    
    print("1) Fetching Fig 8 Preset...")
    case, config = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    
    if cache_path.exists():
        print(f"--> Loading cached simulation history from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            history = data["history"]
            print("--> Simulation cache loaded successfully.")
        except Exception as e:
            print(f"--> Failed to load simulation cache ({e}). Re-running...")
            history = None
    else:
        history = None
        
    if history is None:
        print("2) Running 1D Rad-Hydro Simulation (N=1000)...")
        config = replace(config, show_plot=False, show_slider=False)
        _, _, _, history = simulate_rad_hydro(case, config)
        
        print(f"--> Saving simulation cache to {cache_path}...")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"history": history}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"--> Failed to save simulation cache ({e})")
            
    print("3) Instantiating Menahem Patched Ablation Solver...")
    ablation_solver = AblationSolver(**_ablation_kwargs_from_case(case))
    
    return case, history, ablation_solver

# =============================================================================
# Main Verification Plotting
# =============================================================================

def main():
    case, history, ablation_solver = get_data()
    times = np.array(history.t)
    mass_grid = _build_mass_grid(case, num_cells=200)
    
    # Setup Figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.subplots_adjust(bottom=0.2, top=0.9)
    
    ax_rho, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_T = axes[1, 0], axes[1, 1]
    
    p_scale, u_scale = 1e12, 1e5
    
    # Initial time step (around mid-simulation)
    k_init = len(times) // 2
    t_init = times[k_init]
    
    # 1) Fetch Simulation Data
    sim_m = history.m[k_init]
    sim_rho = history.rho[k_init]
    sim_p = history.p[k_init] / p_scale
    sim_u = history.u[k_init] / u_scale
    sim_T = history.T[k_init]
    
    # 2) Fetch Menahem Solver Data
    sol_men = ablation_solver.solve(mass=mass_grid, time=t_init)
    men_rho = sol_men["density"]
    men_p = sol_men["pressure"] / p_scale
    men_u = sol_men["velocity"] / u_scale
    men_T = sol_men["temperature"]
    
    # 3) Fetch Hardcoded Level 3 Curve Fits
    fits = evaluate_level3_fits(mass_grid, t_init)
    fit_rho = fits["density"]
    fit_p = fits["pressure"] / p_scale
    fit_u = fits["velocity"] / u_scale
    fit_T = fits["temperature"]
    
    # Draw Lines
    # Density Subplot
    line_sim_rho, = ax_rho.plot(sim_m, sim_rho, 'b-', lw=2.2, label='Simulation')
    line_men_rho, = ax_rho.plot(mass_grid, men_rho, 'm--', lw=1.8, label='Menahem Patched')
    line_fit_rho, = ax_rho.plot(mass_grid, fit_rho, 'g-', lw=2.2, label='Analytic Fits (Level 3)')
    
    # Pressure Subplot
    line_sim_p, = ax_p.plot(sim_m, sim_p, 'b-', lw=2.2, label='Simulation')
    line_men_p, = ax_p.plot(mass_grid, men_p, 'm--', lw=1.8, label='Menahem Patched')
    line_fit_p, = ax_p.plot(mass_grid, fit_p, 'g-', lw=2.2, label='Analytic Fits (Level 3)')
    
    # Velocity Subplot
    line_sim_u, = ax_u.plot(sim_m, sim_u, 'b-', lw=2.2, label='Simulation')
    line_men_u, = ax_u.plot(mass_grid, men_u, 'm--', lw=1.8, label='Menahem Patched')
    line_fit_u, = ax_u.plot(mass_grid, fit_u, 'g-', lw=2.2, label='Analytic Fits (Level 3)')
    
    # Temperature Subplot
    line_sim_T, = ax_T.plot(sim_m, sim_T, 'b-', lw=2.2, label='Simulation')
    line_men_T, = ax_T.plot(mass_grid, men_T, 'm--', lw=1.8, label='Menahem Patched')
    line_fit_T, = ax_T.plot(mass_grid, fit_T, 'g-', lw=2.2, label='Analytic Fits (Level 3)')
    
    # Front vertical lines
    v_bnd_rho = ax_rho.axvline(0.0, color='grey', ls='-', alpha=0.5)
    v_abl_rho = ax_rho.axvline(fits["m_f"], color='purple', ls='--', alpha=0.7, label='Ablation Front')
    v_shk_rho = ax_rho.axvline(fits["m_s"], color='red', ls='--', alpha=0.7, label='Shock Front')
    
    v_bnd_p = ax_p.axvline(0.0, color='grey', ls='-', alpha=0.5)
    v_abl_p = ax_p.axvline(fits["m_f"], color='black', ls='-', alpha=0.7)
    v_shk_p = ax_p.axvline(fits["m_s"], color='red', ls='--', alpha=0.7)
    
    v_bnd_u = ax_u.axvline(0.0, color='grey', ls='-', alpha=0.5)
    v_abl_u = ax_u.axvline(fits["m_f"], color='black', ls='-', alpha=0.7)
    v_shk_u = ax_u.axvline(fits["m_s"], color='red', ls='--', alpha=0.7)
    
    v_bnd_T = ax_T.axvline(0.0, color='grey', ls='-', alpha=0.5)
    v_abl_T = ax_T.axvline(fits["m_f"], color='black', ls='-', alpha=0.7)
    v_shk_T = ax_T.axvline(fits["m_s"], color='red', ls='--', alpha=0.7)
    
    # Styling and Labels
    ax_rho.set_ylabel(r"$\rho$ [g/cm³]", fontsize=12)
    ax_p.set_ylabel(r"$P$ [MBar]", fontsize=12)
    ax_u.set_ylabel(r"$u$ [km/s]", fontsize=12)
    ax_T.set_ylabel(r"$T$ [Kelvin]", fontsize=12)
    
    for ax in [ax_rho, ax_p, ax_u, ax_T]:
        ax.set_xlabel(r"Mass coordinate $m$ [g/cm²]", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=9)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        
    title = fig.suptitle(f"Level 3 Piecewise Analytical Fits vs Simulation & Solver\nTime: {t_init*1e9:.3f} ns", fontsize=14, fontweight='bold')
    
    # Set y limits nicely
    ax_rho.set_ylim(-10, 200)
    ax_p.set_ylim(-0.5, 8.0)
    ax_u.set_ylim(-350, 50)
    ax_T.set_ylim(-1e5, 1.3e6)
    
    # Slider Setup
    ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Time [ns]', times.min()*1e9, times.max()*1e9, valinit=t_init*1e9, valfmt='%.3f ns')
    
    def update(val):
        t_val = val * 1e-9
        k = np.argmin(np.abs(times - t_val))
        t_actual = times[k]
        
        # 1) Update Sim
        line_sim_rho.set_data(history.m[k], history.rho[k])
        line_sim_p.set_data(history.m[k], history.p[k] / p_scale)
        line_sim_u.set_data(history.m[k], history.u[k] / u_scale)
        line_sim_T.set_data(history.m[k], history.T[k])
        
        # 2) Update Menahem
        sol = ablation_solver.solve(mass=mass_grid, time=max(t_actual, 1e-18))
        line_men_rho.set_data(mass_grid, sol["density"])
        line_men_p.set_data(mass_grid, sol["pressure"] / p_scale)
        line_men_u.set_data(mass_grid, sol["velocity"] / u_scale)
        line_men_T.set_data(mass_grid, sol["temperature"])
        
        # 3) Update Fits
        fit_vals = evaluate_level3_fits(mass_grid, t_actual)
        line_fit_rho.set_data(mass_grid, fit_vals["density"])
        line_fit_p.set_data(mass_grid, fit_vals["pressure"] / p_scale)
        line_fit_u.set_data(mass_grid, fit_vals["velocity"] / u_scale)
        line_fit_T.set_data(mass_grid, fit_vals["temperature"])
        
        # Update front lines
        v_abl_rho.set_xdata([fit_vals["m_f"], fit_vals["m_f"]])
        v_shk_rho.set_xdata([fit_vals["m_s"], fit_vals["m_s"]])
        v_abl_p.set_xdata([fit_vals["m_f"], fit_vals["m_f"]])
        v_shk_p.set_xdata([fit_vals["m_s"], fit_vals["m_s"]])
        v_abl_u.set_xdata([fit_vals["m_f"], fit_vals["m_f"]])
        v_shk_u.set_xdata([fit_vals["m_s"], fit_vals["m_s"]])
        v_abl_T.set_xdata([fit_vals["m_f"], fit_vals["m_f"]])
        v_shk_T.set_xdata([fit_vals["m_s"], fit_vals["m_s"]])
        
        title.set_text(f"Level 3 Piecewise Analytical Fits vs Simulation & Solver\nTime: {t_actual*1e9:.3f} ns")
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    
    # =============================================================================
    # GIF Generation
    # =============================================================================
    gif_dir = _REPO_ROOT / "results" / "ictt" / "evolution"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / "fig_8_final_comparison.gif"
    
    print(f"4) Saving Level 3 verification animation to {gif_path}...")
    # Select 25 equally spaced times for a smooth but reasonably sized GIF
    gif_times = _get_equally_spaced_elements(times, 25)
    gif_time_vals_ns = gif_times * 1e9
    
    def update_frame(frame_idx):
        t_ns_frame = gif_time_vals_ns[frame_idx]
        slider.set_val(t_ns_frame)
        return [line_sim_rho, line_men_rho, line_fit_rho, 
                line_sim_p, line_men_p, line_fit_p,
                line_sim_u, line_men_u, line_fit_u,
                line_sim_T, line_men_T, line_fit_T]
                
    anim = FuncAnimation(fig, update_frame, frames=len(gif_time_vals_ns), blit=False)
    anim.save(str(gif_path), writer=PillowWriter(fps=6))
    print("5) Animation GIF saved successfully!")
    
    # Show plot window only if in interactive session
    if sys.flags.interactive or hasattr(sys, 'ps1'):
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()

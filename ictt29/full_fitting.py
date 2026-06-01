# ictt29/interactive_ablation_fits.py
"""
Interactive Slider and animated GIF comparison for Fig 8 constant temperature drive.
Overlays:
1. 1D Rad-Hydro Simulation
2. Menahem Patched Ablation Solver (piecewise ODE shooting)
3. Piecewise Patched Self-Similar Analytical Curve Fits

Outputs:
- results/ictt/evolution/fig_8_piecewise_ablation_fits.gif
- An interactive Matplotlib slider window (if run in an interactive shell)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
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
from project3_code.menahem_new.ablation_solver_og import AblationSolver

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
# Curve Fit Evaluation Functions
# =============================================================================

def evaluate_piecewise_fits(mass_grid: np.ndarray, t_sec: float) -> dict:
    """
    Computes the explicit self-similar analytical fits for each physical profile.
    Time t_sec must be positive.
    Returns profiles for p, u, rho, T in CGS units.
    """
    t_ns = t_sec * 1e9
    
    # 1. Front Positions
    m_f = 0.00109 * (t_ns ** 0.515625)
    m_s = 0.0120786 * (t_ns ** 0.7765)
    
    # Pre-allocate profile arrays
    n = len(mass_grid)
    p = np.zeros(n)
    u = np.zeros(n)
    rho = np.zeros(n)
    T = np.zeros(n)
    
    for i, m in enumerate(mass_grid):
        if m < m_f:
            # Subsonic Ablation Regime
            y = m / m_f
            
            # Temperature T [Kelvin]
            T_val_hev = 1.011087 * ((1.0 - y) ** 0.241582)
            T[i] = T_val_hev * KELVIN_PER_HEV
            
            # Density rho [g/cm^3] calculated from Dimensionless EOS
            # rho = (r * T^beta / P)^(1 / (mu - 1))
            r_val = 0.25
            beta_val = 1.6
            mu_val = 0.14
            # P here should be the dimensionless P, and T should be dimensionless T
            # Actually, we can use dimensional values if we use the full EOS, but the dimensionless formula is simpler.
            # Using dimensional variables: p = r * rho * T * (f * rho^-mu)^(-1/beta) -> rho = ((r * f * T^beta) / p)^(1/(mu - 1))
            # Or just evaluate the dimensionless one first:
            T_tilde = 1.011087 * ((1.0 - y) ** 0.241582)
            P_tilde = 0.35486 * y**0.87677 + 0.02905 * y**20.94836
            # Ensure P_tilde > 0 to avoid division by zero at y=0
            if P_tilde <= 0: P_tilde = 1e-15
            rho_tilde = ((r_val * T_tilde**beta_val) / P_tilde) ** (1.0 / (mu_val - 1.0))
            
            # Convert dimensionless rho to dimensional rho
            # rho(m,t) = rho_tilde * A_val^(115/96) * B_val^(25/48) * t^(-25/48)
            # The CGS conversion factor for rho_tilde is exactly what we had: pre_rho = 0.174190 / 0.216463 = 0.80471027
            rho_val = (0.174190 / 0.216463) * rho_tilde * (t_ns ** -0.520833)
            rho[i] = rho_val
            
            # Pressure p [Barye] (1 MBar = 1e12 Barye)
            p_val_mbar = 7.051327 * (t_ns ** -0.447917) * (0.348591 * (y ** 0.876766) + 0.029050 * (y ** 20.948361))
            p[i] = p_val_mbar * 1e12
            
            # Velocity u [cm/s] (1 km/s = 1e5 cm/s)
            # Using the Rational Fit for exact document consistency
            u_val_kms = -191.294 * (t_ns ** 0.036458) * (1.0 - y) / (1.0 + 4.78201 * y)
            u[i] = u_val_kms * 1e5
            
            # Temperature T [Kelvin]
            # (already evaluated above for EOS calculation)
            
        elif m < m_s:
            # Shock compressed Regime
            y_s = (m - m_f) / (m_s - m_f)
            # Avoid y_s = 0 exactly to prevent singular divisions in Temperature
            y_s = max(y_s, 1e-6)
            
            # Density rho [g/cm^3]
            rho[i] = 173.88 * (y_s ** 0.647468)
            
            # Pressure p [Barye]
            p_val_mbar = 2.710000 * (t_ns ** -0.447) * (1.0 + 0.493384 * (y_s ** 1.133029))
            p[i] = p_val_mbar * 1e12
            
            # Velocity u [cm/s]
            u_val_kms = (t_ns ** -0.2235) * (3.65776 + 0.65734 * (y_s ** 0.644937))
            u[i] = u_val_kms * 1e5
            
            # Temperature T [Kelvin]
            T_val_hev = 3.2091e5 * (t_ns ** -0.447) * (y_s ** -0.438483)
            T[i] = T_val_hev * KELVIN_PER_HEV
            
        else:
            # Unshocked Cold Material
            rho[i] = 19.32
            p[i] = 1e-6
            u[i] = 0.0
            T[i] = 300.0  # 300 K initial temperature
            
    return {"density": rho, "pressure": p, "velocity": u, "temperature": T, "m_f": m_f, "m_s": m_s}

# =============================================================================
# Data Orchestration
# =============================================================================

def get_data():
    import pickle
    
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
    
    history = None
    if history is None:
        # Run simulation with N=200 to keep it extremely fast but highly resolved
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
# Main Plotting & Interactive Construction
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
    
    # 3) Fetch Self-Similar Patched Fits
    fits = evaluate_piecewise_fits(mass_grid, t_init)
    fit_rho = fits["density"]
    fit_p = fits["pressure"] / p_scale
    fit_u = fits["velocity"] / u_scale
    fit_T = fits["temperature"]
    
    # Draw Lines
    # Density Subplot
    line_sim_rho, = ax_rho.plot(sim_m, sim_rho, 'b-', lw=2.2, label='Simulation')
    line_men_rho, = ax_rho.plot(mass_grid, men_rho, 'm--', lw=1.8, label='Menahem Patched')
    line_fit_rho, = ax_rho.plot(mass_grid, fit_rho, 'g-', lw=2.2, label='Analytic Fits')
    
    # Pressure Subplot
    line_sim_p, = ax_p.plot(sim_m, sim_p, 'b-', lw=2.2, label='Simulation')
    line_men_p, = ax_p.plot(mass_grid, men_p, 'm--', lw=1.8, label='Menahem Patched')
    line_fit_p, = ax_p.plot(mass_grid, fit_p, 'g-', lw=2.2, label='Analytic Fits')
    
    # Velocity Subplot
    line_sim_u, = ax_u.plot(sim_m, sim_u, 'b-', lw=2.2, label='Simulation')
    line_men_u, = ax_u.plot(mass_grid, men_u, 'm--', lw=1.8, label='Menahem Patched')
    line_fit_u, = ax_u.plot(mass_grid, fit_u, 'g-', lw=2.2, label='Analytic Fits')
    
    # Temperature Subplot
    line_sim_T, = ax_T.plot(sim_m, sim_T, 'b-', lw=2.2, label='Simulation')
    line_men_T, = ax_T.plot(mass_grid, men_T, 'm--', lw=1.8, label='Menahem Patched')
    line_fit_T, = ax_T.plot(mass_grid, fit_T, 'g-', lw=2.2, label='Analytic Fits')
    
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
    
    title = fig.suptitle(f"Piecewise Patched Profile Verification\nTime: {t_init*1e9:.3f} ns", fontsize=14, fontweight='bold')
    
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
        fit_vals = evaluate_piecewise_fits(mass_grid, t_actual)
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
        
        title.set_text(f"Piecewise Patched Profile Verification\nTime: {t_actual*1e9:.3f} ns")
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    
    # =============================================================================
    # GIF Generation
    # =============================================================================
    gif_dir = _REPO_ROOT / "results" / "ictt" / "evolution"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / "fig_8_piecewise_ablation_fits.gif"
    
    print(f"4) Saving verification animation to {gif_path}...")
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

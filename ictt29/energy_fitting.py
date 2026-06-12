# project3_code/ictt29/energy_fitting.py
"""
Energy integration, fitting, and verification script.

Compares cumulative total energy (algebraic vs trapezoidal, exact vs fit) 
in both Subsonic Heat Wave and Piston Shock regimes. Plots:
1. Dimensionless self-similar energy comparison and relative errors (semi-log)
2. Dimensional verification vs mass coordinate at 0.5, 0.75, and 1.0 of t_max
3. Total energy in the front as a function of time (Simulation vs Solver vs Fit Algebraic vs Fit Trapezoidal)
"""
from __future__ import annotations

import os
import sys
sys.setrecursionlimit(200000)
import pickle
from pathlib import Path
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

from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_9_CONSTANT_FLUX,
    PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
)
from project3_code.ictt29.data_loader import get_sim_history, get_sub_similarity_solver, get_shock_similarity_solver
from project3_code.ictt29.sub_fitting import perform_subsonic_fitting, fit_by_params as sub_fit_by_params
from project3_code.ictt29.shock_fitting import perform_shock_fitting, fit_by_params as shock_fit_by_params

# Simple power law function for curve fitting E_trapz_exact directly
def simple_power_law(y, A, B):
    return A * (y ** B)

def get_plot_paths(case_label: str) -> dict[str, str]:
    out_dir = Path("results/ictt") / case_label / "energy_fitting"
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dimensionless": str(out_dir / "dimensionless_energy_comparison.png"),
        "dimensional": str(out_dir / "dimensional_energy_comparison.png"),
        "time_evolution": str(out_dir / "energy_vs_time.png")
    }

def analyze_subsonic_energy(solver, params_sub, num_points=2000):
    """Computes exact and fit dimensionless subsonic energy profiles."""
    xsi_vec = solver.get_xsi_grid(xsi_f=solver.xsi_f, fac=2.0)
    result = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    V_ex, U_ex, P_ex, S_ex = result["V"], result["U"], result["P"], result["S"]
    valid = np.isfinite(V_ex) & np.isfinite(U_ex) & np.isfinite(P_ex) & np.isfinite(S_ex) & (V_ex > 0)
    
    xsi_vec = xsi_vec[valid]
    V_ex, U_ex, P_ex, S_ex = V_ex[valid], U_ex[valid], P_ex[valid], S_ex[valid]
    y_grid = xsi_vec / solver.xsi_f
    
    # 1. Exact Solver Integration
    integrand_ex = P_ex * V_ex / solver.r + 0.5 * U_ex**2
    etot_trapz_ex = scipy.integrate.cumtrapz(y=integrand_ex, x=xsi_vec, initial=0.0)
    
    denom = 2.0 * solver.c2 - solver.c
    integral_ex = (-S_ex - P_ex * U_ex - solver.c * xsi_vec * integrand_ex) / denom
    etot_alg_ex = integral_ex - integral_ex[0]
    
    # 2. Fit Profiles Integration (direct substitution)
    T_fit, P_fit, U_fit, rho_fit = sub_fit_by_params(y_grid, params_sub)
    V_fit = 1.0 / rho_fit
    
    Pp_fit = np.gradient(P_fit, xsi_vec)
    Vp_fit = np.gradient(V_fit, xsi_vec)
    S_fit = -P_fit**(solver.n - 1.0) * V_fit**(solver.q) * (V_fit * Pp_fit + solver.k * P_fit * Vp_fit)
    
    integrand_fit = P_fit * V_fit / solver.r + 0.5 * U_fit**2
    etot_trapz_fit = scipy.integrate.cumtrapz(y=integrand_fit, x=xsi_vec, initial=0.0)
    
    integral_fit = (-S_fit - P_fit * U_fit - solver.c * xsi_vec * integrand_fit) / denom
    etot_alg_fit = integral_fit - integral_fit[0]
    
    # 3. Direct Power-law Fit to trapz_exact
    popt_trapz, _ = curve_fit(simple_power_law, y_grid, etot_trapz_ex, p0=[etot_trapz_ex[-1], 1.5])
    etot_trapz_power_law = simple_power_law(y_grid, *popt_trapz)
    
    return {
        "xsi_vec": xsi_vec,
        "y_grid": y_grid,
        "trapz_ex": etot_trapz_ex,
        "alg_ex": etot_alg_ex,
        "trapz_fit": etot_trapz_fit,
        "alg_fit": etot_alg_fit,
        "trapz_power_law": etot_trapz_power_law,
        "popt_trapz": popt_trapz
    }

def analyze_shock_energy(solver, params_shock, num_points=2000):
    """Computes exact and fit dimensionless piston shock energy profiles."""
    xsi_vec = np.linspace(1e-10, solver.xsi_s, num_points)
    V_ex, U_ex, P_ex = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    valid = np.isfinite(V_ex) & np.isfinite(U_ex) & np.isfinite(P_ex) & (V_ex > 0)
    xsi_vec = xsi_vec[valid]
    V_ex, U_ex, P_ex = V_ex[valid], U_ex[valid], P_ex[valid]
    y_grid = xsi_vec / solver.xsi_s
    
    # 1. Exact Solver Integration
    integrand_ex = P_ex * V_ex / solver.r + 0.5 * U_ex**2
    etot_trapz_ex = scipy.integrate.cumtrapz(y=integrand_ex, x=xsi_vec, initial=0.0)
    
    denom = solver.tau * (3.0 - solver.omega) + 2.0
    etot_alg_ex = ((1.0 - solver.omega) * (2.0 + solver.tau) * xsi_vec * integrand_ex - 
                   (2.0 - solver.omega) * (P_ex * U_ex - U_ex[0])) / denom
    
    # 2. Fit Profiles Integration (direct substitution)
    T_fit, P_fit, U_fit, rho_fit = shock_fit_by_params(y_grid, params_shock)
    V_fit = 1.0 / rho_fit
    
    integrand_fit = P_fit * V_fit / solver.r + 0.5 * U_fit**2
    etot_trapz_fit = scipy.integrate.cumtrapz(y=integrand_fit, x=xsi_vec, initial=0.0)
    
    etot_alg_fit = ((1.0 - solver.omega) * (2.0 + solver.tau) * xsi_vec * integrand_fit - 
                    (2.0 - solver.omega) * (P_fit * U_fit - U_fit[0])) / denom
    
    # 3. Direct Power-law Fit to trapz_exact
    popt_trapz, _ = curve_fit(simple_power_law, y_grid, etot_trapz_ex, p0=[etot_trapz_ex[-1], 1.5])
    etot_trapz_power_law = simple_power_law(y_grid, *popt_trapz)
    
    return {
        "xsi_vec": xsi_vec,
        "y_grid": y_grid,
        "trapz_ex": etot_trapz_ex,
        "alg_ex": etot_alg_ex,
        "trapz_fit": etot_trapz_fit,
        "alg_fit": etot_alg_fit,
        "trapz_power_law": etot_trapz_power_law,
        "popt_trapz": popt_trapz
    }

def plot_dimensionless_energy(sub_data, shock_data, path, case_title):
    """Plot cumulative energy profile comparisons and relative errors in semi-log scale."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel 0,0: Subsonic Profiles
    ax = axes[0, 0]
    ax.plot(sub_data["y_grid"], sub_data["trapz_ex"], 'b-', label='Trapz Exact', lw=2.5)
    ax.plot(sub_data["y_grid"], sub_data["alg_ex"], 'r--', label='Alg Exact', lw=2)
    ax.plot(sub_data["y_grid"], sub_data["trapz_fit"], 'g:', label='Trapz Fit (from profiles)', lw=2)
    ax.plot(sub_data["y_grid"], sub_data["alg_fit"], 'c-.', label='Alg Fit (from profiles)', lw=2)
    popt = sub_data["popt_trapz"]
    ax.plot(sub_data["y_grid"], sub_data["trapz_power_law"], 'm--', 
            label=f'Direct Fit: {popt[0]:.4f}$y^{{{popt[1]:.3f}}}$', lw=1.5)
    ax.set_ylabel("Dimensionless Cumulative Energy $E_{\\text{cum}}(y)$", fontsize=13)
    ax.set_xlabel("Normalized coordinate $y = \\xi / \\xi_f$", fontsize=13)
    ax.set_title("Subsonic Heat Wave: Cumulative Energy Profiles", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax.tick_params(labelsize=11)
    
    # Panel 0,1: Subsonic Relative Errors
    ax = axes[0, 1]
    err_trapz_fit = np.abs((sub_data["trapz_fit"] - sub_data["trapz_ex"]) / (sub_data["trapz_ex"] + 1e-15))
    err_alg_fit = np.abs((sub_data["alg_fit"] - sub_data["alg_ex"]) / (sub_data["alg_ex"] + 1e-15))
    err_trapz_direct = np.abs((sub_data["trapz_power_law"] - sub_data["trapz_ex"]) / (sub_data["trapz_ex"] + 1e-15))
    err_alg_vs_trapz = np.abs((sub_data["alg_ex"] - sub_data["trapz_ex"]) / (sub_data["trapz_ex"] + 1e-15))
    
    ax.plot(sub_data["y_grid"], err_trapz_fit, 'g-', label='Trapz Fit vs Trapz Exact')
    ax.plot(sub_data["y_grid"], err_alg_fit, 'c-', label='Alg Fit vs Alg Exact')
    ax.plot(sub_data["y_grid"], err_trapz_direct, 'm-', label='Direct Power-Law vs Trapz Exact')
    ax.plot(sub_data["y_grid"], err_alg_vs_trapz, 'r--', label='Alg Exact vs Trapz Exact')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1.0)
    ax.set_ylabel("Relative Error", fontsize=13)
    ax.set_xlabel("Normalized coordinate $y = \\xi / \\xi_f$", fontsize=13)
    ax.set_title("Subsonic Heat Wave: Relative Errors (semi-log)", fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc='lower left', fontsize=11)
    ax.tick_params(labelsize=11)
    
    # Panel 1,0: Shock Profiles
    ax = axes[1, 0]
    ax.plot(shock_data["y_grid"], shock_data["trapz_ex"], 'b-', label='Trapz Exact', lw=2.5)
    ax.plot(shock_data["y_grid"], shock_data["alg_ex"], 'r--', label='Alg Exact', lw=2)
    ax.plot(shock_data["y_grid"], shock_data["trapz_fit"], 'g:', label='Trapz Fit (from profiles)', lw=2)
    ax.plot(shock_data["y_grid"], shock_data["alg_fit"], 'c-.', label='Alg Fit (from profiles)', lw=2)
    popt = shock_data["popt_trapz"]
    ax.plot(shock_data["y_grid"], shock_data["trapz_power_law"], 'm--', 
            label=f'Direct Fit: {popt[0]:.4f}$y^{{{popt[1]:.3f}}}$', lw=1.5)
    ax.set_ylabel("Dimensionless Cumulative Energy $E_{\\text{cum}}(y)$", fontsize=13)
    ax.set_xlabel("Normalized coordinate $y = \\xi / \\xi_s$", fontsize=13)
    ax.set_title("Piston Shock: Cumulative Energy Profiles", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax.tick_params(labelsize=11)
    
    # Panel 1,1: Shock Relative Errors
    ax = axes[1, 1]
    err_trapz_fit = np.abs((shock_data["trapz_fit"] - shock_data["trapz_ex"]) / (shock_data["trapz_ex"] + 1e-15))
    err_alg_fit = np.abs((shock_data["alg_fit"] - shock_data["alg_ex"]) / (shock_data["alg_ex"] + 1e-15))
    err_trapz_direct = np.abs((shock_data["trapz_power_law"] - shock_data["trapz_ex"]) / (shock_data["trapz_ex"] + 1e-15))
    err_alg_vs_trapz = np.abs((shock_data["alg_ex"] - shock_data["trapz_ex"]) / (shock_data["trapz_ex"] + 1e-15))
    
    ax.plot(shock_data["y_grid"], err_trapz_fit, 'g-', label='Trapz Fit vs Trapz Exact')
    ax.plot(shock_data["y_grid"], err_alg_fit, 'c-', label='Alg Fit vs Alg Exact')
    ax.plot(shock_data["y_grid"], err_trapz_direct, 'm-', label='Direct Power-Law vs Trapz Exact')
    ax.plot(shock_data["y_grid"], err_alg_vs_trapz, 'r--', label='Alg Exact vs Trapz Exact')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1.0)
    ax.set_ylabel("Relative Error", fontsize=13)
    ax.set_xlabel("Normalized coordinate $y = \\xi / \\xi_s$", fontsize=13)
    ax.set_title("Piston Shock: Relative Errors (semi-log)", fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc='lower left', fontsize=11)
    ax.tick_params(labelsize=11)
    
    plt.suptitle(f"Dimensionless Cumulative Energy Comparisons & Fitting Relative Errors\n{case_title}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved dimensionless energy comparison to {path}")

def plot_dimensional_comparison(history, sub_solver, sub_data, shock_solver, shock_data, path, case_title):
    """Plot dimensional physical cumulative energy vs mass coordinate at 0.5, 0.75, and 1.0 of t_max."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    
    t_max = max(history.t)
    target_times = [0.5 * t_max, 0.75 * t_max, t_max]
    colors = ["royalblue", "darkorange", "crimson"]
    
    # Subsonic Heat Wave
    ax = axes[0]
    for i, (t_target, color) in enumerate(zip(target_times, colors)):
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        t_actual = history.t[idx_sim]
        m_sim = history.m[idx_sim]
        u_sim = history.u[idx_sim]
        e_sim = history.e[idx_sim]
        
        m_front = sub_solver.ablated_mass(time=t_actual)
        sub_mask = m_sim <= m_front
        m_sim_sub = m_sim[sub_mask]
        u_sim_sub = u_sim[sub_mask]
        e_sim_sub = e_sim[sub_mask]
        
        # Simulation cumulative energy
        dm = np.diff(m_sim_sub, prepend=0.0)
        E_sim_cum = np.cumsum((0.5 * u_sim_sub**2 + e_sim_sub) * dm)
        
        # Solver & Fit profiles on Lagrangian grid
        mass_solver = np.linspace(1e-12, m_front, 1000)
        y = mass_solver / m_front
        
        # Linearly interpolate dimensionless algebraic curves onto the target grid
        etot_ex_interp = np.interp(y, sub_data["y_grid"], sub_data["alg_ex"])
        etot_fit_interp = np.interp(y, sub_data["y_grid"], sub_data["alg_fit"])
        
        factor = sub_solver._energy_temporal_factor(time=t_actual)
        E_ex_cum = factor * etot_ex_interp
        E_fit_cum = factor * etot_fit_interp
        
        show_lbl = i == 0
        ax.plot(m_sim_sub * 1e3, E_sim_cum, '-', color=color, alpha=0.7, 
                label=f"Simulation ({t_target*1e9:.1f} ns)" if show_lbl else None)
        ax.plot(mass_solver * 1e3, E_ex_cum, '--', color='black', lw=1.8, 
                label="Exact Solver" if show_lbl else None)
        ax.plot(mass_solver * 1e3, E_fit_cum, ':', color='forestgreen', lw=1.8, 
                label="Analytic Fit" if show_lbl else None)
        
    ax.set_ylabel("Integrated Cumulative Energy [erg/cm$^2$]")
    ax.set_xlabel("Lagrangian Mass coordinate $m$ [mg/cm$^2$]")
    ax.set_title("Subsonic Heat Wave Ablation Front Region", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Custom Legend
    time_handles = [Line2D([0], [0], color=colors[k], lw=2, label=f"{target_times[k]*1e9:.1f} ns") for k in range(3)]
    style_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Exact Solver'),
        Line2D([0], [0], color='forestgreen', lw=2, linestyle=':', label='Analytic Fit'),
    ]
    ax.legend(handles=time_handles + style_handles, loc="upper left")
    
    # Piston Shock
    ax = axes[1]
    for i, (t_target, color) in enumerate(zip(target_times, colors)):
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        t_actual = history.t[idx_sim]
        m_sim = history.m[idx_sim]
        u_sim = history.u[idx_sim]
        e_sim = history.e[idx_sim]
        
        m_front = shock_solver.shocked_mass(time=t_actual)
        shock_mask = m_sim <= m_front
        m_sim_shock = m_sim[shock_mask][:-2]  # trim boundary transients
        u_sim_shock = u_sim[shock_mask][:-2]
        e_sim_shock = e_sim[shock_mask][:-2]
        
        dm = np.diff(m_sim_shock, prepend=0.0)
        E_sim_cum = np.cumsum((0.5 * u_sim_shock**2 + e_sim_shock) * dm)
        
        mass_solver = np.linspace(1e-12, m_front, 1000)
        y = mass_solver / m_front
        
        etot_ex_interp = np.interp(y, shock_data["y_grid"], shock_data["alg_ex"])
        etot_fit_interp = np.interp(y, shock_data["y_grid"], shock_data["alg_fit"])
        
        factor = shock_solver._energy_temporal_factor(time=t_actual)
        E_ex_cum = factor * etot_ex_interp
        E_fit_cum = factor * etot_fit_interp
        
        show_lbl = i == 0
        ax.plot(m_sim_shock * 1e3, E_sim_cum, '-', color=color, alpha=0.7)
        ax.plot(mass_solver * 1e3, E_ex_cum, '--', color='black', lw=1.8)
        ax.plot(mass_solver * 1e3, E_fit_cum, ':', color='forestgreen', lw=1.8)
        
    ax.set_ylabel("Integrated Cumulative Energy [erg/cm$^2$]")
    ax.set_xlabel("Lagrangian Mass coordinate $m$ [mg/cm$^2$]")
    ax.set_title("Piston Shock Wave Shocked Region", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Dimensional Cumulative Energy Comparison\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved dimensional energy comparison to {path}")

def plot_time_evolution(history, sub_solver, sub_data, shock_solver, shock_data, path, case_title):
    """Plot total energy at the front as a function of time (Simulation vs Solver vs Fit Alg vs Fit Trapz)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # We skip first few time steps to avoid initial simulation transients and ODE singularity at t=0
    times = history.t[5:]
    times_ns = times * 1e9
    
    # Subsonic Heat Wave
    ax = axes[0]
    E_sim = []
    E_sol = []
    E_alg_fit = []
    E_trapz_fit = []
    
    for t_val in times:
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_val))
        m_sim = history.m[idx_sim]
        u_sim = history.u[idx_sim]
        e_sim = history.e[idx_sim]
        
        m_front = sub_solver.ablated_mass(time=t_val)
        mask = m_sim <= m_front
        m_sub = m_sim[mask]
        u_sub = u_sim[mask]
        e_sub = e_sim[mask]
        
        dm = np.diff(m_sub, prepend=0.0)
        E_sim.append(np.sum((0.5 * u_sub**2 + e_sub) * dm))
        
        E_sol.append(sub_solver.total_energy(time=t_val))
        
        factor = sub_solver._energy_temporal_factor(time=t_val)
        E_alg_fit.append(factor * sub_data["alg_fit"][-1])
        E_trapz_fit.append(factor * sub_data["trapz_fit"][-1])
        
    ax.plot(times_ns, E_sim, 'k-', label='Simulation', lw=2.5)
    ax.plot(times_ns, E_sol, 'b--', label='Solver (exact)', lw=2.0)
    ax.plot(times_ns, E_alg_fit, 'c-.', label='Fit Algebraic', lw=2.0)
    ax.plot(times_ns, E_trapz_fit, 'g:', label='Fit Trapezoidal', lw=2.0)
    ax.set_ylabel("Total Front Energy [erg/cm$^2$]")
    ax.set_xlabel("Time [ns]")
    ax.set_title("Subsonic Heat Wave: Energy vs Time", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Piston Shock
    ax = axes[1]
    E_sim = []
    E_sol = []
    E_alg_fit = []
    E_trapz_fit = []
    
    for t_val in times:
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_val))
        m_sim = history.m[idx_sim]
        u_sim = history.u[idx_sim]
        e_sim = history.e[idx_sim]
        
        m_front = shock_solver.shocked_mass(time=t_val)
        mask = m_sim <= m_front
        m_shock = m_sim[mask][:-2]  # trim transient
        u_shock = u_sim[mask][:-2]
        e_shock = e_sim[mask][:-2]
        
        dm = np.diff(m_shock, prepend=0.0)
        E_sim.append(np.sum((0.5 * u_shock**2 + e_shock) * dm))
        
        E_sol.append(shock_solver.total_energy(time=t_val))
        
        factor = shock_solver._energy_temporal_factor(time=t_val)
        E_alg_fit.append(factor * shock_data["alg_fit"][-1])
        E_trapz_fit.append(factor * shock_data["trapz_fit"][-1])
        
    ax.plot(times_ns, E_sim, 'k-', label='Simulation', lw=2.5)
    ax.plot(times_ns, E_sol, 'b--', label='Solver (exact)', lw=2.0)
    ax.plot(times_ns, E_alg_fit, 'c-.', label='Fit Algebraic', lw=2.0)
    ax.plot(times_ns, E_trapz_fit, 'g:', label='Fit Trapezoidal', lw=2.0)
    ax.set_ylabel("Total Front Energy [erg/cm$^2$]")
    ax.set_xlabel("Time [ns]")
    ax.set_title("Piston Shock: Energy vs Time", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    plt.suptitle(f"Total Front Integrated Energy Evolution over Time\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved total energy vs time evolution to {path}")

def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run energy integration analysis, fits, and comparisons for a given preset."""
    print("=" * 80)
    print(f"PROCESSING ENERGY FITTING: {preset_name} -> {case_label}")
    print("=" * 80)
    
    # 1. Load History and Solvers
    case, history = get_sim_history(preset_name, case_label)
    sub_solver = get_sub_similarity_solver(case, case_label)
    shock_solver = get_shock_similarity_solver(case, case_label)
    
    # 2. Get Fits
    params_sub = perform_subsonic_fitting(sub_solver)
    params_shock = perform_shock_fitting(shock_solver)
    
    # 3. Analyze Energy
    sub_data = analyze_subsonic_energy(sub_solver, params_sub)
    shock_data = analyze_shock_energy(shock_solver, params_shock)
    
    # 4. Generate Figures
    paths = get_plot_paths(case_label)
    
    plot_dimensionless_energy(sub_data, shock_data, paths["dimensionless"], case_title)
    plot_dimensional_comparison(history, sub_solver, sub_data, shock_solver, shock_data, paths["dimensional"], case_title)
    plot_time_evolution(history, sub_solver, sub_data, shock_solver, shock_data, paths["time_evolution"], case_title)
    
    print(f"Energy analysis for preset {preset_name} completed successfully.")

def main():
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
    print("\nAll custom energy simulations, calculations, fitting, and plots completed successfully!")

if __name__ == "__main__":
    main()

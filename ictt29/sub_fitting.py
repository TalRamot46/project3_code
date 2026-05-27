# ictt29/sub_fitting.py
"""
Subsonic self-similar profile fitting and dimensional comparison script.

Produces:
1. 2x2 self-similar profile fits (T, rho_EOS, P, U) with LaTeX formula + error labels
2. Semi-log relative error plot for all 4 profiles
3. Velocity candidate comparison (6 fits side-by-side)
4. Dimensional subsonic comparison (solver vs fit vs rad-hydro)
"""
from __future__ import annotations

import os
import sys
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

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
)
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _heat_kwargs_from_case,
)
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

from subsonic_heat_wave import SubsonicHeatWave

RUN_GIFS = False  # Set to True to generate animated evolution GIFs (which takes time)

def get_cached_subsonic_solver(case, case_label):
    """Solve subsonic similarity ODEs once and cache the solver object (with found xsi_f)."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver_cache_path = cache_dir / f"{case_label}_similarity_solver.pkl"
    
    if solver_cache_path.exists():
        print(f"Loading cached subsonic similarity solver from {solver_cache_path}...")
        try:
            with open(solver_cache_path, "rb") as f:
                solver = pickle.load(f)
            # Re-bind the ODE solver which contains method callbacks
            solver.ode_solver = scipy.integrate.ode(solver.fode).set_integrator(solver.ode_scheme)
            print("Similarity solver loaded successfully.")
            return solver
        except Exception as e:
            print(f"Failed to load solver cache: {e}. Re-solving subsonic ODEs...")
            
    print("Solving subsonic similarity ODEs (finding xsi_f via shooting method)...")
    heat_kwargs = _heat_kwargs_from_case(case)
    solver = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
    
    # Save cache by removing ode_solver temporarily to avoid pickling issues
    try:
        ode_solver = solver.ode_solver
        del solver.ode_solver
        with open(solver_cache_path, "wb") as f:
            pickle.dump(solver, f, protocol=pickle.HIGHEST_PROTOCOL)
        solver.ode_solver = ode_solver
        print(f"Saved subsonic similarity solver to cache: {solver_cache_path}")
    except Exception as e:
        print(f"Failed to save solver cache: {e}")
        
    return solver


def evaluate_subsonic_fits(mass_grid, t_sec, solver, popt_T, popt_P, best_u):
    """
    Computes the explicit self-similar analytical fits for subsonic regime in CGS units.
    Uses the optimized velocity fit.
    """
    t_ns = t_sec * 1e9
    m_f = solver.xsi_f / (solver.A**solver.a * solver.B**solver.b * (1e-9)**solver.c) * (t_ns ** 0.515625)
    
    n = len(mass_grid)
    rho = np.zeros(n)
    p = np.zeros(n)
    u = np.zeros(n)
    T = np.zeros(n)
    
    # Subsonic Fits
    r_val = 0.25
    beta_val = 1.6
    mu_val = 0.14
    f_cgs = 3.4e13
    
    for i, m in enumerate(mass_grid):
        if m < m_f:
            y = m / m_f
            
            # 1. Temperature fit in HeV
            T_hev = ((1.0 - y) * (1.0 + popt_T[0] * y)) ** (10.0 / 39.0)
            T[i] = T_hev * 1.160451812e6  # Kelvin
            
            # 2. Pressure fit in MBar
            pre_p = solver.Pf * (solver.A**solver.a3) * (solver.B**solver.b3)
            p_val_mbar = (pre_p * 1e-12) * (t_ns ** -0.447917) * (popt_P[0] * y**popt_P[2] + popt_P[1] * y**(popt_P[2]+popt_P[3]))
            p[i] = p_val_mbar * 1e12  # Barye
            
            # 3. Velocity fit in km/s
            coeff = (solver.A**solver.a2) * (solver.B**solver.b2) * 1e-5
            U_dim_kms = coeff * best_u["func"](y, *best_u["popt"]) * (t_ns ** 0.036458)
            u[i] = U_dim_kms * 1e5  # cm/s
            
            # 4. Density derived via CGS EOS
            rho[i] = ((r_val * f_cgs * T_hev**beta_val) / np.maximum(p[i], 1e-15)) ** (1.0 / (mu_val - 1.0))
        else:
            # Outside subsonic front
            rho[i] = 19.32
            p[i] = 1e-6
            u[i] = 0.0
            T[i] = 300.0
            
    return {"density": rho, "pressure": p, "velocity": u, "temperature": T, "m_f": m_f}


def perform_subsonic_fitting(solver):
    y_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
    xsi_vec = y_grid * solver.xsi_f
    profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    P_val = profiles["P"]
    T_val = profiles["T"]
    U_val = profiles["U"]
    V_val = profiles["V"]
    rho_val = 1.0 / V_val
    
    valid_idx = (y_grid > 0.005) & np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(T_val)
    y_valid = y_grid[valid_idx]
    T_valid = T_val[valid_idx]
    P_valid = P_val[valid_idx]
    U_valid = U_val[valid_idx]
    rho_valid = rho_val[valid_idx]
    
    mask = y_valid < 0.99
    y_valid_masked = y_valid[mask]
    T_valid_masked = T_valid[mask]
    rho_valid_masked = rho_valid[mask]
    
    # Fits
    def smith_approximation(y, R):
        return ((1.0 - y) * (1.0 + R * y))**(10.0 / 39.0)
        
    def power_law_origin(y, a, b, c, d):
        return a * y**c + b * y**(c+d)
        
    popt_T, _ = curve_fit(smith_approximation, y_valid_masked, T_valid_masked, p0=[0.5])
    popt_P, _ = curve_fit(power_law_origin, y_valid, P_valid, p0=[0.355, 0.5, 0.04, 2.3])
    
    # Velocity fits
    def fit_u_1(y, c, d): return c * (1.0 - y)**d
    def fit_u_2(y, c, b): return c * (1.0 - y) / (1.0 + b * y)
    def fit_u_3(y, c, b): return c * (1.0 - y) * (y**(-b))
    def fit_u_4(y, c, a, b): return c * (1.0 - y**a) * (y**(-b))
    def fit_u_5(y, c, a, b): return c * (1.0 - y) * (y**a + y**(-b))
    def fit_u_6(y, c, a, b): return c * (1.0 - y**a) / (1.0 + b * y)
    
    candidates = [
        {"id": 1, "func": fit_u_1, "name": "Power Law: $c(1-y)^d$", "latex": r"U(y) \approx %.5f (1 - y)^{%.5f}", "p0": [U_valid[0], 0.5]},
        {"id": 2, "func": fit_u_2, "name": "Rational: $c(1-y)/(1+b y)$", "latex": r"U(y) \approx \frac{%.5f (1 - y)}{1 + %.5f y}", "p0": [U_valid[0], 0.5]},
        {"id": 3, "func": fit_u_3, "name": "Singular 2P: $c(1-y)y^{-b}$", "latex": r"U(y) \approx %.5f (1 - y) y^{-%.5f}", "p0": [-1.0, 0.3]},
        {"id": 4, "func": fit_u_4, "name": "Singular 3P: $c(1-y^a)y^{-b}$", "latex": r"U(y) \approx %.5f (1 - y^{%.5f}) y^{-%.5f}", "p0": [-1.0, 1.0, 0.3]},
        {"id": 5, "func": fit_u_5, "name": "User 3P: $c(1-y)(y^a + y^{-b})$", "latex": r"U(y) \approx %.5f (1 - y) (y^{%.5f} + y^{-%.5f)}", "p0": [-1.0, 1.0, 0.3]},
        {"id": 6, "func": fit_u_6, "name": "Rational Frac: $c(1-y^a)/(1+b y)$", "latex": r"U(y) \approx \frac{%.5f (1 - y^{%.5f})}{1 + %.5f y}", "p0": [U_valid[0], 1.0, 1.0]}
    ]
    
    mask_u = y_valid < 0.99
    y_valid_u = y_valid[mask_u]
    U_valid_u = U_valid[mask_u]
    
    best_u = None
    min_avg_err = float("inf")
    fits_u = {}
    
    for cand in candidates:
        try:
            popt, _ = curve_fit(cand["func"], y_valid_u, U_valid_u, p0=cand["p0"], maxfev=10000)
            U_fit = cand["func"](y_valid, *popt)
            U_fit_u = cand["func"](y_valid_u, *popt)
            rel_err_u = np.abs((U_fit_u - U_valid_u) / (U_valid_u + 1e-15)) * 100
            avg_err = np.mean(rel_err_u)
            max_err = np.max(rel_err_u)
            
            fits_u[cand["id"]] = (popt, U_fit, avg_err, max_err, cand["name"], cand["latex"])
            
            if avg_err < min_avg_err:
                min_avg_err = avg_err
                best_u = {
                    "id": cand["id"],
                    "popt": popt,
                    "func": cand["func"],
                    "name": cand["name"],
                    "latex": cand["latex"],
                    "avg_err": avg_err,
                    "max_err": max_err,
                    "fit_val": U_fit
                }
        except Exception as e:
            print(f"Subsonic velocity fit {cand['id']} failed: {e}")
            fits_u[cand["id"]] = (None, None, 0.0, 0.0, cand["name"], cand["latex"])
            
    return solver, y_grid, y_valid, T_valid, P_valid, U_valid, rho_valid, popt_T, popt_P, best_u, fits_u


def plot_and_fit_self_similar(solver, self_similar_path, standalone_path, relative_errors_path, case_title):
    """Plot similarity profiles, fits, error curves using the pre-solved solver."""
    # Run the exact fitting pipeline
    _, y_grid, y_valid, T_valid, P_valid, U_valid, rho_valid, popt_T, popt_P, best_u, fits_u = perform_subsonic_fitting(solver)
    
    # ----------------------------------------------------
    # Re-evaluate all chosen fits on y_valid
    # ----------------------------------------------------
    T_fit = ((1.0 - y_valid) * (1.0 + popt_T[0] * y_valid)) ** (10.0 / 39.0)
    P_fit = popt_P[0] * y_valid**popt_P[2] + popt_P[1] * y_valid**(popt_P[2]+popt_P[3])
    U_fit = best_u["fit_val"]
    
    r_val, beta_val, mu_val = 0.25, 1.6, 0.14
    rho_fit = ((r_val * T_fit**beta_val) / P_fit) ** (1.0 / (mu_val - 1.0))
    
    # Relative Errors (restricted to y < 0.99 for T and rho to avoid numerical singularities at the front boundary)
    mask_bulk = y_valid < 0.99
    y_bulk = y_valid[mask_bulk]
    
    err_T = np.abs((T_fit[mask_bulk] - T_valid[mask_bulk]) / T_valid[mask_bulk]) * 100
    err_rho = np.abs((rho_fit[mask_bulk] - rho_valid[mask_bulk]) / rho_valid[mask_bulk]) * 100
    err_P = np.abs((P_fit - P_valid) / P_valid) * 100
    err_U = np.abs((U_fit[mask_bulk] - U_valid[mask_bulk]) / (U_valid[mask_bulk] + 1e-15)) * 100
    
    avg_T, max_T = np.mean(err_T), np.max(err_T)
    avg_rho, max_rho = np.mean(err_rho), np.max(err_rho)
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = best_u["avg_err"], best_u["max_err"]
    
    # ----------------------------------------------------
    # 1) 2x2 Similarity Profiles Fit Plot (Task 1)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # Panel (0,0): Temperature
    ax = axes[0, 0]
    ax.plot(y_valid, T_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, T_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Temperature", fontsize=13, fontweight='bold')
    lbl_T = f"$T(y) \\approx [(1-y)(1+{popt_T[0]:.5f}y)]^{{10/39}}$\nAvg Err: {avg_T:.4f}%, Max Err: {max_T:.4f}%"
    ax.text(0.05, 0.05, lbl_T, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (0,1): Density
    ax = axes[0, 1]
    ax.plot(y_valid, rho_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, rho_fit, 'r--', label='EOS Derived Fit', lw=1.5)
    ax.set_ylabel(r"Density $\rho(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Density", fontsize=13, fontweight='bold')
    lbl_rho = r"$\rho(y) \approx \left(\frac{r \cdot T(y)^\beta}{P(y)}\right)^{\frac{1}{\mu - 1}}$" + f"\nAvg Err: {avg_rho:.4f}%, Max Err: {max_rho:.4f}%"
    ax.text(0.05, 0.05, lbl_rho, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,0): Pressure
    ax = axes[1, 0]
    ax.plot(y_valid, P_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, P_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Pressure $P(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Pressure", fontsize=13, fontweight='bold')
    lbl_P = f"$P(y) \\approx {popt_P[0]:.5f} y^{{{popt_P[2]:.5f}}} + {popt_P[1]:.5f} y^{{{popt_P[2]+popt_P[3]:.5f}}}$\nAvg Err: {avg_P:.4f}%, Max Err: {max_P:.4f}%"
    ax.text(0.05, 0.70, lbl_P, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,1): Velocity (using optimal selected fit)
    ax = axes[1, 1]
    ax.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, U_fit, 'r--', label='Optimized Fit', lw=1.5)
    ax.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax.set_title(f"Subsonic: Velocity ({best_u['name']})", fontsize=13, fontweight='bold')
    
    # Format the dynamic velocity formula in latex
    if best_u["id"] == 1:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1-y)^{{{best_u['popt'][1]:.5f}}}$"
    elif best_u["id"] == 2:
        u_formula = f"$U(y) \\approx \\frac{{{best_u['popt'][0]:.5f} (1-y)}}{{1 + {best_u['popt'][1]:.5f} y}}$"
    elif best_u["id"] == 3:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1-y) y^{{-{best_u['popt'][1]:.5f}}}$"
    elif best_u["id"] == 4:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1-y^{{{best_u['popt'][1]:.5f}}}) y^{{-{best_u['popt'][2]:.5f}}}$"
    elif best_u["id"] == 5:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1-y) (y^{{{best_u['popt'][1]:.5f}}} + y^{{-{best_u['popt'][2]:.5f}}})$"
    elif best_u["id"] == 6:
        u_formula = f"$U(y) \\approx \\frac{{{best_u['popt'][0]:.5f} (1-y^{{{best_u['popt'][1]:.5f}}} )}}{{1 + {best_u['popt'][2]:.5f} y}}$"
    
    lbl_U = u_formula + f"\nAvg Err: {avg_U:.4f}%, Max Err: {max_U:.4f}%"
    ax.text(0.05, 0.70, lbl_U, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    for ax in axes.flat:
        ax.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=9.5)
        
    fig.suptitle(f"Subsonic self-similar Profiles & Analytical Fits\n{case_title}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(self_similar_path, dpi=200)
    plt.close(fig)
    print(f"Saved self-similar subsonic profiles to {self_similar_path}")
    
    # ----------------------------------------------------
    # 2) Profiles Combined Relative Error Plot (Task 2)
    # ----------------------------------------------------
    fig_err, ax_err = plt.subplots(figsize=(10, 7.5))
    ax_err.plot(y_bulk, err_T, label=f'Temperature $T(y)$ (Avg: {avg_T:.4f}%, Max: {max_T:.4f}%)', lw=2.0)
    ax_err.plot(y_bulk, err_rho, label=f'Density $\\rho(y)$ (Avg: {avg_rho:.4f}%, Max: {max_rho:.4f}%)', lw=2.0)
    ax_err.plot(y_valid, err_P, label=f'Pressure $P(y)$ (Avg: {avg_P:.4f}%, Max: {max_P:.4f}%)', lw=2.0)
    ax_err.plot(y_bulk, err_U, label=f'Velocity $U(y)$ (Avg: {avg_U:.4f}%, Max: {max_U:.4f}%)', lw=2.0)
    
    ax_err.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_err.set_ylabel(r"Relative Error [$\%$]", fontsize=12)
    ax_err.set_yscale('log')
    ax_err.grid(True, which="both", ls=":", alpha=0.5)
    ax_err.legend(loc="best", fontsize=10.5)
    ax_err.set_title(f"Relative Errors of Subsonic self-similar Fits (semi-log)\n{case_title}", fontsize=13, fontweight='bold')
    
    fig_err.tight_layout()
    fig_err.savefig(relative_errors_path, dpi=200)
    plt.close(fig_err)
    print(f"Saved subsonic relative errors to {relative_errors_path}")
    
    # ----------------------------------------------------
    # 3) Velocity Standalone Fits Comparison Plot (Task 3)
    # ----------------------------------------------------
    fig_sa, (ax_sa1, ax_sa2) = plt.subplots(1, 2, figsize=(18, 8.5))
    
    # Left: Fits vs Numerical
    ax_sa1.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=3.0)
    colors_u = {1: 'crimson', 2: 'darkorange', 3: 'forestgreen', 4: 'darkviolet', 5: 'deeppink', 6: 'teal'}
    
    for i in range(1, 7):
        popt, U_fit, avg_err, max_err, name, latex = fits_u[i]
        if popt is not None:
            lbl = f"Fit {i}: {name}\nAvg Err: {avg_err:.3f}%, Max Err: {max_err:.3f}%"
            lw = 2.2 if i == best_u["id"] else 1.5
            ax_sa1.plot(y_valid, U_fit, colors_u[i], linestyle='--', label=lbl, lw=lw)
            
            # Right: Semi-log Error curves
            err_curve = np.abs((U_fit[mask_bulk] - U_valid[mask_bulk]) / (U_valid[mask_bulk] + 1e-15)) * 100
            ax_sa2.plot(y_bulk, err_curve, colors_u[i], label=f"Fit {i} (Avg: {avg_err:.3f}%)", lw=lw)
            
    ax_sa1.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_sa1.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax_sa1.legend(loc='best', fontsize=9.0)
    ax_sa1.grid(True, alpha=0.3)
    ax_sa1.set_title("Dimensionless Velocity $U(y)$ vs 6 Candidates", fontsize=13, fontweight='bold')
    
    ax_sa2.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_sa2.set_ylabel(r"Relative Error [$\%$]", fontsize=12)
    ax_sa2.set_yscale('log')
    ax_sa2.legend(loc='best', fontsize=9.5)
    ax_sa2.grid(True, which="both", ls=":", alpha=0.5)
    ax_sa2.set_title("Relative Errors of Velocity Fits (semi-log)", fontsize=13, fontweight='bold')
    
    fig_sa.suptitle(f"Subsonic Velocity Profile Curve Fitting & Optimization\nChosen Formal Fit: Fit {best_u['id']} ({best_u['name']})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_sa.savefig(standalone_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sa)
    print(f"Saved standalone velocity fits to {standalone_path}")


# =============================================================================
# Dimensional Subsonic Comparison (Task 5)
# =============================================================================

def load_full_rad_hydro_numerical_data() -> dict:
    """Loads the Fig 8 (tau = 0) Rad-Hydro simulation results from saved npz."""
    fn = os.path.join(r'C:\Users\TLP-001\Documents\GitHub\project3_code\rad_hydro_sim\data', 
                      'sim_data_Fig_8_comparison_\u03c40_Shussman_verification.npz')
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Missing saved full rad-hydro npz file at {fn}")
    
    loaded = np.load(fn, allow_pickle=True)
    return {
        "times": np.asarray(loaded["times"], dtype=float),
        "m": [np.asarray(arr, dtype=float) for arr in loaded["m"]],
        "x": [np.asarray(arr, dtype=float) for arr in loaded["x"]],
        "rho": [np.asarray(arr, dtype=float) for arr in loaded["rho"]],
        "p": [np.asarray(arr, dtype=float) for arr in loaded["p"]],
        "u": [np.asarray(arr, dtype=float) for arr in loaded["u"]],
        "T": [np.asarray(arr, dtype=float) for arr in loaded["T"]],
    }


def plot_subsonic_ablation_comparison(hs, numerical_data: dict, output_dir: Path):
    """Plots subsonic ablation solver vs fits vs numerical full rad-hydro data."""
    target_times = np.array([1e-9, 1.5e-9, 2e-9])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    # Add Zoomed Inset for Density near front (y in [0.8, 0.99])
    axins = axes[0].inset_axes([0.18, 0.48, 0.35, 0.35])
    
    colors = ["red", "green", "blue"]
    
    for i, t_val in enumerate(target_times):
        # Closest numerical index
        idx = np.argmin(np.abs(numerical_data["times"] - t_val))
        t_sec = numerical_data["times"][idx]
        t_ns = t_sec * 1e9
        
        # Numerical ablated mass front
        m_f_val = hs.ablated_mass(time=t_sec)
        
        # Subsonic Solver Lagrangian Grid
        mass_solver = np.linspace(1e-12, m_f_val, 300)
        sol_hs = hs.solve(mass=mass_solver, time=t_sec)
        
        # Slicing numerical simulation to the subsonic region (m <= m_f)
        m_sim = numerical_data["m"][idx]
        sub_mask = m_sim <= m_f_val
        m_sim_sub = m_sim[sub_mask]
        
        # Reconstruct fits
        y = m_sim_sub / m_f_val
        y_solver = mass_solver / m_f_val
        
        # Fits (CGS & HeV values converted for plotting)
        # Pressure (converted from MBar to Barye: 1 MBar = 1e12 Ba)
        p_fit_MBar = 7.05133 * (t_ns)**-0.44792 * (0.34866 * y**0.87714 + 0.02903 * y**21.08862)
        p_fit = p_fit_MBar * 1e12
        # Velocity (converted from km/s to cm/s: 1 km/s = 1e5 cm/s)
        u_fit_kms = -191.29403 * (t_ns)**0.03646 * (1.0 - y) / (1.0 + 4.78201 * y)
        u_fit = u_fit_kms * 1e5
        # Temperature (converted from HeV to Kelvin)
        T_fit_HeV = ((1.0 - y) * (1.0 + 0.20224 * y)) ** (10.0 / 39.0)
        T_fit = T_fit_HeV * KELVIN_PER_HEV

        # Density (g/cm^3) derived from EOS (T and P fits)
        r_val = float(hs.r)
        f_val = float(hs.f)
        beta_val = float(hs.beta)
        mu_val = float(hs.mu)
        p_fit_safe = np.maximum(p_fit, 1e-15)
        rho_fit = ((r_val * f_val * T_fit**beta_val) / p_fit_safe) ** (1.0 / (mu_val - 1.0))

        # Profile 1: Density
        axes[0].plot(m_sim_sub * 1e3, numerical_data["rho"][idx][sub_mask], 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[0].plot(mass_solver * 1e3, sol_hs["density"], '-', color=colors[i], label=f"Solver t={t_ns:.2f} ns" if i==0 else None)
        axes[0].plot(m_sim_sub * 1e3, rho_fit, '--', color=colors[i], label=f"Fit t={t_ns:.2f} ns" if i==0 else None)
        
        # Zoomed inset plotting near front (y in [0.8, 0.99])
        zoom_mask = (y >= 0.8) & (y <= 0.99)
        m_sim_zoom = m_sim_sub[zoom_mask]
        rho_sim_zoom = numerical_data["rho"][idx][sub_mask][zoom_mask]
        rho_fit_zoom = rho_fit[zoom_mask]
        
        solver_zoom_mask = (y_solver >= 0.8) & (y_solver <= 0.99)
        m_sol_zoom = mass_solver[solver_zoom_mask]
        rho_sol_zoom = sol_hs["density"][solver_zoom_mask]
        
        axins.plot(m_sim_zoom * 1e3, rho_sim_zoom, 'o', color=colors[i], markersize=2, alpha=0.5)
        axins.plot(m_sol_zoom * 1e3, rho_sol_zoom, '-', color=colors[i])
        axins.plot(m_sim_zoom * 1e3, rho_fit_zoom, '--', color=colors[i])
        
        # Profile 2: Pressure
        axes[1].plot(m_sim_sub * 1e3, numerical_data["p"][idx][sub_mask] * 1e-12, 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[1].plot(mass_solver * 1e3, sol_hs["pressure"] * 1e-12, '-', color=colors[i])
        axes[1].plot(m_sim_sub * 1e3, p_fit_MBar, '--', color=colors[i])

        # Profile 3: Velocity
        axes[2].plot(m_sim_sub * 1e3, numerical_data["u"][idx][sub_mask] * 1e-5, 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[2].plot(mass_solver * 1e3, sol_hs["velocity"] * 1e-5, '-', color=colors[i])
        axes[2].plot(m_sim_sub * 1e3, u_fit_kms, '--', color=colors[i])

        # Profile 4: Temperature
        axes[3].plot(m_sim_sub * 1e3, numerical_data["T"][idx][sub_mask] / KELVIN_PER_HEV, 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[3].plot(mass_solver * 1e3, sol_hs["temperature"] / KELVIN_PER_HEV, '-', color=colors[i])
        axes[3].plot(m_sim_sub * 1e3, T_fit_HeV, '--', color=colors[i])

    # Styling
    labels = ["Density [g/cm$^3$]", "Pressure [MBar]", "Velocity [km/s]", "Temperature [HeV]"]
    for j, ax in enumerate(axes):
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels[j], fontsize=12)
        ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm$^2$]", fontsize=12)
        if j == 0:
            ax.legend(loc="upper left")
            
    # Style inset
    axins.grid(True, alpha=0.3)
    axins.set_title("Zoom near front", fontsize=9)
    axes[0].indicate_inset_zoom(axins, edgecolor="black")
            
    plt.suptitle("Subsonic Ablation Region Verification\nSolid: Subsonic Solver, Dashed: Piecewise Analytical Fit, Circles: Full Rad-Hydro Data", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "ablation_region_comparison.png", dpi=200)
    print(f"Saved: ablation_region_comparison.png")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    case, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    
    # Output dirs
    ss_dir = Path("results/ictt/self_similar")
    dim_dir = Path("results/ictt/test fits lagrangian")
    ss_dir.mkdir(parents=True, exist_ok=True)
    dim_dir.mkdir(parents=True, exist_ok=True)
    
    # Solve subsonic similarity ODEs
    solver = get_cached_subsonic_solver(case, "constant_boundary_temperature_tau_0")
    
    # Tasks 1-3: Self-similar fits, errors, velocity candidates
    self_similar_path = str(ss_dir / "subsonic_self_similar.png")
    standalone_path = str(ss_dir / "subsonic_velocity_fits_standalone.png")
    relative_errors_path = str(ss_dir / "subsonic_relative_errors.png")
    plot_and_fit_self_similar(solver, self_similar_path, standalone_path, relative_errors_path,
                             "Constant Boundary Temperature (tau=0)")
    
    # Task 5: Dimensional subsonic comparison
    numerical_data = load_full_rad_hydro_numerical_data()
    plot_subsonic_ablation_comparison(solver, numerical_data, dim_dir)
    
    print("\nSubsonic fitting completed successfully!")


if __name__ == "__main__":
    main()

# ictt29/plot_piston_shock.py
"""
verification and similarity profile fitting script for the Piston Shock regime.

Compiles and compares:
1. 1D Rad-Hydro Simulation (hydro-only scenario).
2. PistonShock reference solver (piston-driven shock wave).

Outputs generated inside results/ictt/ (split by graph types):
- xt/ Space-time trajectory and fronts (PNG)
- material_hydro/ e, u, p, rho vs mass coordinate m (PNG)
- self_similar/ Dimensionless similarity profiles R, P, U, T with curve-fits (PNG)
- self_similar/ Dimensionless velocity fits standalone comparing alternative fits (PNG)
- evolution/ Animated 5-way GIFs (Simulation vs PistonShock).
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
    PRESET_CONSTANT_PRESSURE,
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE,
)
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _shock_kwargs_from_case,
    _ns_amplitude_rescale,
    _build_mass_grid,
)
from project3_code.hydro_sim.plotting.hydro_plots import _create_7panel_vertical_figure

from piston_shock import PistonShock

USE_CACHE = False  # Set to True to use pre-saved pickle files, False to run again
Y_FIT_MIN = 0.2   # Configure the lower bound for fitting Temperature T

def get_cached_shock_solver(case, case_label):
    """Solve shock similarity ODEs once and cache the solver object (with found xsi_s)."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver_cache_path = cache_dir / f"{case_label}_similarity_solver.pkl"
    
    if USE_CACHE and solver_cache_path.exists():
        print(f"Loading cached shock similarity solver from {solver_cache_path}...")
        try:
            with open(solver_cache_path, "rb") as f:
                solver = pickle.load(f)
            # Re-bind the ODE solver which contains method callbacks
            solver.ode_solver = scipy.integrate.ode(solver.fode).set_integrator(solver.ode_scheme)
            print("Similarity solver loaded successfully.")
            return solver
        except Exception as e:
            print(f"Failed to load solver cache: {e}. Re-solving shock ODEs...")
            
    print("Solving shock similarity ODEs (finding xsi_s via root-finding)...")
    tau = float(case.tau or 0.0)
    p0 = _ns_amplitude_rescale(float(case.P0_Barye), tau)
    omega = float(getattr(case, "omega", 0.0))
    gamma = float(case.r) + 1.0
    
    solver = PistonShock(
        rho0=float(case.rho0),
        omega=omega,
        p0=p0,
        tau=tau,
        gamma=gamma
    )
    
    # Save cache by removing ode_solver temporarily to avoid pickling issues
    if USE_CACHE:
        try:
            ode_solver = solver.ode_solver
            del solver.ode_solver
            with open(solver_cache_path, "wb") as f:
                pickle.dump(solver, f, protocol=pickle.HIGHEST_PROTOCOL)
            solver.ode_solver = ode_solver
            print(f"Saved shock similarity solver to cache: {solver_cache_path}")
        except Exception as e:
            print(f"Failed to save solver cache: {e}")
        
    return solver



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


def _sim_piston_position(
    x_cells: np.ndarray,
    p_cells: np.ndarray,
    p_drive: float,
    rel_tol: float,
) -> tuple[int, float]:
    """First (leftmost) cell close enough to drive pressure; fallback to nearest."""
    denom = max(abs(float(p_drive)), 1e-30)
    rel_err = np.abs(np.asarray(p_cells, dtype=float) - float(p_drive)) / denom
    close_idx = np.flatnonzero(rel_err <= rel_tol)
    if close_idx.size > 0:
        i = int(close_idx[0])
        return i, float(x_cells[i])
    i = int(np.argmin(rel_err))
    return i, float(x_cells[i])


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

# (plot_xt_trajectories has been removed to focus exclusively on self-similar fits and relative errors)


def evaluate_shock_fits(mass_grid, time, solver, case, popt_P, popt_T, best_u):
    """Map self-similar fit parameters to dimensional (CGS) physical profiles on mass_grid at time."""
    xsi_over_m_val = solver.xsi_over_m(time=time)
    m_s = solver.xsi_s / xsi_over_m_val

    # Dimensionless shock-front boundary values
    Ps = solver.Ps
    Ts = float(solver.Ps * solver.Rs_or_Vs / solver.r)   # EOS: T = P*V/r at shock
    Us_dim = solver.Us

    # Initialise output arrays
    rho = np.zeros(len(mass_grid), dtype=float)
    p   = np.zeros(len(mass_grid), dtype=float)
    u   = np.zeros(len(mass_grid), dtype=float)
    T   = np.zeros(len(mass_grid), dtype=float)

    # Exact solver at this time to get dimensional scale factors at the shock front
    sol_exact = solver.solve(mass=mass_grid, time=time)
    p_s_cgs   = float(sol_exact["pressure"][-1])    # pressure at shock front [Barye]
    rho_s_cgs = float(sol_exact["density"][-1])     # density at shock front [g/cm^3]
    u_s_cgs   = float(sol_exact["velocity"][-1])    # velocity at shock front [cm/s]
    e_s_cgs = p_s_cgs / (rho_s_cgs * float(case.r))          # specific internal energy at shock front [erg/g]
    T_s_cgs = (e_s_cgs * rho_s_cgs**(0.14)/6730)**(1/1.6)  # temperature at shock front from EOS [K]

    for i, m in enumerate(mass_grid):
        if m >= m_s:
            # Outside shock front (unshocked region)
            rho[i] = float(case.rho0)
            p[i]   = 1e-6   # tiny ambient pressure [Barye]
            u[i]   = 0.0
            T[i]   = 300.0  # ambient temperature [K]
        else:
            # y = normalised coordinate in [0, 1]
            y = float(m) / m_s
            y = max(y, 1e-10)

            # Dimensionless fits
            P_fit_y = 1.0 - (1.0 - Ps) * (y ** popt_P[0])
            T_fit_y = Ts * (y ** popt_T[0])
            U_fit_y = best_u["func"](np.array([y]), *best_u["popt"])[0]

            # Scale to CGS via shock-front boundary values
            p[i]   = P_fit_y * (p_s_cgs / max(Ps, 1e-30))
            T[i]   = T_fit_y * (T_s_cgs / max(Ts, 1e-30))
            u[i]   = U_fit_y * (u_s_cgs / max(abs(Us_dim), 1e-30))
            # Density from ideal gas EOS: rho = p / (r * T_cgs)
            rho[i] = p[i] / (float(case.r) * T[i]) if T[i] > 0 else rho_s_cgs

    return {"density": rho, "pressure": p, "velocity": u, "temperature": T, "m_s": m_s}


def perform_shock_fitting(solver):
    y_grid = np.linspace(0.0, 1.0, 500)
    xsi_vec = y_grid * solver.xsi_s
    xsi_vec[0] = 1e-10
    
    V_val, U_val, P_val = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    rho_val = np.where(V_val > 0, 1.0 / V_val, np.nan)
    e_val = P_val * V_val / solver.r
    T_val = (e_val * (rho_val**(0.14))/6730)**(1/1.6)
    
    valid_idx = np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(T_val)
    y_valid = y_grid[valid_idx]
    T_valid = T_val[valid_idx]
    P_valid = P_val[valid_idx]
    U_valid = U_val[valid_idx]
    rho_valid = rho_val[valid_idx]
    
    U_0 = U_valid[0]
    U_s = U_valid[-1]
    
    # Pressure Fit (1-parameter power law: P = 1 - (1-Ps)*y^d)
    def power_law_P(y, d):
        return 1.0 - (1.0 - solver.Ps) * (y**d)
        
    popt_P, _ = curve_fit(power_law_P, y_valid, P_valid, p0=[1.0])
    
    # Temperature Fit Candidate Selection and Optimization
    # Ts is the true EOS temperature at the shock boundary
    Es = float(solver.Ps * solver.Rs_or_Vs / solver.r)
    Ts = T_valid[-1]
    T_0 = T_valid[0]
    
    # 5 Candidates
    def fit_T_1(y, a):
        return Ts + (T_0 - Ts) * (1.0 - y)**a
        
    def fit_T_2(y, c, a, b):
        return Ts + c * (1.0 - y)**a + (T_0 - Ts - c) * (1.0 - y)**b
        
    def fit_T_3(y, a, b):
        return Ts + (T_0 - Ts) * (1.0 - y)**a * np.exp(-b * y)
        
    def fit_T_4(y, c1, c2, a, b):
        return Ts + c1 * (1.0 - y)**a + c2 * (1.0 - y)**b
        
    def fit_T_5(y, a, b):
        return Ts + (T_0 - Ts) * (1.0 - y)**a / (1.0 + b * y)
        
    candidates_T = [
        {"id": 1, "func": fit_T_1, "name": "Single Power Law (BC)", "latex": r"T(y) \approx T_s + (T_0-T_s)(1-y)^{%.5f}", "p0": [1.0]},
        {"id": 2, "func": fit_T_2, "name": "Double Power Law (BC)", "latex": r"T(y) \approx T_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}", "p0": [(T_0-Ts)/2.0, 1.0, 2.0]},
        {"id": 3, "func": fit_T_3, "name": "Exponential-Damped Power Law (BC)", "latex": r"T(y) \approx T_s + (T_0-T_s)(1-y)^{%.5f} e^{-%.5f y}", "p0": [1.0, 0.1]},
        {"id": 4, "func": fit_T_4, "name": "Linear Combo (Shock-Conforming)", "latex": r"T(y) \approx T_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}", "p0": [(T_0-Ts)/2.0, (T_0-Ts)/2.0, 1.0, 2.0]},
        {"id": 5, "func": fit_T_5, "name": "Rational Power Law (BC)", "latex": r"T(y) \approx T_s + (T_0-T_s)\frac{(1-y)^{%.5f}}{1+%.5f y}", "p0": [1.0, 0.1]}
    ]
    
    # We will perform fitting only in the Y_FIT_MIN <= y <= 1.0 domain
    fit_mask_T = y_valid >= Y_FIT_MIN
    y_fit_T = y_valid[fit_mask_T]
    T_fit_data = T_valid[fit_mask_T]
    
    best_T = None
    min_avg_err_T = float("inf")
    fits_T = {}
    
    for cand in candidates_T:
        try:
            popt, _ = curve_fit(cand["func"], y_fit_T, T_fit_data, p0=cand["p0"], maxfev=10000)
            T_fit = cand["func"](y_valid, *popt)
            
            # Calculate errors inside the dynamic fitting domain y >= Y_FIT_MIN
            T_fit_domain = T_fit[fit_mask_T]
            T_valid_domain = T_valid[fit_mask_T]
            rel_err_T = np.abs((T_fit_domain - T_valid_domain) / T_valid_domain) * 100
            
            avg_err = np.mean(rel_err_T)
            max_err = np.max(rel_err_T)
            
            if cand["id"] == 1:
                latex_str = r"T(y) \approx T_s + (T_0-T_s)(1-y)^{%.5f}" % tuple(popt)
            elif cand["id"] == 2:
                c, a, b = popt
                latex_str = r"T(y) \approx T_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}" % (c, a, T_0 - Ts - c, b)
            elif cand["id"] == 3:
                latex_str = r"T(y) \approx T_s + (T_0-T_s)(1-y)^{%.5f} e^{-%.5f y}" % tuple(popt)
            elif cand["id"] == 4:
                c1, c2, a, b = popt
                latex_str = r"T(y) \approx T_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}" % (c1, a, c2, b)
            elif cand["id"] == 5:
                latex_str = r"T(y) \approx T_s + (T_0-T_s)\frac{(1-y)^{%.5f}}{1+%.5f y}" % tuple(popt)
                
            fits_T[cand["id"]] = (popt, T_fit, avg_err, max_err, cand["name"], latex_str)
            
            if avg_err < min_avg_err_T:
                min_avg_err_T = avg_err
                best_T = {
                    "id": cand["id"],
                    "popt": popt,
                    "func": cand["func"],
                    "name": cand["name"],
                    "latex": latex_str,
                    "avg_err": avg_err,
                    "max_err": max_err,
                    "fit_val": T_fit
                }
        except Exception as e:
            print(f"Shock temperature fit {cand['id']} failed: {e}")
            fits_T[cand["id"]] = (None, None, 0.0, 0.0, cand["name"], cand["latex"])
            
    # Velocity fits
    def fit_u_1(y, d): return U_0 - (U_0 - U_s) * (y**d)
    def fit_u_2(y, c, d): return c - (c - U_s) * (y**d)
    def fit_u_3(y, c, a, b): return c - a * (y**b)
    def fit_u_4(y, c, d): return c * (1.0 - y) / (1.0 + d * y) + U_s * y
    def fit_u_5(y, c, a, b): return c * (1.0 - y**a) * (y**(-b)) + U_s * y
    def fit_u_6(y, c, a, b): return c * (1.0 - y**a)**b + U_s * y
    
    candidates = [
        {"id": 1, "func": fit_u_1, "name": "Power Law: $U_0-(U_0-U_s)y^d$", "latex": r"U(y) \approx %.5f - %.5f y^{%.5f}", "p0": [1.0]},
        {"id": 2, "func": fit_u_2, "name": "2P Power Law: $c-(c-U_s)y^d$", "latex": r"U(y) \approx %.5f - (%.5f - U_s) y^{%.5f}", "p0": [U_0, 1.0]},
        {"id": 3, "func": fit_u_3, "name": "3P Power Law: $c-a y^b$", "latex": r"U(y) \approx %.5f - %.5f y^{%.5f}", "p0": [U_0, U_0-U_s, 1.0]},
        {"id": 4, "func": fit_u_4, "name": "Rational: $c(1-y)/(1+d y) + U_s y$", "latex": r"U(y) \approx \frac{%.5f (1 - y)}{1 + %.5f y} + U_s y", "p0": [U_0, 1.0]},
        {"id": 5, "func": fit_u_5, "name": "Singular: $c(1-y^a)y^{-b} + U_s y$", "latex": r"U(y) \approx %.5f (1 - y^{%.5f}) y^{-%.5f} + U_s y", "p0": [U_0, 1.0, 0.3]},
        {"id": 6, "func": fit_u_6, "name": "Gen Power: $c(1-y^a)^b + U_s y$", "latex": r"U(y) \approx %.5f (1 - y^{%.5f})^{%.5f} + U_s y", "p0": [U_0, 1.0, 1.0]}
    ]
    
    best_u = None
    min_avg_err = float("inf")
    fits_u = {}
    
    for cand in candidates:
        try:
            popt, _ = curve_fit(cand["func"], y_valid, U_valid, p0=cand["p0"], maxfev=10000)
            U_fit = cand["func"](y_valid, *popt)
            rel_err_u = np.abs((U_fit - U_valid) / (U_valid + 1e-15)) * 100
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
            print(f"Shock velocity fit {cand['id']} failed: {e}")
            fits_u[cand["id"]] = (None, None, 0.0, 0.0, cand["name"], cand["latex"])
            
    return y_grid, y_valid, T_valid, P_valid, U_valid, rho_valid, popt_P, best_T, fits_T, best_u, fits_u


def evaluate_shock_fits_arrays(mass_grid, time, solver, case, y_valid, P_fit, T_fit, U_fit):
    """Map self-similar fit arrays to dimensional (CGS) physical profiles on mass_grid at time."""
    xsi_over_m_val = solver.xsi_over_m(time=time)
    m_s = solver.xsi_s / xsi_over_m_val

    # Dimensionless shock-front boundary values
    Ps = solver.Ps
    Es = float(solver.Ps * solver.Rs_or_Vs / solver.r)
    # rho_s is 1/V_s = 1/Rs_or_Vs
    rho_s = 1.0 / solver.Rs_or_Vs
    Ts = (Es * rho_s**(0.14)/6730)**(1/1.6)
    Us_dim = solver.Us

    # Initialise output arrays
    rho = np.zeros(len(mass_grid), dtype=float)
    p   = np.zeros(len(mass_grid), dtype=float)
    u   = np.zeros(len(mass_grid), dtype=float)
    T   = np.zeros(len(mass_grid), dtype=float)

    # Exact solver at this time to get dimensional scale factors at the shock front
    sol_exact = solver.solve(mass=mass_grid, time=time)
    p_s_cgs   = float(sol_exact["pressure"][-1])    # pressure at shock front [Barye]
    rho_s_cgs = float(sol_exact["density"][-1])     # density at shock front [g/cm^3]
    u_s_cgs   = float(sol_exact["velocity"][-1])    # velocity at shock front [cm/s]
    e_s_cgs = p_s_cgs / (rho_s_cgs * float(case.r))          # specific internal energy at shock front [erg/g]
    T_s_cgs = (e_s_cgs * rho_s_cgs**(0.14)/6730)**(1/1.6)  # temperature at shock front from EOS [K]

    for i, m in enumerate(mass_grid):
        if m >= m_s:
            # Outside shock front (unshocked region)
            rho[i] = float(case.rho0)
            p[i]   = 1e-6   # tiny ambient pressure [Barye]
            u[i]   = 0.0
            T[i]   = 300.0  # ambient temperature [K]
        else:
            # y = normalised coordinate in [0, 1]
            y = float(m) / m_s
            y = max(y, 1e-10)

            # Interpolate from dimensionless fit arrays
            P_fit_y = np.interp(y, y_valid, P_fit)
            T_fit_y = np.interp(y, y_valid, T_fit)
            U_fit_y = np.interp(y, y_valid, U_fit)

            # Scale to CGS via shock-front boundary values
            p[i]   = P_fit_y * (p_s_cgs / max(Ps, 1e-30))
            T[i]   = T_fit_y * (T_s_cgs / max(Ts, 1e-30))
            u[i]   = U_fit_y * (u_s_cgs / max(abs(Us_dim), 1e-30))
            # Density derived from non-ideal EOS: rho = (p / (6730 * r * T^1.6))^(1/0.86)
            rho[i] = (p[i] / (6730.0 * float(case.r) * T[i]**1.6))**(1.0/0.86) if T[i] > 0 else rho_s_cgs

    return {"density": rho, "pressure": p, "velocity": u, "temperature": T}


def plot_dimensional_fit_comparison(history, solver, case, y_valid, P_fit, T_fit, U_fit, material_hydro_path, case_title):
    """Plot overlay profiles of T, rho, P, u vs m comparing Simulation, Exact Solver, and Analytic fits."""
    print(f"Generating physical shock profiles comparison for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    target_times = [1e-9, 1.5e-9, 2e-9]
    colors = ["red", "green", "blue"]
    p_scale, u_scale = 1e12, 1e5
    
    ax_T = axes[0, 0]
    ax_rho = axes[0, 1]
    ax_p = axes[1, 0]
    ax_u = axes[1, 1]
    
    for t_target, color in zip(target_times, colors):
        # 1) Simulation
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]
        
        sim_rho = history.rho[idx_sim]
        sim_p = history.p[idx_sim] / p_scale
        sim_u = history.u[idx_sim] / u_scale
        
        # Temp from ideal gas relation
        with np.errstate(divide='ignore', invalid='ignore'):
            sim_T = np.where(sim_rho > 0, history.p[idx_sim] / (sim_rho * float(case.r)), 0.0)
            
        # 2) Exact Solver
        xsi_over_m_val = solver.xsi_over_m(time=t_actual)
        m_s_exact = solver.xsi_s / xsi_over_m_val
        mass_exact = np.linspace(1e-12, m_s_exact, 200)
        sol_exact = solver.solve(mass=mass_exact, time=t_actual)
        
        exact_rho = sol_exact["density"]
        exact_p = sol_exact["pressure"] / p_scale
        exact_u = sol_exact["velocity"] / u_scale
        
        # Exact Temp
        with np.errstate(divide='ignore', invalid='ignore'):
            exact_e = np.where(exact_rho > 0, sol_exact["pressure"] / (exact_rho * float(case.r)), 0.0)
            f = 6711
            mu = 0.14
            beta = 1.6
            exact_T = (exact_e / (f * exact_rho**(-mu)))**(1/beta)
        # 3) Analytical fits mapped to CGS via array interpolation
        fits = evaluate_shock_fits_arrays(mass_exact, t_actual, solver, case, y_valid, P_fit, T_fit, U_fit)
        fit_rho = fits["density"]
        fit_p = fits["pressure"] / p_scale
        fit_u = fits["velocity"] / u_scale
        fit_T = fits["temperature"]
        
        # Plot Temperature (Kelvin / Energy Units)
        ax_T.plot(m_sim * 1e3, sim_T, '.', color=color, alpha=0.3, markersize=3, label=f'Simulation ({t_target*1e9:.1f} ns)' if color=='red' else None)
        ax_T.plot(mass_exact * 1e3, exact_T, '-', color=color, lw=2.0, label=f'Exact Solver ({t_target*1e9:.1f} ns)' if color=='red' else None)
        ax_T.plot(mass_exact * 1e3, fit_T, '--', color=color, lw=1.5, label=f'Analytic Fit ({t_target*1e9:.1f} ns)' if color=='red' else None)
        
        # Plot Density (g/cm^3)
        ax_rho.plot(m_sim * 1e3, sim_rho, '.', color=color, alpha=0.3, markersize=3)
        ax_rho.plot(mass_exact * 1e3, exact_rho, '-', color=color, lw=2.0)
        ax_rho.plot(mass_exact * 1e3, fit_rho, '--', color=color, lw=1.5)
        
        # Plot Pressure (MBar)
        ax_p.plot(m_sim * 1e3, sim_p, '.', color=color, alpha=0.3, markersize=3)
        ax_p.plot(mass_exact * 1e3, exact_p, '-', color=color, lw=2.0)
        ax_p.plot(mass_exact * 1e3, fit_p, '--', color=color, lw=1.5)
        
        # Plot Velocity (km/s)
        ax_u.plot(m_sim * 1e3, sim_u, '.', color=color, alpha=0.3, markersize=3)
        ax_u.plot(mass_exact * 1e3, exact_u, '-', color=color, lw=2.0)
        ax_u.plot(mass_exact * 1e3, fit_u, '--', color=color, lw=1.5)
        
    # Labels and Titles
    ax_T.set_ylabel(r"$T$ [Kelvin equivalent]", fontsize=12)
    ax_rho.set_ylabel(r"$\rho$ [g/cm³]", fontsize=12)
    ax_p.set_ylabel(r"$P$ [MBar]", fontsize=12)
    ax_u.set_ylabel(r"$u$ [km/s]", fontsize=12)
    
    ax_T.legend(loc='best', fontsize=9)
    
    for ax in axes.flat:
        ax.set_xlabel(r"Mass coordinate $m$ [mg/cm²]", fontsize=11)
        ax.grid(True, alpha=0.25)
        
    fig.suptitle(f"Dimensional Shock Profiles Comparison\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(material_hydro_path, dpi=200)
    plt.close(fig)
    print(f"Saved dimensional shock profiles comparison to {material_hydro_path}")


def plot_and_fit_self_similar(
    solver, y_valid, T_valid, P_valid, U_valid, rho_valid, 
    popt_P, best_T, best_u, fits_u, 
    self_similar_path, standalone_path, case_title
):
    print("Generating shock 2x2 self-similar fitting plots...")
    
    # Exact shock boundary temperature
    Ts = T_valid[-1]
    
    # Evaluate fits
    P_fit = 1.0 - (1.0 - solver.Ps) * y_valid**popt_P[0]
    T_fit = best_T["fit_val"]
    rho_fit = (P_fit / (6730.0 * solver.r * T_fit**1.6))**(1.0/0.86)
    U_fit = best_u["fit_val"]
    
    err_T = np.abs((T_fit - T_valid) / T_valid) * 100
    err_rho = np.abs((rho_fit - rho_valid) / rho_valid) * 100
    err_P = np.abs((P_fit - P_valid) / P_valid) * 100
    err_U = np.abs((U_fit - U_valid) / (U_valid + 1e-15)) * 100
    
    # Restrict Temperature and Density error computation only to the Y_FIT_MIN <= y <= 1.0 domain
    err_T_filtered = err_T[y_valid >= Y_FIT_MIN]
    avg_T, max_T = np.mean(err_T_filtered), np.max(err_T_filtered)
    
    err_rho_filtered = err_rho[y_valid >= Y_FIT_MIN]
    avg_rho, max_rho = np.mean(err_rho_filtered), np.max(err_rho_filtered)
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = np.mean(err_U), np.max(err_U)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # Panel (0,0): Temperature
    ax = axes[0, 0]
    ax.plot(y_valid, T_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, T_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    ax.set_title(f"Shock: Temperature ({best_T['name']})", fontsize=13, fontweight='bold')
    lbl_T = f"${best_T['latex']}$\nAvg Err ({Y_FIT_MIN}<y<1): {avg_T:.4f}%\nMax Err ({Y_FIT_MIN}<y<1): {max_T:.4f}%"
    ax.text(0.05, 0.05, lbl_T, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Close-up inset around Y_FIT_MIN
    ax_inset = ax.inset_axes([0.45, 0.45, 0.48, 0.48])
    ax_inset.plot(y_valid, T_valid, 'b-', lw=1.5)
    ax_inset.plot(y_valid, T_fit, 'r--', lw=1.2)
    inset_min = max(0.0, Y_FIT_MIN - 0.15)
    inset_max = min(1.0, Y_FIT_MIN + 0.15)
    ax_inset.set_xlim(inset_min, inset_max)
    idx_inset = (y_valid >= inset_min) & (y_valid <= inset_max)
    if np.any(idx_inset):
        ax_inset.set_ylim(np.min(T_valid[idx_inset]) * 0.9, np.max(T_valid[idx_inset]) * 1.1)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_title(f"Close-up at y={Y_FIT_MIN}", fontsize=8)
    ax_inset.tick_params(labelsize=8)
    
    # Panel (0,1): Density
    ax = axes[0, 1]
    ax.plot(y_valid, rho_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, rho_fit, 'r--', label='EOS Derived Fit', lw=1.5)
    ax.set_ylabel(r"Density $\rho(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Shock: Density", fontsize=13, fontweight='bold')
    lbl_R = r"$\rho(y)$" + f"\nAvg Err ({Y_FIT_MIN}<y<1): {avg_rho:.4f}%\nMax Err ({Y_FIT_MIN}<y<1): {max_rho:.4f}%"
    ax.text(0.05, 0.05, lbl_R, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,0): Pressure
    ax = axes[1, 0]
    ax.plot(y_valid, P_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, P_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Pressure $P(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Shock: Pressure", fontsize=13, fontweight='bold')
    lbl_P = f"$P(y) \\approx 1.0 - (1.0 - P_s) y^{{{popt_P[0]:.5f}}}$\nAvg Err: {avg_P:.4f}%, Max Err: {max_P:.4f}%"
    ax.text(0.05, 0.05, lbl_P, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,1): Velocity
    ax = axes[1, 1]
    ax.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, U_fit, 'r--', label='Optimized Fit', lw=1.5)
    ax.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax.set_title(f"Shock: Velocity ({best_u['name']})", fontsize=13, fontweight='bold')
    
    # Format the dynamic velocity formula in latex
    if best_u["id"] == 1:
        u_formula = f"$U(y) \\approx U_0 - (U_0 - U_s) y^{{{best_u['popt'][0]:.5f}}}$"
    elif best_u["id"] == 2:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} - ({best_u['popt'][0]:.5f} - U_s) y^{{{best_u['popt'][1]:.5f}}}$"
    elif best_u["id"] == 3:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} - {best_u['popt'][1]:.5f} y^{{{best_u['popt'][2]:.5f}}}$"
    elif best_u["id"] == 4:
        u_formula = f"$U(y) \\approx \\frac{{{best_u['popt'][0]:.5f} (1-y)}}{{1 + {best_u['popt'][1]:.5f} y}} + U_s y$"
    elif best_u["id"] == 5:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1 - y^{{{best_u['popt'][1]:.5f}}}) y^{{-{best_u['popt'][2]:.5f}}} + U_s y$"
    elif best_u["id"] == 6:
        u_formula = f"$U(y) \\approx {best_u['popt'][0]:.5f} (1 - y^{{{best_u['popt'][1]:.5f}}})^{{{best_u['popt'][2]:.5f}}} + U_s y$"
        
    lbl_U = u_formula + f"\nAvg Err: {avg_U:.4f}%, Max Err: {max_U:.4f}%"
    ax.text(0.05, 0.05, lbl_U, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    for ax in axes.flat:
        ax.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=9.5)
        
    fig.suptitle(f"Shock self-similar Profiles & Analytical Fits\n{case_title}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(self_similar_path, dpi=200)
    plt.close(fig)
    print(f"Saved self-similar shock profiles to {self_similar_path}")
    
    # Standalone velocity comparisons plot
    fig_sa, (ax_sa1, ax_sa2) = plt.subplots(1, 2, figsize=(18, 8.5))
    ax_sa1.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=3.0)
    colors_u = {1: 'crimson', 2: 'darkorange', 3: 'forestgreen', 4: 'darkviolet', 5: 'deeppink', 6: 'teal'}
    
    for i in range(1, 7):
        popt, U_fit_cand, avg_err, max_err, name, latex = fits_u[i]
        if popt is not None:
            lbl = f"Fit {i}: {name}\nAvg Err: {avg_err:.3f}%, Max Err: {max_err:.3f}%"
            lw = 2.2 if i == best_u["id"] else 1.5
            ax_sa1.plot(y_valid, U_fit_cand, colors_u[i], linestyle='--', label=lbl, lw=lw)
            
            err_curve = np.abs((U_fit_cand - U_valid) / (U_valid + 1e-15)) * 100
            ax_sa2.plot(y_valid, err_curve, colors_u[i], label=f"Fit {i} (Avg: {avg_err:.3f}%)", lw=lw)
            
    ax_sa1.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_sa1.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax_sa1.legend(loc='best', fontsize=9.0)
    ax_sa1.grid(True, alpha=0.3)
    ax_sa1.set_title("Dimensionless Velocity $U(y)$ vs 6 Candidates", fontsize=13, fontweight='bold')
    
    ax_sa2.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_sa2.set_ylabel(r"Relative Error [$\%$]", fontsize=12)
    ax_sa2.set_yscale('log')
    ax_sa2.legend(loc='best', fontsize=9.5)
    ax_sa2.grid(True, which="both", ls=":", alpha=0.5)
    ax_sa2.set_title("Relative Errors of Velocity Fits (semi-log)", fontsize=13, fontweight='bold')
    
    fig_sa.suptitle(f"Shock Velocity Profile Curve Fitting & Optimization\nChosen Formal Fit: Fit {best_u['id']} ({best_u['name']})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_sa.savefig(standalone_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sa)
    print(f"Saved standalone shock velocity fits to {standalone_path}")


def plot_standalone_temperature_fits(
    y_valid, T_valid, fits_T, best_T, standalone_path, case_title
):
    print("Generating standalone temperature fitting comparison plots...")
    fig_sa, (ax_sa1, ax_sa2) = plt.subplots(2, 1, figsize=(11, 10))
    
    ax_sa1.plot(y_valid, T_valid, 'b-', label='Numerical Solver', lw=3.0)
    
    colors_T = {1: 'crimson', 2: 'darkorange', 3: 'forestgreen', 4: 'darkviolet', 5: 'deeppink'}
    
    # We only plot relative errors for the fitting domain y >= Y_FIT_MIN
    fit_mask_T = y_valid >= Y_FIT_MIN
    y_valid_T = y_valid[fit_mask_T]
    T_valid_domain = T_valid[fit_mask_T]
    
    for i in range(1, 6):
        popt, T_fit_cand, avg_err, max_err, name, latex = fits_T[i]
        if popt is not None:
            lbl = f"Fit {i}: {name}\nAvg Err ({Y_FIT_MIN}<y<1): {avg_err:.3f}%, Max: {max_err:.3f}%"
            lw = 2.2 if i == best_T["id"] else 1.5
            ax_sa1.plot(y_valid, T_fit_cand, colors_T[i], linestyle='--', label=lbl, lw=lw)
            
            # Error only plotted inside fitting domain
            T_fit_cand_domain = T_fit_cand[fit_mask_T]
            err_curve = np.abs((T_fit_cand_domain - T_valid_domain) / T_valid_domain) * 100
            ax_sa2.plot(y_valid_T, err_curve, colors_T[i], label=f"Fit {i} (Avg: {avg_err:.3f}%)", lw=lw)
            
    ax_sa1.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_sa1.set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    ax_sa1.legend(loc='best', fontsize=9.0)
    ax_sa1.grid(True, alpha=0.3)
    ax_sa1.set_title("Dimensionless Temperature $T(y)$ vs 5 Candidates", fontsize=13, fontweight='bold')
    
    ax_sa2.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_sa2.set_ylabel(r"Relative Error [$\%$]", fontsize=12)
    ax_sa2.set_yscale('log')
    ax_sa2.legend(loc='best', fontsize=9.5)
    ax_sa2.grid(True, which="both", ls=":", alpha=0.5)
    ax_sa2.set_title(f"Relative Errors of Temperature Fits on fitting domain y >= {Y_FIT_MIN} (semi-log)", fontsize=13, fontweight='bold')
    
    fig_sa.suptitle(f"Shock Temperature Curve Fitting & Optimization\nChosen Formal Fit: Fit {best_T['id']} ({best_T['name']})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_sa.savefig(standalone_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sa)
    print(f"Saved standalone shock temperature fits to {standalone_path}")


def plot_relative_errors(
    solver, y_valid, T_valid, P_valid, U_valid, R_valid, 
    popt_P, best_T, best_u, relative_errors_path, case_title
):
    print("Generating shock relative error plots...")
    P_fit = 1.0 - (1.0 - solver.Ps) * y_valid**popt_P[0]
    T_fit = best_T["fit_val"]
    R_fit = (P_fit / (6730.0 * solver.r * T_fit**1.6))**(1.0/0.86)
    U_fit = best_u["fit_val"]
    
    err_T = np.abs((T_fit - T_valid) / T_valid) * 100
    error_rho = np.abs((R_fit - R_valid) / R_valid) * 100
    err_P = np.abs((P_fit - P_valid) / P_valid) * 100
    err_U = np.abs((U_fit - U_valid) / (U_valid + 1e-15)) * 100
    
    # Filter Temperature and Density errors to Y_FIT_MIN <= y <= 1.0
    fit_mask_T = y_valid >= Y_FIT_MIN
    y_valid_T = y_valid[fit_mask_T]
    err_T_filtered = err_T[fit_mask_T]
    err_rho_filtered = error_rho[fit_mask_T]
    
    avg_T, max_T = np.mean(err_T_filtered), np.max(err_T_filtered)
    avg_R, max_R = np.mean(err_rho_filtered), np.max(err_rho_filtered)
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = best_u["avg_err"], best_u["max_err"]
    
    fig_err, ax_err = plt.subplots(figsize=(10, 7.5))
    ax_err.plot(y_valid_T, err_T_filtered, label=f'Temperature $T(y)$ (Avg ({Y_FIT_MIN}<y<1): {avg_T:.4f}%, Max ({Y_FIT_MIN}<y<1): {max_T:.4f}%)', lw=2.0)
    ax_err.plot(y_valid_T, err_rho_filtered, label=f'Density $\\rho(y)$ (Avg ({Y_FIT_MIN}<y<1): {avg_R:.4f}%, Max ({Y_FIT_MIN}<y<1): {max_R:.4f}%)', lw=2.0)
    ax_err.plot(y_valid, err_P, label=f'Pressure $P(y)$ (Avg: {avg_P:.4f}%, Max: {max_P:.4f}%)', lw=2.0)
    ax_err.plot(y_valid, err_U, label=f'Velocity $U(y)$ (Avg: {avg_U:.4f}%, Max: {max_U:.4f}%)', lw=2.0)
    
    ax_err.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_err.set_ylabel(r"Relative Error [$\%$]", fontsize=12)
    ax_err.set_yscale('log')
    ax_err.grid(True, which="both", ls=":", alpha=0.5)
    ax_err.legend(loc="best", fontsize=10.5)
    ax_err.set_title(f"Relative Errors of Shock self-similar Fits (semi-log)\n{case_title}", fontsize=13, fontweight='bold')
    
    fig_err.tight_layout()
    fig_err.savefig(relative_errors_path, dpi=200)
    plt.close(fig_err)
    print(f"Saved shock relative errors to {relative_errors_path}")


# =============================================================================
# Custom 5-way animated GIF Evolution saver
# =============================================================================

def run_simulation_and_references(preset_name: str, case_label: str):
    """Run full simulation and build PistonShock reference solver, or load from cache."""
    from dataclasses import replace as dc_replace
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_label}_cache.pkl"

    case, config = get_preset(preset_name)

    if USE_CACHE and cache_path.exists():
        print(f"Loading cached simulation and reference data from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            solver = data["solver"]
            # Re-bind the ODE solver which contains method callbacks
            if not hasattr(solver, "ode_solver") or solver.ode_solver is None:
                solver.ode_solver = scipy.integrate.ode(solver.fode).set_integrator(solver.ode_scheme)
            return data["case"], data["history"], solver
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running simulation...")

    # Run simulation
    print("Running simulation...")
    _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)

    # Get PistonShock solver
    print("Getting cached PistonShock solver...")
    solver = get_cached_shock_solver(case, case_label)

    cache_data = {"case": case, "history": history, "solver": solver}
    if USE_CACHE:
        print(f"Saving simulation and reference data cache to {cache_path}...")
        try:
            # PistonShock solver contains an un-picklable ode_solver, detach temporarily
            ode_solver = getattr(solver, "ode_solver", None)
            if hasattr(solver, "ode_solver"):
                del solver.ode_solver
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            if ode_solver is not None:
                solver.ode_solver = ode_solver
        except Exception as e:
            print(f"Failed to save cache: {e}")

    return case, history, solver


def generate_verification_plots(
    history,
    case,
    solver,
    case_label: str,
    case_title: str,
):
    """Generate analytical self-similar fits, dimensional comparison, and relative error plots."""
    out_dir = Path("results/ictt")
    ss_dir = out_dir / "self_similar"
    dv_dir = out_dir / "dimensional_verification"
    ss_dir.mkdir(parents=True, exist_ok=True)
    dv_dir.mkdir(parents=True, exist_ok=True)

    self_similar_path = str(ss_dir / f"{case_label}_self_similar.png")
    standalone_path = str(ss_dir / f"{case_label}_velocity_fits_standalone.png")
    standalone_T_path = str(ss_dir / f"{case_label}_temperature_fits_standalone.png")
    relative_errors_path = str(ss_dir / f"{case_label}_relative_errors.png")
    dimensional_fit_path = str(dv_dir / f"{case_label}_dimensional_fit_comparison.png")

    y_grid, y_valid, T_valid, P_valid, U_valid, R_valid, popt_P, best_T, fits_T, best_u, fits_u = perform_shock_fitting(solver)

    # Compute fit arrays
    P_fit = 1.0 - (1.0 - solver.Ps) * y_valid**popt_P[0]
    T_fit = best_T["fit_val"]
    U_fit = best_u["fit_val"]

    # 1) Standalone Temperature fits comparison and relative errors
    plot_standalone_temperature_fits(y_valid, T_valid, fits_T, best_T, standalone_T_path, case_title)

    # 2) Self-similar profiles and fits (using optimal models)
    plot_and_fit_self_similar(solver, y_valid, T_valid, P_valid, U_valid, R_valid, popt_P, best_T, best_u, fits_u, self_similar_path, standalone_path, case_title)

    # 3) Dimensional fit comparison
    plot_dimensional_fit_comparison(history, solver, case, y_valid, P_fit, T_fit, U_fit, dimensional_fit_path, case_title)

    # 4) Relative errors of self-similar fits
    plot_relative_errors(solver, y_valid, T_valid, P_valid, U_valid, R_valid, popt_P, best_T, best_u, relative_errors_path, case_title)


def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run full verification comparison pipeline for a given preset."""
    print("=" * 80)
    print(f"PROCESSING PRESET: {preset_name} -> {case_label}")
    print("=" * 80)

    case, history, solver = run_simulation_and_references(preset_name, case_label)

    generate_verification_plots(
        history=history,
        case=case,
        solver=solver,
        case_label=case_label,
        case_title=case_title,
    )
    print(f"Preset {preset_name} processed successfully.")


def main():
    # Process only the power-law drive piston shock matching the boundary condition (tau=-43/96)
    # run_preset_workflow(
    #     PRESET_CONSTANT_PRESSURE,
    #     "constant_pressure_drive",
    #     "Constant Pressure Drive (tau=0)"
    # )
    run_preset_workflow(
        PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE,
        "power_law_pressure_drive",
        "Power-Law Pressure Drive (tau=-43/96)"
    )
    print("\nAll custom simulations, PistonShock comparisons, plotting, fitting, and exports completed successfully!")


if __name__ == "__main__":
    main()

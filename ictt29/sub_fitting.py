# ictt29/sub_fitting.py
"""
Subsonic self-similar profile fitting and dimensional comparison script.

Produces:
1. 2x2 self-similar profile fits (T, rho_EOS, P, U) with LaTeX formula + error labels
2. Semi-log relative error plot for all 4 profiles
3. Velocity candidate comparison (9 fits side-by-side)
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
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro

from project3_code.menahem_new.subsonic_heat_wave_og import SubsonicHeatWave

USE_CACHE = True  # Set to True to use pre-saved pickle files, False to run again


def get_cached_sub_solver(case, case_label):
    """Solve subsonic similarity ODEs once and cache the solver object (with found xsi_f)."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver_cache_path = cache_dir / f"{case_label}_similarity_solver.pkl"

    if USE_CACHE and solver_cache_path.exists():
        print(f"Loading cached subsonic similarity solver from {solver_cache_path}...")
        try:
            with open(solver_cache_path, "rb") as f:
                solver = pickle.load(f)
            # Re-bind the ODE solver which contains method callbacks
            if solver is not None:
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
        ode_solver = getattr(solver, "ode_solver", None)
        if hasattr(solver, "ode_solver"):
            del solver.ode_solver
        with open(solver_cache_path, "wb") as f:
            pickle.dump(solver, f, protocol=pickle.HIGHEST_PROTOCOL)
        if ode_solver is not None:
            solver.ode_solver = ode_solver
        print(f"Saved subsonic similarity solver to cache: {solver_cache_path}")
    except Exception as e:
        print(f"Failed to save solver cache: {e}")

    return solver


def run_simulation_and_references(preset_name: str, case_label: str):
    """Run full simulation and build SubsonicHeatWave reference solver, or load from cache."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_label}_cache.pkl"

    case, config = get_preset(preset_name)

    if USE_CACHE and cache_path.exists():
        print(f"Loading cached simulation and reference data from {cache_path}...")
        try:
            import sys
            import numpy.core
            sys.modules['numpy._core'] = sys.modules.get('numpy.core')
            sys.modules['numpy._core.numeric'] = sys.modules.get('numpy.core.numeric')
            sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray')
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            solver = data["solver"]
            # Re-bind the ODE solver which contains method callbacks
            if solver is not None:
                solver.ode_solver = scipy.integrate.ode(solver.fode).set_integrator(solver.ode_scheme)
            print("Loaded successfully from cache.")
            return data["case"], data["history"], solver
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running simulation...")

    # Run simulation
    print("Running simulation...")
    _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)

    # Get subsonic solver
    solver = get_cached_sub_solver(case, case_label)

    cache_data = {"case": case, "history": history, "solver": solver}
    print(f"Saving simulation and reference data cache to {cache_path}...")
    try:
        # SubsonicHeatWave solver contains an un-picklable ode_solver, detach temporarily
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

def dimensionless_density_from_eos(T, P, beta=1.6, mu=0.14):
    return (T**beta/P)**(1/(mu-1))

def dimensional_temperature_from_eos(P, V, beta=1.6, mu=0.14, r=0.25, f=6730.91):
    return (P * V**(1-mu) / (r*f))**(1./beta)


def trim_noisy_tail_with_coordinate(x, values, rel_err_threshold=0.5, eps=1e-15):
    """Trim profile tail based on point-to-point relative error and keep coordinate aligned."""
    x = np.asarray(x)
    values = np.asarray(values)
    if x.shape != values.shape:
        raise ValueError("x and values must have the same shape")

    cut_idx = len(values)
    for i in range(len(values) - 1, 0, -1):
        rel_err = np.abs((values[i] - values[i - 1]) / (values[i - 1] + eps))
        if rel_err < rel_err_threshold:
            cut_idx = i + 1
            break

    return x[:cut_idx], values[:cut_idx]

def perform_subsonic_fitting(solver):
    y_grid = np.linspace(0.0, 1.0 - 1e-10, 2000)
    xsi_vec = solver.xsi_f * y_grid
    profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    P_val = profiles["P"]
    T_val = profiles["T"]
    U_val = profiles["U"]
    V_val = profiles["V"]
        
    # Calculate density using the exact uniform EOS formula requested by user
    rho_val = dimensionless_density_from_eos(T_val, P_val)
    y_rho, rho_val = trim_noisy_tail_with_coordinate(y_grid, rho_val)

    print(rho_val[-10:])
    valid_idx = np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(T_val)
    y_valid = y_grid[valid_idx]
    T_valid = T_val[valid_idx]
    P_valid = P_val[valid_idx]
    U_valid = U_val[valid_idx]
    rho_valid_idx = np.isfinite(rho_val)
    y_rho_valid = y_rho[rho_valid_idx]
    rho_valid = rho_val[rho_valid_idx]
    
    # Fits
    def smith_approximation(y, R):
        return ((1.0 - y) * (1.0 + R * y))**(10.0 / 39.0)
        
    # Pressure Fit with a constant P_valid[0] offset
    P_0 = P_valid[0] 
    solver.P0 = P_0 # important - the 
    def power_law_origin(y, a, b, c, d):
        return P_0 + a * y**c + b * y**(c+d)
        
    popt_T, _ = curve_fit(smith_approximation, y_valid, T_valid, p0=[0.5])
    popt_P, _ = curve_fit(power_law_origin, y_valid, P_valid, p0=[0.355, 0.5, 0.04, 2.3])
    u_0 = U_valid[0]
    u_f = U_valid[-1]

    # Velocity fits
    def fit_u_1(y, c, d): return c * (1.0 - y)**d
    def fit_u_2(y, c, b): return c * (1.0 - y) / (1.0 + b * y)
    def fit_u_3(y, a, b, epsilon=1e-5):
        y = np.asarray(y)
        # Ensures smooth singularity at 0, while perfectly striking u_f at 1
        return u_f + (u_0 - u_f) * ((1.0 - y) ** a) * ((y + epsilon) ** (-b))
    def fit_u_4(y, c, a, b): 
        y = np.asarray(y)
        return c * (1.0 - y**a) * (y**(-b))
    def fit_u_5(y, c, a, b, epsilon=1e-5): 
        y = np.asarray(y)
        # The clamp (1-y)**a forces the right boundary to match u_f exactly
        # The ln(y) provides the steep vertical plunge at the origin
        return u_f + (u_0 - u_f) * ((1.0 - y) ** a) * (1.0 - c * np.log(y + epsilon))
    def fit_u_6(y, c, a, b): 
        y = np.asarray(y)
        # Clip to avoid negative roots if solver searches weird parameter spaces
        y_clipped = np.clip(y, 1e-10, 1.0)
        numerator = 1.0 - y_clipped**a
        denominator = 1.0 + c * (y_clipped**b)
        return u_f + (u_0 - u_f) * (numerator / denominator)
    
    
    # Custom front velocity Candidates requested by user
    def fit_u_7(y, a, b):
        return u_f + (u_0 - u_f) * (1.0 - y**a)**b
        
    def fit_u_8(y, c):
        return u_f + (u_0 - u_f) * (1.0 - y) / (1.0 + c * y)
        
    def fit_u_9(y, a, b, d):
        y = np.asarray(y)
        res = np.zeros_like(y)
        c = (u_0 - u_f + a * 0.2**b) / (0.8**d)
        left_mask = y <= 0.2
        res[left_mask] = u_0 + a * y[left_mask]**b
        res[~left_mask] = u_f + c * (1.0 - y[~left_mask])**d
        return res
        
    def fit_u_10(y, a1, a2, alpha, b1, b2, y0):
        y = np.asarray(y, dtype=float)
        y_clipped = np.clip(y, 1e-12, 1.0)
        u_left = u_0 + a1 * (y_clipped ** alpha) + a2 * (y_clipped ** (2.0 * alpha))
        dx = 1.0 - y_clipped
        u_right = u_f + b1 * dx + b2 * (dx ** 2)
        weight = (1.0 - y_clipped) / (1.0 + (y_clipped / y0) ** 4)
        return weight * u_left + (1.0 - weight) * u_right
        
    def fit_u_11(y, a1, a2, a3, alpha, b1, b2, b3, y0):
        y = np.asarray(y, dtype=float)
        y_clipped = np.clip(y, 1e-12, 1.0)
        u_left = u_0 + a1 * (y_clipped ** alpha) + a2 * (y_clipped ** (2.0 * alpha)) + a3 * (y_clipped ** (3.0 * alpha))
        dx = 1.0 - y_clipped
        u_right = u_f + b1 * dx + b2 * (dx ** 2) + b3 * (dx ** 3)
        weight = (1.0 - y_clipped) / (1.0 + (y_clipped / y0) ** 4)
        return weight * u_left + (1.0 - weight) * u_right
        
    def fit_u_12(y, a1, a2, a3, a4, alpha, b1, b2, b3, b4, y0):
        y = np.asarray(y, dtype=float)
        y_clipped = np.clip(y, 1e-12, 1.0)
        u_left = u_0 + a1 * (y_clipped ** alpha) + a2 * (y_clipped ** (2.0 * alpha)) + a3 * (y_clipped ** (3.0 * alpha)) + a4 * (y_clipped ** (4.0 * alpha))
        dx = 1.0 - y_clipped
        u_right = u_f + b1 * dx + b2 * (dx ** 2) + b3 * (dx ** 3) + b4 * (dx ** 4)
        weight = (1.0 - y_clipped) / (1.0 + (y_clipped / y0) ** 4)
        return weight * u_left + (1.0 - weight) * u_right
    
    candidates = [
        {"id": 1, "func": fit_u_1, "name": "Power Law: $c(1-y)^d$", "latex": r"U(y) \approx %.5f (1 - y)^{%.5f}", "p0": [u_0, 0.5], "bounds": (-np.inf, np.inf)},
        {"id": 2, "func": fit_u_2, "name": "Rational: $c(1-y)/(1+b y)$", "latex": r"U(y) \approx \frac{%.5f (1 - y)}{1 + %.5f y}", "p0": [u_0, 0.5], "bounds": (-np.inf, np.inf)},
        {"id": 3, "func": fit_u_3, "name": r"Singular Front 3P: $u_f + (u_0-u_f)(1-y)^a y^{-b}$", "latex": r"U(y) \approx u_f + (u_0 - u_f)(1-y)^{%.5f}y^{-%.5f}", "p0": [1.0, 0.3], "bounds": (0.0, np.inf)},
        {"id": 4, "func": fit_u_4, "name": "Singular 3P: $c(1-y^a)y^{-b}$", "latex": r"U(y) \approx %.5f (1 - y^{%.5f}) y^{-%.5f}", "p0": [-1.0, 1.0, 0.3], "bounds": (-np.inf, np.inf)},
        {"id": 5, "func": fit_u_5, "name": r"Log Front 3P: $u_f + (u_0-u_f)(1-y)^a(1-c\ln(y))$", "latex": r"U(y) \approx u_f + (u_0 - u_f)(1-y)^{%.5f}(1 - %.5f \ln(y))", "p0": [-1.0, 1.0, 0.3], "bounds": (-np.inf, np.inf)},
        {"id": 6, "func": fit_u_6, "name": r"Rational Frac Front 3P: $u_f + (u_0-u_f)(1-y^a)/(1+cy^b)$", "latex": r"U(y) \approx u_f + (u_0 - u_f) \frac{1 - y^{%.5f}}{1 + %.5f y^{%.5f}}", "p0": [u_0, 1.0, 1.0], "bounds": (-np.inf, np.inf)},
        {"id": 7, "func": fit_u_7, "name": "Power Law Front: $u_f + (u_0-u_f)(1-y^a)^b$", "latex": r"U(y) \approx u_f + (u_0 - u_f)(1 - y^{%.5f})^{%.5f}", "p0": [1.0, 1.0], "bounds": (0.0, np.inf)},
        {"id": 8, "func": fit_u_8, "name": "Rational Front: $u_f + (u_0-u_f)(1-y)/(1+cy)$", "latex": r"U(y) \approx u_f + \frac{(u_0 - u_f)(1 - y)}{1 + %.5f y}", "p0": [1.0], "bounds": (-0.99, np.inf)},
        {"id": 9, "func": fit_u_9, "name": "Piecewise Power-Law (y=0.2)", "latex": r"U(y) \approx Piecewise", "p0": [1.0, 0.5, 0.5], "bounds": (0.0, np.inf)},
        {"id": 10, "func": fit_u_10, "name": "Asymptotic Blend 6P", "latex": r"U(y) \approx W U_{left} + (1-W) U_{right}", "p0": [-5.0, -2.0, 0.25, 2.0, 0.5, 0.2], "bounds": ([-50.0, -50.0, 0.01, -20.0, -20.0, 0.05], [50.0, 50.0, 0.99, 20.0, 20.0, 0.50])},
        {"id": 11, "func": fit_u_11, "name": "Asymptotic Blend 8P (3 Pow)", "latex": r"U(y) \approx W U_{left,3} + (1-W) U_{right,3}", "p0": [-5.0, -2.0, -0.5, 0.25, 2.0, 0.5, 0.1, 0.2], "bounds": ([-50.0, -50.0, -50.0, 0.01, -20.0, -20.0, -20.0, 0.05], [50.0, 50.0, 50.0, 0.99, 20.0, 20.0, 20.0, 0.50])},
        {"id": 12, "func": fit_u_12, "name": "Asymptotic Blend 10P (4 Pow)", "latex": r"U(y) \approx W U_{left,4} + (1-W) U_{right,4}", "p0": [-5.0, -2.0, -0.5, -0.1, 0.25, 2.0, 0.5, 0.1, 0.05, 0.2], "bounds": ([-50.0, -50.0, -50.0, -50.0, 0.01, -20.0, -20.0, -20.0, -20.0, 0.05], [50.0, 50.0, 50.0, 50.0, 0.99, 20.0, 20.0, 20.0, 20.0, 0.50])}
    ]
    
    y_valid_u = y_valid
    U_valid_u = U_valid
    
    best_u = None
    min_avg_err = float("inf")
    fits_u = {}
    
    for cand in candidates:
        try:
            popt, _ = curve_fit(cand["func"], y_valid_u, U_valid_u, p0=cand["p0"], bounds=cand["bounds"], maxfev=10000)
            U_fit = cand["func"](y_valid, *popt)
            U_fit_u = cand["func"](y_valid_u, *popt)
            rel_err_u = np.abs((U_fit_u - U_valid_u) / (U_valid_u + 1e-15))  # Absolute error
            avg_err = np.mean(rel_err_u)
            max_err = np.max(rel_err_u)
            print(f"  Candidate {cand['id']} ({cand['name']}): Avg Err: {avg_err:.3e}, Max Err: {max_err:.3e}")
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
            
    print(f"Optimal Velocity Candidate: Fit {best_u['id']} ({best_u['name']}) - Avg Err: {best_u['avg_err']:.3e}, Max Err: {best_u['max_err']:.3e}")
    
    params = {
        "y_valid": y_valid,
        "y_rho": y_rho_valid,
        "T_valid": T_valid,
        "P_valid": P_valid,
        "U_valid": U_valid,
        "rho_valid": rho_valid,
        "popt_T": popt_T,
        "popt_P": popt_P,
        "best_u": best_u,
        "fits_u": fits_u,
        "solver": solver
    }
    return params


def fit_by_params(y: np.ndarray, params: dict):
    """Compute the self-similar fit profiles (T, P, U, rho) on the similarity coordinate xsi_vec."""
    solver = params["solver"]
    popt_T = params["popt_T"]
    popt_P = params["popt_P"]
    best_u = params["best_u"]
    
    # Temperature fit
    T_fit = ((1.0 - y) * (1.0 + popt_T[0] * y)) ** (10.0 / 39.0)
    
    # Pressure fit
    P_0 = solver.P0
    P_fit = P_0 + popt_P[0] * y**popt_P[2] + popt_P[1] * y**(popt_P[2] + popt_P[3])
    
    # Velocity fit (evaluated on y)
    U_fit = best_u["func"](y, *best_u["popt"])
    
    # Density fit (represents specific volume in the self-similar EOS context)
    rho_fit = dimensionless_density_from_eos(T_fit, P_fit)
    
    return T_fit, P_fit, U_fit, rho_fit


def calculate_dimensional_fits(mass_grid, t_actual, solver, params):
    """Map subsonic self-similar fit arrays to dimensional (CGS) physical profiles on mass_grid at time t_actual."""
    xsi_vec = mass_grid * solver.xsi_over_m(time=t_actual)
    y = xsi_vec / solver.xsi_f
    T_fit, P_fit, U_fit, rho_fit = fit_by_params(y, params)
    
    rho_fit_dimensional = rho_fit * (solver.A**(-solver.a1)) * (solver.B**(-solver.b1)) * (t_actual ** (-solver.c1))    
    p_fit_dimensional = P_fit * (solver.A**solver.a3) * (solver.B**solver.b3) * (t_actual ** solver.c3)
    u_fit_dimensional = U_fit * (solver.A**solver.a2) * (solver.B**solver.b2) * (t_actual ** solver.c2)
    T_fit_dimensional = dimensional_temperature_from_eos(
        p_fit_dimensional, 
        1.0/rho_fit_dimensional, 
        beta=solver.beta, 
        mu=solver.mu, 
        r=solver.r, 
        f=solver.f
    )
    return {
        "density": rho_fit_dimensional, 
        "pressure": p_fit_dimensional, 
        "velocity": u_fit_dimensional, 
        "temperature": T_fit_dimensional
    }


def plot_dimensional_fit_comparison(history, solver, case, params, dimensional_fit_path, case_title):
    """Plot overlay profiles of T, rho, P, u vs m comparing Simulation, Exact Solver, and Analytic fits."""
    print(f"Generating physical subsonic profiles comparison for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    target_times = [1.0e-9, 1.5e-9, 2.0e-9]
    plasma = plt.get_cmap("plasma")
    sim_colors = [plasma(v) for v in np.linspace(0, 1, len(target_times))]
    
    ax_rho = axes[0, 0]
    ax_p = axes[0, 1]
    ax_u = axes[1, 0]
    ax_T = axes[1, 1]
    
    for i, (t_target, sim_color) in enumerate(zip(target_times, sim_colors)):
        # 1) Simulation
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]
        
        # Subsonic ablated mass front from solver at this actual time
        m_f_val = solver.ablated_mass(time=t_actual)
        
        sub_mask = m_sim <= m_f_val
        m_sim_sub = m_sim[sub_mask]
        
        sim_rho = history.rho[idx_sim][sub_mask]
        sim_p = history.p[idx_sim][sub_mask]
        sim_u = history.u[idx_sim][sub_mask] 
        sim_T = history.T[idx_sim][sub_mask] 
        
        # 2) Exact Solver Lagrangian Grid
        mass_solver = np.linspace(1e-12, m_f_val, 300)
        sol_hs = solver.solve(mass=mass_solver, time=t_actual)
        mass_exact_rho, exact_rho = trim_noisy_tail_with_coordinate(mass_solver, sol_hs["density"])

        exact_p = sol_hs["pressure"]
        exact_u = sol_hs["velocity"]
        exact_T = sol_hs["temperature"]
        
        # 3) Analytical fits mapped from dimensionless to CGS
        fits = calculate_dimensional_fits(mass_solver, t_actual, solver, params)
        
        mass_fit_rho, fit_rho = trim_noisy_tail_with_coordinate(mass_solver, fits["density"])
        fit_p = fits["pressure"]
        fit_u = fits["velocity"]
        fit_T = fits["temperature"]
        
        show_label = i == 0
        
        # Plot Density
        ax_rho.plot(m_sim_sub * 1e3, sim_rho, '-', color=sim_color, markersize=3, alpha=0.7, label=f"Simulation ({t_target*1e9:.1f} ns)" if show_label else None)
        ax_rho.plot(mass_exact_rho * 1e3, exact_rho, '--', color='black', lw=2.0, label="Exact Solver" if show_label else None)
        ax_rho.plot(mass_fit_rho * 1e3, fit_rho, '.', color='green', lw=0.5, alpha=0.3, label="Analytic Fit" if show_label else None)
        
        # # Zoomed inset density plot near front (y in [0.8, 0.99])
        # y_sim = m_sim_sub / m_f_val
        # y_sol_exact = mass_exact_rho / m_f_val
        # y_sol_fit = mass_fit_rho / m_f_val
        # zoom_mask = (y_sim >= 0.8) & (y_sim <= 0.99)
        # sol_zoom_mask_exact = (y_sol_exact >= 0.8) & (y_sol_exact <= 0.99)
        # sol_zoom_mask_fit = (y_sol_fit >= 0.8) & (y_sol_fit <= 0.99)
        
        # axins.plot(m_sim_sub[zoom_mask] * 1e3, sim_rho[zoom_mask], '-', color=sim_color, markersize=2, alpha=0.7)
        # axins.plot(mass_exact_rho[sol_zoom_mask_exact] * 1e3, exact_rho[sol_zoom_mask_exact], '--', color='black', lw=1.5)
        # axins.plot(mass_fit_rho[sol_zoom_mask_fit] * 1e3, fit_rho[sol_zoom_mask_fit], '--', color='green', lw=1.2)
        
        # Plot Pressure
        ax_p.plot(m_sim_sub * 1e3, sim_p, '-', color=sim_color, markersize=3, alpha=0.7)
        ax_p.plot(mass_solver * 1e3, exact_p, '--', color='black', lw=2.0)
        ax_p.plot(mass_solver * 1e3, fit_p, '.', color='green', lw=0.5, alpha=0.3)
        
        # Plot Velocity
        ax_u.plot(m_sim_sub * 1e3, sim_u, '-', color=sim_color, markersize=3, alpha=0.7)
        ax_u.plot(mass_solver * 1e3, exact_u, '--', color='black', lw=2.0)
        ax_u.plot(mass_solver * 1e3, fit_u, '.', color='green', lw=0.5, alpha=0.3)
        
        # Plot Temperature
        ax_T.plot(m_sim_sub * 1e3, sim_T, '-', color=sim_color, markersize=3, alpha=0.7)
        ax_T.plot(mass_solver * 1e3, exact_T, '--', color='black', lw=2.0)
        ax_T.plot(mass_solver * 1e3, fit_T, '.', color='green', lw=0.5, alpha=0.3)

    # Build time legend entries using plasma colors
    time_handles = [
        Line2D([0], [0], color=sim_colors[k], lw=2, label=f"{target_times[k]*1e9:.1f} ns")
        for k in range(len(target_times))
    ]

    # Styling
    labels = ["Density [g/cm$^3$]", "Pressure [MBar]", "Velocity [km/s]", "Temperature [HeV]"]
    for j, ax in enumerate([ax_rho, ax_p, ax_u, ax_T]):
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels[j], fontsize=12)
        ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm$^2$]", fontsize=12)
        if j == 0:
            style_handles = [
                Line2D([0], [0], color='black', lw=2, linestyle='--', label='Exact Solver'),
                Line2D([0], [0], color='green', lw=2, linestyle='--', label='Analytic Fit'),
            ]
            ax.legend(handles=time_handles + style_handles, loc="upper left")
            
    # Style inset
    # axins.grid(True, alpha=0.3)
    # axins.set_title("Zoom near front", fontsize=9)
    # ax_rho.indicate_inset_zoom(axins, edgecolor="black")
            
    plt.suptitle(f"Subsonic Ablation Region Verification\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(dimensional_fit_path, dpi=200)
    print(f"Saved dimensional subsonic profiles comparison to {dimensional_fit_path}")
    plt.close(fig)


def plot_and_fit_self_similar(solver, params, self_similar_path, case_title):
    print("Generating subsonic 2x2 self-similar fitting plots...")
    y_valid = params["y_valid"]
    y_rho = params["y_rho"]
    T_valid = params["T_valid"]
    P_valid = params["P_valid"]
    U_valid = params["U_valid"]
    rho_valid = params["rho_valid"]
    popt_T = params["popt_T"]
    popt_P = params["popt_P"]
    best_u = params["best_u"]
    
    # Compute chosen fit arrays via unified fit_by_params
    T_fit, P_fit, U_fit, _ = fit_by_params(y_valid, params)
    _, _, _, rho_fit = fit_by_params(y_rho, params)
    
    # Evaluate errors on the full y_valid coordinate range (unmasked)
    err_T = np.abs((T_fit - T_valid) / T_valid)
    err_rho = np.abs((rho_fit - rho_valid) / rho_valid)
    err_P = np.abs((P_fit - P_valid) / P_valid)
    err_U = np.abs((U_fit - U_valid) / (U_valid + 1e-15))
    
    avg_T, max_T = np.mean(err_T), np.max(err_T)
    avg_rho, max_rho = np.mean(err_rho), np.max(err_rho)
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = best_u["avg_err"], best_u["max_err"]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # Panel (0,0): Temperature
    ax = axes[0, 0]
    ax.plot(y_valid, T_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, T_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Temperature", fontsize=13, fontweight='bold')
    lbl_T = f"$T(y) \\approx [(1-y)(1+{popt_T[0]:.5f}y)]^{{10/39}}$\nAvg Err: {avg_T:.4e}, Max Err: {max_T:.4e}"
    ax.text(0.05, 0.05, lbl_T, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (0,1): Density
    ax = axes[0, 1]
    ax.plot(y_rho, rho_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_rho, rho_fit, 'r--', label='EOS Derived Fit', lw=1.5)
    ax.set_ylabel(r"Density $\rho(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Density", fontsize=13, fontweight='bold')
    lbl_rho = r"$\rho(y) \approx \left(\frac{P(y)}{rfT(y)^{\beta}}\right)^{\frac{1}{1+\mu}}$" + f"\nAvg Err: {avg_rho:.4e}, Max Err: {max_rho:.4e}"
    ax.text(0.05, 0.05, lbl_rho, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,0): Pressure
    ax = axes[1, 0]
    ax.plot(y_valid, P_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, P_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Pressure $P(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Subsonic: Pressure", fontsize=13, fontweight='bold')
    lbl_P = f"$P(y) \\approx P(0) + {popt_P[0]:.5f} y^{{{popt_P[2]:.5f}}} + {popt_P[1]:.5f} y^{{{popt_P[2]+popt_P[3]:.5f}}}$\nAvg Err: {avg_P:.4e}, Max Err: {max_P:.4e}"
    ax.text(0.05, 0.70, lbl_P, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,1): Velocity
    ax = axes[1, 1]
    ax.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, U_fit, 'r--', label='Optimized Fit', lw=1.5)
    ax.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax.set_title(f"Subsonic: Velocity ({best_u['name']})", fontsize=13, fontweight='bold')
    
    # Format the dynamic velocity formula in latex
    u_0_val, u_f_val = U_valid[0], U_valid[-1]
    if best_u["id"] == 1:
        u_formula = r"$U(y) \approx {:.5f} (1-y)^{{{:.5f}}}$".format(best_u['popt'][0], best_u['popt'][1])
    elif best_u["id"] == 2:
        u_formula = r"$U(y) \approx \frac{{{:.5f} (1-y)}}{{1 + {:.5f} y}}$".format(best_u['popt'][0], best_u['popt'][1])
    elif best_u["id"] == 3:
        u_formula = r"$U(y) \approx {:.4f} + ({:.4f}) (1-y)^{{{:.4f}}} y^{{-{:.4f}}}$".format(u_f_val, u_0_val - u_f_val, best_u['popt'][0], best_u['popt'][1])
    elif best_u["id"] == 4:
        u_formula = r"$U(y) \approx {:.5f} (1-y^{{{:.5f}}}) y^{{-{:.5f}}}$".format(best_u['popt'][0], best_u['popt'][1], best_u['popt'][2])
    elif best_u["id"] == 5:
        u_formula = r"$U(y) \approx {:.4f} + ({:.4f}) (1-y)^{{{:.4f}}} (1 - {:.4f}\ln(y))$".format(u_f_val, u_0_val - u_f_val, best_u['popt'][1], best_u['popt'][0])
    elif best_u["id"] == 6:
        u_formula = r"$U(y) \approx {:.4f} + ({:.4f}) \frac{{1 - y^{{{:.4f}}}}}{{1 + {:.4f} y^{{{:.4f}}}}}$".format(u_f_val, u_0_val - u_f_val, best_u['popt'][1], best_u['popt'][0], best_u['popt'][2])
    elif best_u["id"] == 7:
        u_formula = r"$U(y) \approx {:.4f} + ({:.4f})(1 - y^{{{:.4f}}})^{{{:.4f}}}$".format(u_f_val, u_0_val - u_f_val, best_u['popt'][0], best_u['popt'][1])
    elif best_u["id"] == 8:
        u_formula = r"$U(y) \approx {:.4f} + \frac{{{:.4f}(1 - y)}}{{1 + {:.4f} y}}$".format(u_f_val, u_0_val - u_f_val, best_u['popt'][0])
    elif best_u["id"] == 9:
        u_formula = r"$U(y) \approx Piecewise\ Power-Law$"
    elif best_u["id"] == 10:
        u_formula = r"$U(y) \approx W U_{left} + (1-W) U_{right}$"
    elif best_u["id"] == 11:
        u_formula = r"$U(y) \approx W U_{left,3} + (1-W) U_{right,3}$"
    elif best_u["id"] == 12:
        u_formula = r"$U(y) \approx W U_{left,4} + (1-W) U_{right,4}$"
        
    lbl_U = u_formula + f"\nAvg Err: {avg_U:.4e}, Max Err: {max_U:.4e}"
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


def plot_standalone_velocity_fits(params, standalone_path, case_title):
    print("Generating standalone subsonic velocity fitting comparison plots...")
    y_valid = params["y_valid"]
    U_valid = params["U_valid"]
    fits_u = params["fits_u"]
    best_u = params["best_u"]
    
    fig_sa, (ax_sa1, ax_sa2) = plt.subplots(1, 2, figsize=(18, 8.5))
    
    # Evaluate errors on the full y_valid coordinate range (unmasked)
    y_bulk = y_valid
    
    # Left: Fits vs Numerical
    ax_sa1.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=3.0)
    colors_u = {
        1: 'crimson', 2: 'darkorange', 3: 'forestgreen', 4: 'darkviolet', 
        5: 'deeppink', 6: 'teal', 7: 'chocolate', 8: 'navy', 9: 'black', 10: 'royalblue',
        11: 'brown', 12: 'magenta'
    }
    
    for i in range(1, 13):
        style = '--' if i != best_u["id"] else '-' 
        popt, U_fit, avg_err, max_err, name, latex = fits_u[i]
        if popt is not None:
            lbl = f"Fit {i}: {name}\nAvg Err: {avg_err:.3e}, Max Err: {max_err:.3e}"
            lw = 2.2 if i == best_u["id"] else 1.5
            ax_sa1.plot(y_valid, U_fit, colors_u[i], linestyle=style, label=lbl, lw=lw)
            
            # Right: Semi-log Error curves (using absolute relative errors)
            err_curve = np.abs((U_fit - U_valid) / (U_valid + 1e-15))
            ax_sa2.plot(y_bulk, err_curve, colors_u[i], label=f"Fit {i} (Avg: {avg_err:.3e})", lw=lw)
            
    ax_sa1.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_sa1.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax_sa1.legend(loc='best', fontsize=9.0)
    ax_sa1.grid(True, alpha=0.3)
    ax_sa1.set_title("Dimensionless Velocity $U(y)$ vs 12 Candidates", fontsize=13, fontweight='bold')
    
    ax_sa2.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_sa2.set_ylabel(r"Relative Error", fontsize=12)
    ax_sa2.set_yscale('log')
    ax_sa2.legend(loc='best', fontsize=9.5)
    ax_sa2.grid(True, which="both", ls=":", alpha=0.5)
    ax_sa2.set_title("Relative Errors of Velocity Fits (semi-log)", fontsize=13, fontweight='bold')
    
    fig_sa.suptitle(f"Subsonic Velocity Profile Curve Fitting & Optimization\nChosen Formal Fit: Fit {best_u['id']} ({best_u['name']})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_sa.savefig(standalone_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sa)
    print(f"Saved standalone subsonic velocity fits to {standalone_path}")


def plot_relative_errors(solver, params, relative_errors_path, case_title):
    print("Generating subsonic relative error plots...")
    y_valid = params["y_valid"]
    y_rho = params["y_rho"]
    T_valid = params["T_valid"]
    P_valid = params["P_valid"]
    U_valid = params["U_valid"]
    rho_valid = params["rho_valid"]
    best_u = params["best_u"]
    
    # Call unified fit_by_params for dimensionless fits on y_valid
    T_fit, P_fit, U_fit, _ = fit_by_params(y_valid, params)
    _, _, _, rho_fit = fit_by_params(y_rho, params)
    
    err_T = np.abs((T_fit - T_valid) / T_valid)
    err_rho = np.abs((rho_fit - rho_valid) / rho_valid)
    err_P = np.abs((P_fit - P_valid) / P_valid)
    err_U = np.abs((U_fit - U_valid) / (U_valid + 1e-15))
    
    avg_T, max_T = np.mean(err_T), np.max(err_T)
    avg_rho, max_rho = np.mean(err_rho), np.max(err_rho)
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = best_u["avg_err"], best_u["max_err"]
    
    fig_err, ax_err = plt.subplots(figsize=(10, 7.5))
    
    ax_err.plot(y_valid, err_T, color='blue', label=f'Temperature $T(y)$ (Avg: {avg_T:.3e}, Max: {max_T:.3e})', lw=2.0)
    ax_err.plot(y_rho, err_rho, color='green', label=f'Density $\\rho(y)$ (Avg: {avg_rho:.3e}, Max: {max_rho:.3e})', lw=2.0)
    ax_err.plot(y_valid, err_P, color='red', label=f'Pressure $P(y)$ (Avg: {avg_P:.3e}, Max: {max_P:.3e})', lw=2.0)
    ax_err.plot(y_valid, err_U, color='purple', label=f'Velocity $U(y)$ (Avg: {avg_U:.3e}, Max: {max_U:.3e})', lw=2.0)
    
    ax_err.set_xlabel(r"Normalized coordinate $y = \xi / \xi_f$", fontsize=12)
    ax_err.set_ylabel(r"Relative Error", fontsize=12)
    ax_err.set_yscale('log')
    ax_err.set_ylim(bottom=1e-5)
    ax_err.grid(True, which="both", ls=":", alpha=0.5)
    ax_err.legend(loc="best", fontsize=10.5)
    ax_err.set_title(f"Relative Errors of Subsonic self-similar Fits (semi-log)\n{case_title}", fontsize=13, fontweight='bold')
    
    fig_err.tight_layout()
    fig_err.savefig(relative_errors_path, dpi=200)
    plt.close(fig_err)
    print(f"Saved subsonic relative errors to {relative_errors_path}")


def get_plot_paths(case_label: str) -> dict[str, str]:
    out_dir = Path("results/ictt")
    ss_dir = out_dir / "self_similar"
    dv_dir = out_dir / "dimensional_verification"
    ss_dir.mkdir(parents=True, exist_ok=True)
    dv_dir.mkdir(parents=True, exist_ok=True)

    return {
        "self_similar": str(ss_dir / f"{case_label}_self_similar.png"),
        "velocity_fits": str(ss_dir / f"{case_label}_velocity_fits_standalone.png"),
        "relative_errors": str(ss_dir / f"{case_label}_relative_errors.png"),
        "dimensional_comparison": str(dv_dir / f"{case_label}_dimensional_fit_comparison.png")
    }


def generate_verification_plots(
    history,
    case,
    solver,
    params: dict,
    case_label: str,
    case_title: str,
):
    """Generate analytical self-similar fits, dimensional comparison, and relative error plots."""
    paths = get_plot_paths(case_label)

    # 1) Standalone Velocity fits comparison
    plot_standalone_velocity_fits(params, paths["velocity_fits"], case_title)

    # 2) Self-similar profiles and fits
    plot_and_fit_self_similar(solver, params, paths["self_similar"], case_title)

    # 3) Dimensional fit comparison
    plot_dimensional_fit_comparison(history, solver, case, params, paths["dimensional_comparison"], case_title)

    # 4) Relative errors of self-similar fits
    plot_relative_errors(solver, params, paths["relative_errors"], case_title)


def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run full verification comparison pipeline for a given preset."""
    print("=" * 80)
    print(f"PROCESSING PRESET: {preset_name} -> {case_label}")
    print("=" * 80)

    case, history, solver = run_simulation_and_references(preset_name, case_label)
    params = perform_subsonic_fitting(solver)

    generate_verification_plots(
        history=history,
        case=case,
        solver=solver,
        params=params,
        case_label=case_label,
        case_title=case_title,
    )
    print(f"Preset {preset_name} processed successfully.")


def main():
    run_preset_workflow(
        PRESET_FIG_8_CONSTANT_TEMPERATURE,
        "constant_boundary_temperature_tau_0",
        "Constant Boundary Temperature (tau=0)"
    )
    print("\nAll custom subsonic simulations, SubsonicHeatWave comparisons, plotting, fitting, and exports completed successfully!")


if __name__ == "__main__":
    main()

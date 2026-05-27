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

RUN_GIFS = False  # Set to True to generate animated evolution GIFs (which takes time)

def get_cached_shock_solver(case, case_label):
    """Solve shock similarity ODEs once and cache the solver object (with found xsi_s)."""
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    solver_cache_path = cache_dir / f"{case_label}_similarity_solver.pkl"
    
    if solver_cache_path.exists():
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

def plot_xt_trajectories(history, case, xt_path, case_title, solver=None):
    """Plot cell boundaries x(t) and diagnosed fronts (Simulation vs Analytic)."""
    print(f"Generating space-time (xt) plot for {case_title}...")
    times = np.asarray(history.t, dtype=float)
    x_sim = np.asarray(history.x, dtype=float)
    n_cells = x_sim.shape[1] - 1
    
    if solver is None:
        solver = PistonShock(**_shock_kwargs_from_case(case))
    mass_grid = _build_mass_grid(case, num_cells=n_cells)
    
    times_model = _get_equally_spaced_elements(times, 200)
    results = []
    for t in times_model:
        sol = solver.solve(mass=mass_grid, time=max(float(t), 1e-18))
        results.append(sol)
        
    position_times = np.array([r["position"] for r in results]).T
    shock_position = np.array([r["shock_position"] for r in results], dtype=float)
    piston_position = np.array([r["piston_position"] for r in results], dtype=float)
    
    # 2) Simulation piston/shock trajectories
    x_piston_sim = np.full(times.size, np.nan, dtype=float)
    x_shock_sim = np.full(times.size, np.nan, dtype=float)
    p0_barye = float(case.P0_Barye)
    tau = float(case.tau)
    
    for k, t in enumerate(times):
        xk = np.asarray(history.x[k], dtype=float)
        pk = np.asarray(history.p[k], dtype=float)
        rhok = np.asarray(history.rho[k], dtype=float)
        mk = np.asarray(history.m[k], dtype=float)
        
        # Boundary pressure
        if t <= 0.0:
            p_drive = 0.0
        else:
            p_drive = p0_barye * ((t / 1e-9) ** tau)
            
        _, x_p = _sim_piston_position(xk, pk, p_drive, rel_tol=0.05)
        x_piston_sim[k] = x_p
        
        rhok_smooth = _rolling_mean(rhok, 5)
        ishock, _ = find_shock_front(
            rhok_smooth,
            mk,
            rho_unshocked=float(case.rho0),
            gamma=float(case.r) + 1.0,
            Hugoniot_threshold=0.5,
        )
        if ishock >= 1:
            x_shock_sim[k] = float(xk[ishock])
            
    # Apply linear regression correction for small times (t < 0.002 ns) in shock detection
    later_times = np.array([])
    later_x = np.array([])
    for k in range(1, times.size):
        t_ns = times[k] * 1e9
        if t_ns >= 0.002 and not np.isnan(x_shock_sim[k]):
            later_times = np.append(later_times, times[k])
            later_x = np.append(later_x, x_shock_sim[k])
            
    if len(later_times) >= 2:
        slope, intercept = np.polyfit(np.log(later_times+1e-20), np.log(later_x), 1)
        for k in range(1, times.size):
            t_ns = times[k] * 1e9
            if t_ns < 0.002:
                x_shock_sim[k] = np.exp(slope * np.log(times[k] + 1e-20) + intercept)
                
    # 3) Setup figure
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    
    # Plot mass trajectories
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
    ax.plot(times * 1e9, x_shock_sim, lw=2.5, c="red", label="Shock (simulation)")
    ax.plot(times_model * 1e9, shock_position, lw=2.0, ls="--", c="darkred", label="Shock (Menahem)")
    ax.plot(times * 1e9, x_piston_sim, lw=2.5, c="green", label="Piston (simulation)")
    ax.plot(times_model * 1e9, piston_position, lw=2.0, ls="--", c="darkgreen", label="Piston (Menahem)")
    
    ax.set_xlabel(r"$t$ [ns]", fontsize=12)
    ax.set_ylabel(r"$x$ [cm]", fontsize=12)
    ax.set_title(f"Space-Time (xt) Trajectories and Fronts\n{case_title}", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_xlim(0, times.max() * 1e9)
    ax.set_ylim(0, max(shock_position.max(), x_shock_sim[np.isfinite(x_shock_sim)].max()) * 1.05)
    
    fig.tight_layout()
    fig.savefig(xt_path, dpi=200)
    plt.close(fig)
    print(f"Saved xt trajectory to {xt_path}")


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
    T_s_cgs   = p_s_cgs / (rho_s_cgs * float(case.r)) if rho_s_cgs > 0 else 300.0

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
    R_val = np.where(V_val > 0, 1.0 / V_val, np.nan)
    T_val = P_val * V_val / solver.r
    
    valid_idx = (y_grid > 0.005) & np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(T_val)
    y_valid = y_grid[valid_idx]
    T_valid = T_val[valid_idx]
    P_valid = P_val[valid_idx]
    U_valid = U_val[valid_idx]
    R_valid = R_val[valid_idx]
    
    U_0 = U_valid[0]
    U_s = U_valid[-1]
    
    # Pressure Fit (1-parameter power law: P = 1 - (1-Ps)*y^d)
    def power_law_P(y, d):
        return 1.0 - (1.0 - solver.Ps) * (y**d)
        
    popt_P, _ = curve_fit(power_law_P, y_valid, P_valid, p0=[1.0])
    
    # Temperature Fit (1-parameter power law: T = Ts*y^d)
    # Ts is not a PistonShock attribute; compute from EOS: T = P*V/r at shock boundary
    Ts = float(solver.Ps * solver.Rs_or_Vs / solver.r)
    def power_law_T(y, d):
        return Ts * (y**d)
        
    popt_T, _ = curve_fit(power_law_T, y_valid, T_valid, p0=[-0.3])
    
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
            
    return y_grid, y_valid, T_valid, P_valid, U_valid, R_valid, popt_P, popt_T, best_u, fits_u


def plot_material_hydro_profiles(history, solver, case, popt_P, popt_T, best_u, material_hydro_path, case_title):
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
            exact_T = np.where(exact_rho > 0, sol_exact["pressure"] / (exact_rho * float(case.r)), 0.0)
            
        # 3) Analytical fits mapped to CGS
        fits = evaluate_shock_fits(mass_exact, t_actual, solver, case, popt_P, popt_T, best_u)
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
    solver, y_valid, T_valid, P_valid, U_valid, R_valid, 
    popt_P, popt_T, best_u, fits_u, 
    self_similar_path, standalone_path, case_title
):
    print("Generating shock 2x2 self-similar fitting plots...")
    
    # Evaluate fits
    P_fit = 1.0 - (1.0 - solver.Ps) * y_valid**popt_P[0]
    T_fit = solver.Ts * y_valid**popt_T[0]
    R_fit = P_fit / (solver.r * T_fit)
    U_fit = best_u["fit_val"]
    
    err_T = np.abs((T_fit - T_valid) / T_valid) * 100
    err_R = np.abs((R_fit - R_valid) / R_valid) * 100
    err_P = np.abs((P_fit - P_valid) / P_valid) * 100
    err_U = np.abs((U_fit - U_valid) / (U_valid + 1e-15)) * 100
    
    avg_T, max_T = np.mean(err_T), np.max(err_T)
    avg_R, max_R = np.mean(err_R), np.max(err_R)
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = best_u["avg_err"], best_u["max_err"]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # Panel (0,0): Temperature
    ax = axes[0, 0]
    ax.plot(y_valid, T_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, T_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Shock: Temperature", fontsize=13, fontweight='bold')
    lbl_T = f"$T(y) \\approx T_s y^{{{popt_T[0]:.5f}}}$\nAvg Err: {avg_T:.4f}%, Max Err: {max_T:.4f}%"
    ax.text(0.05, 0.05, lbl_T, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (0,1): Density
    ax = axes[0, 1]
    ax.plot(y_valid, R_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, R_fit, 'r--', label='EOS Derived Fit', lw=1.5)
    ax.set_ylabel(r"Density $R(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Shock: Density", fontsize=13, fontweight='bold')
    lbl_R = r"$R(y) \approx \frac{P(y)}{(\gamma - 1) T(y)}$" + f"\nAvg Err: {avg_R:.4f}%, Max Err: {max_R:.4f}%"
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


def plot_relative_errors(
    solver, y_valid, T_valid, P_valid, U_valid, R_valid, 
    popt_P, popt_T, best_u, relative_errors_path, case_title
):
    print("Generating shock relative error plots...")
    P_fit = 1.0 - (1.0 - solver.Ps) * y_valid**popt_P[0]
    T_fit = solver.Ts * y_valid**popt_T[0]
    R_fit = P_fit / (solver.r * T_fit)
    U_fit = best_u["fit_val"]
    
    err_T = np.abs((T_fit - T_valid) / T_valid) * 100
    err_R = np.abs((R_fit - R_valid) / R_valid) * 100
    err_P = np.abs((P_fit - P_valid) / P_valid) * 100
    err_U = np.abs((U_fit - U_valid) / (U_valid + 1e-15)) * 100
    
    avg_T, max_T = np.mean(err_T), np.max(err_T)
    avg_R, max_R = np.mean(err_R), np.max(err_R)
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = best_u["avg_err"], best_u["max_err"]
    
    fig_err, ax_err = plt.subplots(figsize=(10, 7.5))
    ax_err.plot(y_valid, err_T, label=f'Temperature $T(y)$ (Avg: {avg_T:.4f}%, Max: {max_T:.4f}%)', lw=2.0)
    ax_err.plot(y_valid, err_R, label=f'Density $R(y)$ (Avg: {avg_R:.4f}%, Max: {max_R:.4f}%)', lw=2.0)
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

def save_custom_evolution_gif(
    history,
    case,
    menahem_ref,
    gif_path: str,
    fps: int = 12,
    stride: int = 1,
    subtitle: str | None = None,
):
    """Generate animated 5-way GIF comparing Simulation vs Menahem PistonShock reference."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    p_scale, u_scale, e_scale = 1e12, 1e5, 1e9
    
    # Construct 5 vertical panels
    fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    k0 = 0
    m0 = history.m[k0]
    
    # 1) Simulation lines (blue solid)
    sim_lines = []
    sim_lines.append(axes[0].plot(m0, history.rho[k0], lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[1].plot(m0, history.p[k0] / p_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[2].plot(m0, history.u[k0] / u_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[3].plot(m0, history.e[k0] / e_scale, lw=2, label="simulation", color="blue")[0])
    sim_lines.append(axes[4].plot(m0, history.T[k0] if hasattr(history, 'T') else np.zeros_like(m0), lw=2, label="simulation", color="blue")[0])
    
    # 2) Menahem PistonShock lines (magenta/dashed)
    men_lines = []
    men_lines.append(axes[0].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[1].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[2].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[3].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    men_lines.append(axes[4].plot([], [], lw=1.5, color="magenta", linestyle="--", label="Menahem")[0])
    
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        
    x_mass = r"Mass coordinate $m$ [g/cm²]"
    axes[0].set_ylabel(r"$\rho$ [g/cm³]")
    axes[1].set_ylabel(r"$P$ [MBar]")
    axes[2].set_ylabel(r"$u$ [km/s]")
    axes[3].set_ylabel(r"$e_{\mathrm{mat}}$ [$10^9$ erg/g]")
    axes[4].set_ylabel(r"$T_{\mathrm{mat}}$ [K]")
    axes[4].set_xlabel(x_mass)
    
    title = fig.suptitle("", fontweight="medium")
    frame_ids = np.arange(0, len(history.t), stride)
    
    def init():
        return sim_lines + men_lines
        
    def update(frame_idx):
        k = int(frame_ids[frame_idx])
        mk = history.m[k]
        t = history.t[k]
        sim_lines[0].set_data(mk, history.rho[k])
        sim_lines[1].set_data(mk, history.p[k] / p_scale)
        sim_lines[2].set_data(mk, history.u[k] / u_scale)
        sim_lines[3].set_data(mk, history.e[k] / e_scale)
        
        # Temp from ideal gas relation: p / (rho * r)
        with np.errstate(divide='ignore', invalid='ignore'):
            Tk_sim = np.where(history.rho[k] > 0, history.p[k] / (history.rho[k] * float(case.r)), 0.0)
        sim_lines[4].set_data(mk, Tk_sim)
        
        if menahem_ref is not None:
            men_idx = int(np.argmin(np.abs(menahem_ref.times - t)))
            mm = menahem_ref.m[men_idx]
            men_lines[0].set_data(mm, menahem_ref.rho[men_idx])
            men_lines[1].set_data(mm, menahem_ref.p[men_idx] / p_scale)
            men_lines[2].set_data(mm, menahem_ref.u[men_idx] / u_scale)
            men_lines[3].set_data(mm, menahem_ref.e[men_idx] / e_scale)
            
            # Temp from ideal gas relation
            with np.errstate(divide='ignore', invalid='ignore'):
                Tk_men = np.where(menahem_ref.rho[men_idx] > 0, menahem_ref.p[men_idx] / (menahem_ref.rho[men_idx] * float(case.r)), 0.0)
            men_lines[4].set_data(mm, Tk_men)
            
        case_title = case.title if hasattr(case, "title") and case.title else "Simulation"
        header = f"{case_title}\n{subtitle}" if subtitle else case_title
        title.set_text(
            f"{header}\n"
            f"$P(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0_Barye:.3e},\\; \\tau={case.tau},\\; t={t*1e9:.3f}$ ns"
        )
        
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        return sim_lines + men_lines
        
    anim = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=False)
    
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved custom GIF to {gif_path}")


class ReferenceContainer:
    def __init__(self, times, m, x, rho, p, u, e):
        self.times = times
        self.m = m
        self.x = x
        self.rho = rho
        self.p = p
        self.u = u
        self.e = e


def run_simulation_and_references(preset_name: str, case_label: str):
    """Run full simulation and build PistonShock reference solver, or load from cache."""
    from dataclasses import replace as dc_replace
    cache_dir = Path("results/ictt/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case_label}_cache.pkl"

    case, config = get_preset(preset_name)
    config = dc_replace(config, N=400)

    if cache_path.exists():
        print(f"Loading cached simulation and reference data from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data["case"], data["history"], data["menahem_ref"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running simulation...")

    # Run simulation
    print("Running simulation...")
    _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)

    # Build PistonShock reference for the menahem_ref container
    print("Building PistonShock reference...")
    times_sec = np.array(history.t)
    times_sec = times_sec[times_sec > 0.0]

    kwargs = _shock_kwargs_from_case(case)
    kwargs["omega"] = float(getattr(case, "omega", 0.0))
    ref_solver = PistonShock(**kwargs)

    mass = _build_mass_grid(case, num_cells=400)
    m_list, x_list = [], []
    rho_list, p_list, u_list, e_list = [], [], [], []
    for t in times_sec:
        sol = ref_solver.solve(mass=mass, time=float(t))
        m_list.append(mass.copy())
        x_list.append(np.asarray(sol["position"], dtype=float))
        rho_list.append(np.asarray(sol["density"], dtype=float))
        p_list.append(np.asarray(sol["pressure"], dtype=float))
        u_list.append(np.asarray(sol["velocity"], dtype=float))
        e_list.append(np.asarray(sol["sie"], dtype=float))

    menahem_ref = ReferenceContainer(times_sec, m_list, x_list, rho_list, p_list, u_list, e_list)

    cache_data = {"case": case, "history": history, "menahem_ref": menahem_ref}
    print(f"Saving simulation and reference data cache to {cache_path}...")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return case, history, menahem_ref


def generate_verification_plots(
    history,
    case,
    menahem_ref,
    case_label: str,
    case_title: str,
):
    """Generate all verification comparisons, plotting, fitting, and animated GIF."""
    out_dir = Path("results/ictt")
    xt_dir = out_dir / "xt"
    mh_dir = out_dir / "material_hydro"
    ss_dir = out_dir / "self_similar"
    ev_dir = out_dir / "evolution"

    for d in [xt_dir, mh_dir, ss_dir, ev_dir]:
        d.mkdir(parents=True, exist_ok=True)

    xt_path = str(xt_dir / f"{case_label}_xt.png")
    material_hydro_path = str(mh_dir / f"{case_label}_material_hydro.png")
    self_similar_path = str(ss_dir / f"{case_label}_self_similar.png")
    standalone_path = str(ss_dir / f"{case_label}_velocity_fits_standalone.png")
    relative_errors_path = str(ss_dir / f"{case_label}_relative_errors.png")
    gif_path = str(ev_dir / f"{case_label}_evolution.gif")

    # Solve/load cached shock solver
    solver = get_cached_shock_solver(case, case_label)
    y_grid, y_valid, T_valid, P_valid, U_valid, R_valid, popt_P, popt_T, best_u, fits_u = perform_shock_fitting(solver)

    # 1) Generate space-time trajectories and fronts
    plot_xt_trajectories(history, case, xt_path, case_title, solver=solver)

    # 2) Plot physical material hydrodynamics comparison
    plot_material_hydro_profiles(history, solver, case, popt_P, popt_T, best_u, material_hydro_path, case_title)

    # 3) Self-similar profiles and fits
    plot_and_fit_self_similar(solver, y_valid, T_valid, P_valid, U_valid, R_valid, popt_P, popt_T, best_u, fits_u, self_similar_path, standalone_path, case_title)

    # 4) Relative errors of self-similar fits
    plot_relative_errors(solver, y_valid, T_valid, P_valid, U_valid, R_valid, popt_P, popt_T, best_u, relative_errors_path, case_title)

    # 5) Animated evolution GIF (conditional)
    if RUN_GIFS:
        print(f"Saving animated custom 5-way evolution GIF to {gif_path}...")
        save_custom_evolution_gif(
            history=history,
            case=case,
            menahem_ref=menahem_ref,
            gif_path=gif_path,
            fps=12,
            stride=max(1, len(history.t) // 60),
            subtitle="Simulation vs Menahem PistonShock",
        )
    else:
        print("Skipping evolution GIF generation (RUN_GIFS is set to False)...")


def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run full verification comparison pipeline for a given preset."""
    print("=" * 80)
    print(f"PROCESSING PRESET: {preset_name} -> {case_label}")
    print("=" * 80)

    case, history, menahem_ref = run_simulation_and_references(preset_name, case_label)

    generate_verification_plots(
        history=history,
        case=case,
        menahem_ref=menahem_ref,
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
        "Power-Law Pressure Drive (tau=-0.45)"
    )
    print("\nAll custom simulations, PistonShock comparisons, plotting, fitting, and exports completed successfully!")


if __name__ == "__main__":
    main()

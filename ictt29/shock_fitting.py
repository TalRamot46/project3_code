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
sys.setrecursionlimit(200000)
import pickle
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import sys
import numpy.core
sys.modules['numpy._core'] = sys.modules.get('numpy.core')
sys.modules['numpy._core.numeric'] = sys.modules.get('numpy.core.numeric')
sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray')
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
    PRESET_FIG_9_CONSTANT_FLUX,
    PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
)
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _shock_kwargs_from_case,
    _ns_amplitude_rescale,
    _build_mass_grid,
)
from project3_code.hydro_sim.plotting.hydro_plots import _create_7panel_vertical_figure

from project3_code.menahem_new.piston_shock_og import PistonShock

USE_CACHE = True 

Y_FIT_MIN = 0.1   # Configure the lower bound for fitting Temperature T
FITTING_OPTION = "FIT_RHO_AROUND_PISTON"  # Literal["FIT_TEMP_AROUND_PISTON", "FIT_RHO_AROUND_PISTON", "FIT_RHO_ALL_AROUND"]

# Deleted (Functionality merged into data_loader.py)

def dimensional_temperature_from_eos(P, V, beta=1.6, mu=0.14, r=0.25, f=6730.91):
    return (P * V**(1-mu) / (r*f))**(1./beta)

def dimensionless_temperature_from_eos(P, V, beta=1.6, mu=0.14, r=0.25, f=6730.91):
    return (P * V**(1-mu))**(1./beta)

def perform_shock_fitting(solver):
    y_grid = np.linspace(0.0, 1.0, 500)
    xsi_vec = y_grid * solver.xsi_s
    xsi_vec[0] = 1e-10
    
    V_val, U_val, P_val = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    rho_val = np.where(V_val > 0, 1.0 / V_val, np.nan)

    # calculate temperature using the exact uniform EOS formula
    T_val = dimensional_temperature_from_eos(P_val, V_val)
    
    # making sure to bypass cells with inf value
    valid_idx = np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(T_val)
    y_valid = y_grid[valid_idx]
    T_valid = T_val[valid_idx]
    P_valid = P_val[valid_idx]
    U_valid = U_val[valid_idx]
    rho_valid = rho_val[valid_idx]
    
    # Fits:
    U_0 = U_valid[0]
    U_s = U_valid[-1]
    
    # Pressure Fit (1-parameter power law: P = 1 - (1-Ps)*y^d)
    def power_law_P(y, d):
        return 1.0 - (1.0 - solver.Ps) * (y**d)
        
    popt_P, _ = curve_fit(power_law_P, y_valid, P_valid, p0=[1.0])
    P_fit = 1.0 - (1.0 - solver.Ps) * y_valid**popt_P[0]

    # For debugging purposes:
    # plt.plot(y_valid, np.abs(P_valid - P_fit), label='abs')
    # plt.plot(y_valid, np.abs(P_valid - P_fit) / P_valid, label='rel')
    # plt.yscale('log')
    # plt.show()
    # plt.plot(y_valid, P_valid, label='solver')
    # plt.plot(y_valid, P_fit, label='fit')
    # plt.ylim(1.0, 1.2)
    # plt.yscale('log')
    # plt.show()
    # Temperature or Density Fit Candidate Selection and Optimization
    Ts = T_valid[-1]
    T_0 = T_valid[0]
    rho_s = rho_valid[-1]
    rho_0 = rho_valid[0]
    
    # Define candidates and mask based on FITTING_OPTION
    if FITTING_OPTION == "FIT_TEMP_AROUND_PISTON":
        def fit_cand_1(y, a):
            return Ts + (T_0 - Ts) * (1.0 - y)**a
            
        def fit_cand_2(y, c, a, b):
            return Ts + c * (1.0 - y)**a + (T_0 - Ts - c) * (1.0 - y)**b
            
        def fit_cand_3(y, a, b):
            return Ts + (T_0 - Ts) * (1.0 - y)**a * np.exp(-b * y)
            
        def fit_cand_4(y, c1, c2, a, b):
            return Ts + c1 * (1.0 - y)**a + c2 * (1.0 - y)**b
            
        def fit_cand_5(y, a, b):
            return Ts + (T_0 - Ts) * (1.0 - y)**a / (1.0 + b * y)
            
        candidates_T = [
            {"id": 1, "func": fit_cand_1, "name": "Single Power Law (BC)", "latex": r"T(y) \approx T_s + (T_0-T_s)(1-y)^{%.5f}", "p0": [1.0]},
            {"id": 2, "func": fit_cand_2, "name": "Double Power Law (BC)", "latex": r"T(y) \approx T_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}", "p0": [(T_0-Ts)/2.0, 1.0, 2.0]},
            {"id": 3, "func": fit_cand_3, "name": "Exponential-Damped Power Law (BC)", "latex": r"T(y) \approx T_s + (T_0-T_s)(1-y)^{%.5f} e^{-%.5f y}", "p0": [1.0, 0.1]},
            {"id": 4, "func": fit_cand_4, "name": "Linear Combo (Shock-Conforming)", "latex": r"T(y) \approx T_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}", "p0": [(T_0-Ts)/2.0, (T_0-Ts)/2.0, 1.0, 2.0]},
            {"id": 5, "func": fit_cand_5, "name": "Rational Power Law (BC)", "latex": r"T(y) \approx T_s + (T_0-T_s)\frac{(1-y)^{%.5f}}{1+%.5f y}", "p0": [1.0, 0.1]}
        ]
        
        fit_mask_T = y_valid >= Y_FIT_MIN
        y_fit_data = T_valid[fit_mask_T]
        
    else: # FIT_RHO_AROUND_PISTON or FIT_RHO_ALL_AROUND
        def fit_cand_1(y, a):
            return rho_s + (rho_0 - rho_s) * (1.0 - y)**a
            
        def fit_cand_2(y, c, a, b):
            return rho_s + c * (1.0 - y)**a + (rho_0 - rho_s - c) * (1.0 - y)**b
            
        def fit_cand_3(y, a, b):
            return rho_s + (rho_0 - rho_s) * (1.0 - y)**a * np.exp(-b * y)
            
        def fit_cand_4(y, c1, c2, a, b):
            return rho_s + c1 * (1.0 - y)**a + c2 * (1.0 - y)**b
            
        def fit_cand_5(y, a, b):
            return rho_s + (rho_0 - rho_s) * (1.0 - y)**a / (1.0 + b * y)
            
        candidates_T = [
            {"id": 1, "func": fit_cand_1, "name": "Single Power Law (BC)", "latex": r"\rho(y) \approx \rho_s + (\rho_0-\rho_s)(1-y)^{%.5f}", "p0": [1.0]},
            {"id": 2, "func": fit_cand_2, "name": "Double Power Law (BC)", "latex": r"\rho(y) \approx \rho_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}", "p0": [(rho_0-rho_s)/2.0, 1.0, 2.0]},
            {"id": 3, "func": fit_cand_3, "name": "Exponential-Damped Power Law (BC)", "latex": r"\rho(y) \approx \rho_s + (\rho_0-\rho_s)(1-y)^{%.5f} e^{-%.5f y}", "p0": [1.0, 0.1]},
            {"id": 4, "func": fit_cand_4, "name": "Linear Combo (Shock-Conforming)", "latex": r"\rho(y) \approx \rho_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}", "p0": [(rho_0-rho_s)/2.0, (rho_0-rho_s)/2.0, 1.0, 2.0]},
            {"id": 5, "func": fit_cand_5, "name": "Rational Power Law (BC)", "latex": r"\rho(y) \approx \rho_s + (\rho_0-\rho_s)\frac{(1-y)^{%.5f}}{1+%.5f y}", "p0": [1.0, 0.1]}
        ]
        
        if FITTING_OPTION == "FIT_RHO_AROUND_PISTON":
            fit_mask_T = y_valid >= Y_FIT_MIN
        else: # FIT_RHO_ALL_AROUND
            fit_mask_T = np.ones_like(y_valid, dtype=bool)
            
        y_fit_data = rho_valid[fit_mask_T]
        
    best_T = None
    min_avg_err_T = float("inf")
    fits_T = {}
    
    for cand in candidates_T:
        try:
            popt, _ = curve_fit(cand["func"], y_valid[fit_mask_T], y_fit_data, p0=cand["p0"], maxfev=10000)
            fit_val = cand["func"](y_valid, *popt)
            
            # Derive derived Temperature or Density for error computation
            if FITTING_OPTION == "FIT_TEMP_AROUND_PISTON":
                T_fit = fit_val
                rho_fit = (P_fit / (6730.0 * solver.r * T_fit**1.6))**(1.0/1.14)
            elif FITTING_OPTION == "FIT_RHO_ALL_AROUND":
                rho_fit = fit_val
                T_fit = (P_fit / (6730.0 * solver.r * rho_fit**0.86))**(1.0/1.6)
            else: # FIT_RHO_AROUND_PISTON
                rho_fit_high = fit_val[fit_mask_T]
                T_fit_high = (P_fit[fit_mask_T] / (6730.0 * solver.r * rho_fit_high**0.86))**(1.0/1.6)
                T_fit = np.zeros_like(y_valid)
                T_fit[fit_mask_T] = T_fit_high
                
            # Calculate errors inside the active fitting domain in absolute values
            T_fit_domain = T_fit[fit_mask_T]
            T_valid_domain = T_valid[fit_mask_T]
            rel_err_T = np.abs((T_fit_domain - T_valid_domain) / T_valid_domain)
            
            avg_err = np.mean(rel_err_T)
            max_err = np.max(rel_err_T)
            
            # Format the latex_str
            if FITTING_OPTION == "FIT_TEMP_AROUND_PISTON":
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
            else:
                if cand["id"] == 1:
                    latex_str = r"\rho(y) \approx \rho_s + (\rho_0-\rho_s)(1-y)^{%.5f}" % tuple(popt)
                elif cand["id"] == 2:
                    c, a, b = popt
                    latex_str = r"\rho(y) \approx \rho_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}" % (c, a, rho_0 - rho_s - c, b)
                elif cand["id"] == 3:
                    latex_str = r"\rho(y) \approx \rho_s + (\rho_0-\rho_s)(1-y)^{%.5f} e^{-%.5f y}" % tuple(popt)
                elif cand["id"] == 4:
                    c1, c2, a, b = popt
                    latex_str = r"\rho(y) \approx \rho_s + %.5f (1-y)^{%.5f} + %.5f (1-y)^{%.5f}" % (c1, a, c2, b)
                elif cand["id"] == 5:
                    latex_str = r"\rho(y) \approx \rho_s + (\rho_0-\rho_s)\frac{(1-y)^{%.5f}}{1+%.5f y}" % tuple(popt)
                    
            fits_T[cand["id"]] = (popt, fit_val, avg_err, max_err, cand["name"], latex_str)
            
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
                    "fit_val": fit_val
                }
        except Exception as e:
            print(f"Shock temperature/density fit {cand['id']} failed: {e}")
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
            rel_err_u = np.abs((U_fit - U_valid) / (U_valid + 1e-15))
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
            
    # Construct piecewise composite profiles for all candidates and the chosen best_T
    if FITTING_OPTION == "FIT_RHO_ALL_AROUND":
        low_mask = y_valid < 0.0 # empty
        high_mask = np.ones_like(y_valid, dtype=bool) # full
    else:
        low_mask = y_valid < Y_FIT_MIN
        high_mask = y_valid >= Y_FIT_MIN
        
    P_fit = 1.0 - (1.0 - solver.Ps) * y_valid**popt_P[0]
    
    fits_T_composite = {}
    for cand_id, (popt_cand, fit_val_cand, avg_err_cand, max_err_cand, name, latex_str) in fits_T.items():
        if popt_cand is not None:
            if FITTING_OPTION == "FIT_RHO_ALL_AROUND":
                rho_fit_comp_cand = fit_val_cand
                T_fit_comp_cand = (P_fit / (6730.0 * solver.r * rho_fit_comp_cand**0.86))**(1.0/1.6)
                fits_T_composite[cand_id] = (popt_cand, T_fit_comp_cand, rho_fit_comp_cand, avg_err_cand, max_err_cand, name, latex_str)
            else:
                rho_0 = rho_valid[0]
                rho_s = rho_valid[-1]
                
                # Fit density in low domain: fit_rho_around_zero
                def fit_rho_around_zero(y, a, d):
                    return rho_s + (rho_0 - rho_s)*(1-(y**d))**a
                    
                popt_rho_low_cand, _ = curve_fit(fit_rho_around_zero, y_valid[low_mask], rho_valid[low_mask], p0=[1.0, 1.0], maxfev=10000)
                rho_fit_low_cand = fit_rho_around_zero(y_valid[low_mask], *popt_rho_low_cand)
                T_fit_low_cand = (P_fit[low_mask] / (6730.0 * solver.r * rho_fit_low_cand**0.86))**(1.0/1.6)
                
                if FITTING_OPTION == "FIT_TEMP_AROUND_PISTON":
                    T_fit_high_cand = fit_val_cand[high_mask]
                    rho_fit_high_cand = (P_fit[high_mask] / (6730.0 * solver.r * T_fit_high_cand**1.6))**(1.0/0.86)
                else: # FIT_RHO_AROUND_PISTON
                    rho_fit_high_cand = fit_val_cand[high_mask]
                    T_fit_high_cand = (P_fit[high_mask] / (6730.0 * solver.r * rho_fit_high_cand**0.86))**(1.0/1.6)
                    
                T_fit_comp_cand = np.zeros_like(y_valid)
                T_fit_comp_cand[low_mask] = T_fit_low_cand
                T_fit_comp_cand[high_mask] = T_fit_high_cand
                
                rho_fit_comp_cand = np.zeros_like(y_valid)
                rho_fit_comp_cand[low_mask] = rho_fit_low_cand
                rho_fit_comp_cand[high_mask] = rho_fit_high_cand
                
                # Recalculate average and max error on derived temperature over the fitting mask for consistency
                rel_err_cand = np.abs((T_fit_comp_cand[high_mask] - T_valid[high_mask]) / T_valid[high_mask])
                avg_err_comp = np.mean(rel_err_cand)
                max_err_comp = np.max(rel_err_cand)
                
                fits_T_composite[cand_id] = (popt_cand, T_fit_comp_cand, rho_fit_comp_cand, avg_err_comp, max_err_comp, name, latex_str)
        else:
            fits_T_composite[cand_id] = (None, None, None, 0.0, 0.0, name, latex_str)
            
    # Retrieve low-domain density fit parameters for the composite model
    popt_rho_low = None
    rho_0 = rho_valid[0]
    if FITTING_OPTION != "FIT_RHO_ALL_AROUND" and np.any(low_mask):
        rho_s = rho_valid[-1]
        def fit_rho_around_zero(y_val, a, d):
            return rho_s + (rho_0 - rho_s) * (1.0 - y_val**d)**a
        popt_rho_low, _ = curve_fit(fit_rho_around_zero, y_valid[low_mask], rho_valid[low_mask], p0=[1.0, 1.0], maxfev=10000)

    params = {
        "y_grid": y_grid,
        "y_valid": y_valid,
        "T_valid": T_valid,
        "P_valid": P_valid,
        "U_valid": U_valid,
        "rho_valid": rho_valid,
        "popt_P": popt_P,
        "best_T": best_T,
        "fits_T": fits_T_composite,
        "best_u": best_u,
        "fits_u": fits_u,
        "rho_0": rho_0,
        "popt_rho_low": popt_rho_low,
        "solver": solver
    }
    return params


def fit_by_params(y, params):
    """Compute the self-similar fit profiles (T, P, U, rho) on normalized coordinate y."""
    y = np.asarray(y, dtype=float)
    # Ensure y is clipped or handled appropriately (especially for boundary values)
    y_clipped = np.clip(y, 1e-12, 1.0)
    
    solver = params["solver"]
    popt_P = params["popt_P"]
    best_T = params["best_T"]
    best_u = params["best_u"]
    popt_rho_low = params.get("popt_rho_low", None)
    
    # 1) Pressure fit
    P_fit = 1.0 - (1.0 - solver.Ps) * y_clipped**popt_P[0]
    
    # 2) Velocity fit
    U_fit = best_u["func"](y_clipped, *best_u["popt"])
    
    # 3) Temperature and Density fits (piecewise or global depending on FITTING_OPTION)
    T_fit = np.zeros_like(y_clipped)
    rho_fit = np.zeros_like(y_clipped)
    
    if FITTING_OPTION == "FIT_RHO_ALL_AROUND":
        rho_fit = best_T["func"](y_clipped, *best_T["popt"])
        T_fit = (P_fit / (6730.0 * solver.r * rho_fit**0.86))**(1.0/1.6)
    else:
        low_mask = y_clipped < Y_FIT_MIN
        high_mask = ~low_mask
        
        # Low domain (y < Y_FIT_MIN): Density fit, Temp derived from EOS
        rho_s = 1.0 / solver.Rs_or_Vs
        rho_0 = params["rho_0"]
        def fit_rho_around_zero(y_val, a, d):
            return rho_s + (rho_0 - rho_s) * (1.0 - y_val**d)**a
            
        if np.any(low_mask) and popt_rho_low is not None:
            rho_fit[low_mask] = fit_rho_around_zero(y_clipped[low_mask], *popt_rho_low)
            T_fit[low_mask] = (P_fit[low_mask] / (6730.0 * solver.r * rho_fit[low_mask]**0.86))**(1.0/1.6)
            
        # High domain (y >= Y_FIT_MIN)
        if np.any(high_mask):
            fit_val_high = best_T["func"](y_clipped[high_mask], *best_T["popt"])
            if FITTING_OPTION == "FIT_TEMP_AROUND_PISTON":
                T_fit[high_mask] = fit_val_high
                rho_fit[high_mask] = (P_fit[high_mask] / (6730.0 * solver.r * T_fit[high_mask]**1.6))**(1.0/0.86)
            else: # FIT_RHO_AROUND_PISTON
                rho_fit[high_mask] = fit_val_high
                T_fit[high_mask] = (P_fit[high_mask] / (6730.0 * solver.r * rho_fit[high_mask]**0.86))**(1.0/1.6)
                
    return T_fit, P_fit, U_fit, rho_fit


def calculate_dimensional_fits(mass_grid, t_actual, solver, params):
    """Map shock self-similar fit arrays to dimensional (CGS) physical profiles on mass_grid at time t_actual.
    
    Uses the same analytical scaling factors as PistonShock.solve():
        v = V * (v0*v0*(p0*t**(tau+2.))**omega)**(1./(2.-omega))
        u = U * (v0*p0*t**(omega+tau))**(1./(2.-omega))
        p = P * p0 * t**tau
    Temperature is then derived from the CGS EOS.
    """
    m_s = solver.shocked_mass(time=t_actual)
    y = mass_grid / m_s
    T_fit_ss, P_fit_ss, U_fit_ss, rho_fit_ss = fit_by_params(y, params)

    # Analytical scaling factors (identical to PistonShock.solve() lines 258-260)
    v0, p0, tau, omega = solver.v0, solver.p0, solver.tau, solver.omega
    exp = 1.0 / (2.0 - omega)

    v_scale = (v0 * v0 * (p0 * t_actual**(tau + 2.0))**omega) ** exp
    u_scale = (v0 * p0 * t_actual**(omega + tau)) ** exp
    p_scale = p0 * t_actual**tau

    # Dimensionalize: V_fit = 1/rho_fit (specific volume), then scale to CGS
    v_fit_ss = np.where(rho_fit_ss > 0, 1.0 / rho_fit_ss, np.nan)
    rho = np.where(v_fit_ss > 0, 1.0 / (v_fit_ss * v_scale), np.nan)
    u = U_fit_ss * u_scale
    p = P_fit_ss * p_scale
    T = dimensional_temperature_from_eos(p, 1.0 / rho, r=solver.r)

    # Fill unshocked region (y >= 1)
    outside = mass_grid >= m_s
    rho[outside] = solver.rho0
    p[outside]   = 1e-6
    u[outside]   = 0.0
    T[outside]   = 300.0

    return {"density": rho, "pressure": p, "velocity": u, "temperature": T}


def plot_dimensional_fit_comparison(history, solver, case, params, material_hydro_path, case_title):
    """Plot overlay profiles of T, rho, P, u vs m comparing Simulation, Exact Solver, and Analytic fits."""
    print(f"Generating physical shock profiles comparison for {case_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_max = max(history.t)
    target_times = [0.5 * t_max, 0.75 * t_max, t_max]
    sim_colors = ["royalblue", "darkorange", "crimson"]
    
    ax_rho = axes[0, 0]
    ax_p = axes[0, 1]
    ax_u = axes[1, 0]
    ax_T = axes[1, 1]
    
    # Add Zoomed Inset for Temperature near piston (m ~ 0)
    axins_T = ax_T.inset_axes([0.45, 0.45, 0.48, 0.48])
    
    p_scale = 1e12
    u_scale = 1e5
    T_scale = 1.160451812e6
    
    for i, (t_target, sim_color) in enumerate(zip(target_times, sim_colors)):
        # 1) Simulation
        idx_sim = np.argmin(np.abs(np.array(history.t) - t_target))
        m_sim = history.m[idx_sim]
        t_actual = history.t[idx_sim]

        # shock front for shock solver
        m_s_exact = solver.shocked_mass(time=t_actual)

        shock_mask = m_sim <= m_s_exact
        m_sim_shock = m_sim[shock_mask][:-2]

        sim_rho = history.rho[idx_sim][shock_mask][:-2]
        sim_p = history.p[idx_sim][shock_mask] [:-2]
        sim_u = history.u[idx_sim][shock_mask] [:-2]
        sim_T = history.T[idx_sim][shock_mask][:-2]
            
        # 2) Exact Solver # TODO extract to some N
        mass_solver = np.linspace(1e-12, m_s_exact, 1000)
        sol_exact = solver.solve(mass=mass_solver, time=t_actual)
        exact_rho = sol_exact["density"]
        exact_p = sol_exact["pressure"]
        exact_u = sol_exact["velocity"]
        exact_T = dimensional_temperature_from_eos(exact_p, 1./exact_rho)

        # 3) Analytical fits mapped to CGS
        fits = calculate_dimensional_fits(mass_solver, t_actual, solver, params)
        fit_rho = fits["density"]
        fit_p = fits["pressure"]
        fit_u = fits["velocity"]
        fit_T = fits["temperature"]
        
        show_label = i == 0
        
        # Plot Density
        ax_rho.plot(m_sim_shock * 1e3, sim_rho, '-', color=sim_color, markersize=3, alpha=0.7, label=f"Simulation ({t_target*1e9:.3f} ns)" if show_label else None)
        ax_rho.plot(mass_solver * 1e3, exact_rho, '--', color='black', lw=2.0, label="Exact Solver" if show_label else None)
        ax_rho.plot(mass_solver * 1e3, fit_rho, '.', color='forestgreen', lw=0.5, alpha=0.5, label="Analytic Fit" if show_label else None)
        
        # Plot Pressure
        ax_p.plot(m_sim_shock * 1e3, sim_p / p_scale, '-', color=sim_color, markersize=3, alpha=0.7)
        ax_p.plot(mass_solver * 1e3, exact_p / p_scale, '--', color='black', lw=2.0)
        ax_p.plot((mass_solver) * 1e3, fit_p / p_scale, '.', color='forestgreen', lw=0.5, alpha=0.5)
        
        # Plot Velocity
        ax_u.plot(m_sim_shock * 1e3, sim_u / u_scale, '-', color=sim_color, markersize=3, alpha=0.7)
        ax_u.plot(mass_solver * 1e3, exact_u / u_scale, '--', color='black', lw=2.0)
        ax_u.plot(mass_solver * 1e3, fit_u / u_scale, '.', color='forestgreen', lw=0.5, alpha=0.5)
        
        # Plot Temperature
        ax_T.plot(m_sim_shock * 1e3, sim_T / T_scale, '-', color=sim_color, markersize=3, alpha=0.7)
        ax_T.plot(mass_solver * 1e3, exact_T / T_scale, '--', color='black', lw=2.0)
        ax_T.plot(mass_solver * 1e3, fit_T / T_scale, '.', color='forestgreen', lw=0.5, alpha=0.5)
        
        # Temperature inset near origin
        axins_T.plot(m_sim_shock * 1e3, sim_T / T_scale, '-', color=sim_color, markersize=2, alpha=0.7)
        axins_T.plot(mass_solver * 1e3, exact_T / T_scale, '--', color='black', lw=1.5)
        axins_T.plot(mass_solver * 1e3, fit_T / T_scale, '.', color='forestgreen', markersize=1, alpha=0.5)
        
    # Build time legend entries using plasma colors
    time_handles = [
        Line2D([0], [0], color=sim_colors[k], lw=2, label=f"{target_times[k]*1e9:.3f} ns")
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
                Line2D([0], [0], color='forestgreen', lw=2, linestyle='--', label='Analytic Fit'),
            ]
            ax.legend(handles=time_handles + style_handles, loc="upper left")
            
    # Style temperature inset — zoom to first 20% of the mass range at the latest time
    last_m_s = solver.xsi_s / solver.xsi_over_m(time=target_times[-1])
    zoom_m_max = 0.25 * last_m_s * 1e3  # 25% of smallest shock-front mass, in mg/cm^2
    axins_T.set_xlim(0, zoom_m_max)
    axins_T.set_ylim(0, 1.15)
    axins_T.grid(True, alpha=0.3)
    axins_T.set_title("Zoom near piston", fontsize=9)
    axins_T.tick_params(labelsize=8)
    ax_T.indicate_inset_zoom(axins_T, edgecolor="black")
            
    plt.suptitle(f"Shock Region Verification\n{case_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(material_hydro_path, dpi=200)
    print(f"Saved dimensional shock profiles comparison to {material_hydro_path}")
    plt.close(fig)


def plot_and_fit_self_similar(solver, params, self_similar_path, case_title):
    print("Generating shock 2x2 self-similar fitting plots...")
    y_valid = params["y_valid"]
    T_valid = params["T_valid"]
    P_valid = params["P_valid"]
    U_valid = params["U_valid"]
    rho_valid = params["rho_valid"]
    popt_P = params["popt_P"]
    best_T = params["best_T"]
    best_u = params["best_u"]
    
    # Compute chosen fit arrays via unified fit_by_params
    T_fit, P_fit, U_fit, rho_fit = fit_by_params(y_valid, params)
    
    err_T = np.abs((T_fit - T_valid) / T_valid)
    err_rho = np.abs((rho_fit - rho_valid) / rho_valid)
    err_P = np.abs((P_fit - P_valid) / P_valid)
    err_U = np.abs((U_fit - U_valid) / (U_valid + 1e-15))
    
    # Restrict Temperature and Density error computation only to the Y_FIT_MIN <= y <= 1.0 domain
    low_mask = y_valid < Y_FIT_MIN
    high_mask = y_valid >= Y_FIT_MIN
    
    err_T_high = err_T[high_mask]
    avg_T_high, max_T_high = np.mean(err_T_high), np.max(err_T_high)
    err_rho_high = err_rho[high_mask]
    avg_rho_high, max_rho_high = np.mean(err_rho_high), np.max(err_rho_high)
    
    err_T_low = err_T[low_mask] if np.any(low_mask) else [0.0]
    avg_T_low, max_T_low = np.mean(err_T_low), np.max(err_T_low)
    err_rho_low = err_rho[low_mask] if np.any(low_mask) else [0.0]
    avg_rho_low, max_rho_low = np.mean(err_rho_low), np.max(err_rho_low)
    
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = np.mean(err_U), np.max(err_U)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # Panel (0,0): Temperature
    ax = axes[0, 0]
    ax.plot(y_valid, T_valid, 'b-', label='Numerical Solver', lw=2)
    if np.any(low_mask):
        ax.plot(y_valid[low_mask], T_fit[low_mask], '--', color='darkorange', label=f'EOS Derived (y < {Y_FIT_MIN:.1f})', lw=1.5)
    ax.plot(y_valid[high_mask], T_fit[high_mask], 'r--', label=f'Analytical Fit (y >= {Y_FIT_MIN:.1f})', lw=1.5)
    ax.set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    ax.set_title(f"Shock: Temperature ({best_T['name']})", fontsize=13, fontweight='bold')
    lbl_T = f"${best_T['latex']}$\n(y >= {Y_FIT_MIN:.1f}) Avg: {avg_T_high:.4e}, Max: {max_T_high:.4e}\n(y < {Y_FIT_MIN:.1f}) Avg: {avg_T_low:.4e}, Max: {max_T_low:.4e}"
    ax.text(0.05, 0.05, lbl_T, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Close-up inset around Y_FIT_MIN
    ax_inset = ax.inset_axes([0.45, 0.45, 0.48, 0.48])
    ax_inset.plot(y_valid, T_valid, 'b-', lw=1.5)
    if np.any(low_mask):
        ax_inset.plot(y_valid[low_mask], T_fit[low_mask], '--', color='darkorange', lw=1.2)
    ax_inset.plot(y_valid[high_mask], T_fit[high_mask], 'r--', lw=1.2)
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
    if np.any(low_mask):
        ax.plot(y_valid[low_mask], rho_fit[low_mask], '--', color='darkorange', label=f'Power Law Fit (y < {Y_FIT_MIN:.1f})', lw=1.5)
    ax.plot(y_valid[high_mask], rho_fit[high_mask], 'r--', label=f'EOS Derived (y >= {Y_FIT_MIN:.1f})', lw=1.5)
    ax.set_ylabel(r"Density $\rho(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Shock: Density", fontsize=13, fontweight='bold')
    lbl_R = r"$\rho(y)$" + f"\n(y >= {Y_FIT_MIN:.1f}) Avg: {avg_rho_high:.4e}, Max: {max_rho_high:.4e}\n(y < {Y_FIT_MIN:.1f}) Avg: {avg_rho_low:.4e}, Max: {max_rho_low:.4e}"
    ax.text(0.05, 0.05, lbl_R, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'), transform=ax.transAxes, fontsize=9.5)
    
    # Panel (1,0): Pressure
    ax = axes[1, 0]
    ax.plot(y_valid, P_valid, 'b-', label='Numerical Solver', lw=2)
    ax.plot(y_valid, P_fit, 'r--', label='Analytical Fit', lw=1.5)
    ax.set_ylabel(r"Pressure $P(y)$ [dimensionless]", fontsize=12)
    ax.set_title("Shock: Pressure", fontsize=13, fontweight='bold')
    lbl_P = f"$P(y) \\approx 1.0 - (1.0 - P_s) y^{{{popt_P[0]:.5f}}}$\nAvg Err: {avg_P:.4e}, Max Err: {max_P:.4e}"
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
        
    lbl_U = u_formula + f"\nAvg Err: {avg_U:.4e}\nMax Err: {max_U:.4e}"
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


def plot_standalone_velocity_fits(params, standalone_path, case_title):
    print("Generating standalone shock velocity fitting comparison plots...")
    y_valid = params["y_valid"]
    U_valid = params["U_valid"]
    fits_u = params["fits_u"]
    best_u = params["best_u"]
    
    fig_sa, (ax_sa1, ax_sa2) = plt.subplots(1, 2, figsize=(18, 8.5))
    ax_sa1.plot(y_valid, U_valid, 'b-', label='Numerical Solver', lw=3.0)
    colors_u = {1: 'crimson', 2: 'darkorange', 3: 'forestgreen', 4: 'darkviolet', 5: 'deeppink', 6: 'teal'}
    
    for i in range(1, 7):
        popt, U_fit_cand, avg_err, max_err, name, latex = fits_u[i]
        if popt is not None:
            lbl = f"Fit {i}: {name}\nAvg Err: {avg_err:.3e}, Max Err: {max_err:.3e}"
            lw = 2.2 if i == best_u["id"] else 1.5
            ax_sa1.plot(y_valid, U_fit_cand, colors_u[i], linestyle='--', label=lbl, lw=lw)
            
            err_curve = np.abs((U_fit_cand - U_valid) / (U_valid + 1e-15))
            ax_sa2.plot(y_valid, err_curve, colors_u[i], label=f"Fit {i} (Avg: {avg_err:.3e})", lw=lw)
            
    ax_sa1.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_sa1.set_ylabel(r"Velocity $U(y)$ [dimensionless]", fontsize=12)
    ax_sa1.legend(loc='best', fontsize=9.0)
    ax_sa1.grid(True, alpha=0.3)
    ax_sa1.set_title("Dimensionless Velocity $U(y)$ vs 6 Candidates", fontsize=13, fontweight='bold')
    
    ax_sa2.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_sa2.set_ylabel(r"Relative Error", fontsize=12)
    ax_sa2.set_yscale('log')
    ax_sa2.set_ylim(bottom=1e-10)
    ax_sa2.legend(loc='best', fontsize=9.5)
    ax_sa2.grid(True, which="both", ls=":", alpha=0.5)
    ax_sa2.set_title("Relative Errors of Velocity Fits (semi-log)", fontsize=13, fontweight='bold')
    
    fig_sa.suptitle(f"Shock Velocity Profile Curve Fitting & Optimization\nChosen Formal Fit: Fit {best_u['id']} ({best_u['name']})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_sa.savefig(standalone_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sa)
    print(f"Saved standalone shock velocity fits to {standalone_path}")


def plot_standalone_temperature_fits(params, standalone_T_path, case_title):
    print("Generating standalone temperature fitting comparison plots...")
    y_valid = params["y_valid"]
    T_valid = params["T_valid"]
    fits_T = params["fits_T"]
    best_T = params["best_T"]
    
    fig_sa, (ax_sa1, ax_sa2) = plt.subplots(2, 1, figsize=(11, 10))
    
    ax_sa1.plot(y_valid, T_valid, 'b-', label='Numerical Solver', lw=3.0)
    
    colors_T = {1: 'crimson', 2: 'darkorange', 3: 'forestgreen', 4: 'darkviolet', 5: 'deeppink'}
    
    # We only plot relative errors for the fitting domain y >= Y_FIT_MIN
    low_mask = y_valid < Y_FIT_MIN
    high_mask = y_valid >= Y_FIT_MIN
    
    for i in range(1, 6):
        popt, T_fit_cand, rho_fit_cand, avg_err, max_err, name, latex = fits_T[i]
        if popt is not None:
            lbl = f"Fit {i}: {name}\nAvg Err ({Y_FIT_MIN:.1f}<=y<1): {avg_err:.3e}, Max: {max_err:.3e}"
            lw = 2.2 if i == best_T["id"] else 1.5
            
            # Plot Temperature piecewise
            if np.any(low_mask):
                ax_sa1.plot(y_valid[low_mask], T_fit_cand[low_mask], colors_T[i], linestyle=':', lw=lw*0.7)
            ax_sa1.plot(y_valid[high_mask], T_fit_cand[high_mask], colors_T[i], linestyle='--', label=lbl, lw=lw)
            
            # Error only plotted inside fitting domain (y >= Y_FIT_MIN)
            T_fit_cand_domain = T_fit_cand[high_mask]
            T_valid_domain = T_valid[high_mask]
            err_curve = np.abs((T_fit_cand_domain - T_valid_domain) / T_valid_domain)
            ax_sa2.plot(y_valid[high_mask], err_curve, colors_T[i], label=f"Fit {i} (Avg: {avg_err:.3e})", lw=lw)
            
    ax_sa1.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_sa1.set_ylabel(r"Temperature $T(y)$ [dimensionless]", fontsize=12)
    ax_sa1.legend(loc='best', fontsize=9.0)
    ax_sa1.grid(True, alpha=0.3)
    ax_sa1.set_title("Dimensionless Temperature $T(y)$ vs 5 Candidates", fontsize=13, fontweight='bold')
    
    ax_sa2.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_sa2.set_ylabel(r"Relative Error", fontsize=12)
    ax_sa2.set_yscale('log')
    ax_sa2.set_ylim(bottom=1e-10)
    ax_sa2.legend(loc='best', fontsize=9.5)
    ax_sa2.grid(True, which="both", ls=":", alpha=0.5)
    ax_sa2.set_title(f"Relative Errors of Temperature Fits on domain y >= {Y_FIT_MIN:.1f} (semi-log)", fontsize=13, fontweight='bold')
    
    fig_sa.suptitle(f"Shock Temperature Curve Fitting & Optimization\nChosen Formal Fit: Fit {best_T['id']} ({best_T['name']})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_sa.savefig(standalone_T_path, dpi=200, bbox_inches='tight')
    plt.close(fig_sa)
    print(f"Saved standalone shock temperature fits to {standalone_T_path}")


def plot_relative_errors(solver, params, relative_errors_path, case_title):
    print("Generating shock relative error plots...")
    y_valid = params["y_valid"]
    T_valid = params["T_valid"]
    P_valid = params["P_valid"]
    U_valid = params["U_valid"]
    rho_valid = params["rho_valid"]
    best_T = params["best_T"]
    best_u = params["best_u"]
    
    # Compute chosen fit arrays via unified fit_by_params
    T_fit, P_fit, U_fit, rho_fit = fit_by_params(y_valid, params)
    
    err_T = np.abs((T_fit - T_valid) / T_valid)
    err_rho = np.abs((rho_fit - rho_valid) / rho_valid)
    err_P = np.abs((P_fit - P_valid) / P_valid)
    err_U = np.abs((U_fit - U_valid) / (U_valid + 1e-15))
    
    low_mask = y_valid < Y_FIT_MIN
    high_mask = y_valid >= Y_FIT_MIN
    
    err_T_high = err_T[high_mask]
    avg_T_high, max_T_high = np.mean(err_T_high), np.max(err_T_high)
    err_rho_high = err_rho[high_mask]
    avg_rho_high, max_rho_high = np.mean(err_rho_high), np.max(err_rho_high)
    
    err_T_low = err_T[low_mask] if np.any(low_mask) else [0.0]
    avg_T_low, max_T_low = np.mean(err_T_low), np.max(err_T_low)
    err_rho_low = err_rho[low_mask] if np.any(low_mask) else [0.0]
    avg_rho_low, max_rho_low = np.mean(err_rho_low), np.max(err_rho_low)
    
    avg_P, max_P = np.mean(err_P), np.max(err_P)
    avg_U, max_U = best_u["avg_err"], best_u["max_err"]
    
    fig_err, ax_err = plt.subplots(figsize=(10, 7.5))
    
    # Plot Pressure and Velocity over full domain
    ax_err.plot(y_valid, err_P, color='red', label=f'Pressure $P(y)$ (Avg: {avg_P:.3e}, Max: {max_P:.3e})', lw=2.0)
    ax_err.plot(y_valid, err_U, color='purple', label=f'Velocity $U(y)$ (Avg: {avg_U:.3e}, Max: {max_U:.3e})', lw=2.0)
    
    # Plot Temperature piecewise
    ax_err.plot(y_valid[high_mask], err_T_high, color='blue', label=f'Temperature $T(y)$ (y >= {Y_FIT_MIN:.1f}, Avg: {avg_T_high:.3e}, Max: {max_T_high:.3e})', lw=2.0)
    if np.any(low_mask):
        ax_err.plot(y_valid[low_mask], err_T_low, color='darkorange', linestyle=':', label=f'Temperature $T(y)$ (y < {Y_FIT_MIN:.1f}, Avg: {avg_T_low:.3e}, Max: {max_T_low:.3e})', lw=2.0)
        
    # Plot Density piecewise
    ax_err.plot(y_valid[high_mask], err_rho_high, color='green', label=f'Density $\\rho(y)$ (y >= {Y_FIT_MIN:.1f}, Avg: {avg_rho_high:.3e}, Max: {max_rho_high:.3e})', lw=2.0)
    if np.any(low_mask):
        ax_err.plot(y_valid[low_mask], err_rho_low, color='orange', linestyle=':', label=f'Density $\\rho(y)$ (y < {Y_FIT_MIN:.1f}, Avg: {avg_rho_low:.3e}, Max: {max_rho_low:.3e})', lw=2.0)
    
    ax_err.set_xlabel(r"Normalized coordinate $y = \xi / \xi_s$", fontsize=12)
    ax_err.set_ylabel(r"Relative Error", fontsize=12)
    ax_err.set_yscale('log')
    ax_err.set_ylim(bottom=1e-5)
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
    from data_loader import get_sim_history, get_shock_similarity_solver
    case, history = get_sim_history(preset_name, case_label)
    solver = get_shock_similarity_solver(case, case_label)
    return case, history, solver


def get_plot_paths(case_label: str) -> dict[str, str]:
    out_dir = Path("results/ictt") / case_label
    ss_dir = out_dir / "self_similar_shock"
    dv_dir = out_dir / "dimensional_verification"
    ss_dir.mkdir(parents=True, exist_ok=True)
    dv_dir.mkdir(parents=True, exist_ok=True)

    return {
        "self_similar": str(ss_dir / f"{case_label}_self_similar.png"),
        "velocity_fits": str(ss_dir / f"{case_label}_velocity_fits_standalone.png"),
        "temperature_fits": str(ss_dir / f"{case_label}_temperature_fits_standalone.png"),
        "relative_errors": str(ss_dir / f"{case_label}_relative_errors.png"),
        "dimensional_comparison": str(dv_dir / f"{case_label}_dimensional_fit_comparison_shock.png")
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

    # 1) Standalone Temperature fits comparison and relative errors
    plot_standalone_temperature_fits(params, paths["temperature_fits"], case_title)

    # 2) Standalone Velocity fits comparison
    plot_standalone_velocity_fits(params, paths["velocity_fits"], case_title)

    # 3) Self-similar profiles and fits (using optimal models)
    plot_and_fit_self_similar(solver, params, paths["self_similar"], case_title)

    # 4) Dimensional fit comparison
    plot_dimensional_fit_comparison(history, solver, case, params, paths["dimensional_comparison"], case_title)

    # 5) Relative errors of self-similar fits
    plot_relative_errors(solver, params, paths["relative_errors"], case_title)


def run_preset_workflow(preset_name: str, case_label: str, case_title: str):
    """Run full verification comparison pipeline for a given preset."""
    print("=" * 80)
    print(f"PROCESSING PRESET: {preset_name} -> {case_label}")
    print("=" * 80)

    case, history, solver = run_simulation_and_references(preset_name, case_label)
    params = perform_shock_fitting(solver)

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
    print("\nAll custom simulations, PistonShock comparisons, plotting, fitting, and exports completed successfully!")


if __name__ == "__main__":
    main()

import sys
import os
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_MENAHEM_DIR = Path(__file__).resolve().parents[1] / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE
from project3_code.rad_hydro_sim.verification.menahem_comparison import _shock_kwargs_from_case
from piston_shock import PistonShock

print("Solving Shock self-similar profiles...")
case, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
kwargs = _shock_kwargs_from_case(case)
kwargs["omega"] = float(getattr(case, "omega", 0.0))
solver = PistonShock(**kwargs)

# Grid of y_s = xsi / xsi_s in [0, 1]
y_grid = np.linspace(0.0, 1.0, 500)
xsi_vec = y_grid * solver.xsi_s
xsi_vec[0] = 1e-10

V_val, U_val, P_val = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
R_val = 1.0 / V_val
T_val = P_val * V_val / solver.r

valid_idx = (y_grid > 0.005) & np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(T_val)
y_valid = y_grid[valid_idx]
U_valid = U_val[valid_idx]

U_0 = U_valid[0]
U_s = U_valid[-1]

# Shock velocity fits
def fit_u_1(y, d): return U_0 - (U_0 - U_s) * (y**d)
def fit_u_2(y, c, d): return c - (c - U_s) * (y**d)
def fit_u_3(y, c, a, b): return c - a * (y**b)
def fit_u_4(y, c, d): return c * (1.0 - y) / (1.0 + d * y) + U_s * y
def fit_u_5(y, c, a, b): return c * (1.0 - y**a) * (y**(-b)) + U_s * y
def fit_u_6(y, c, a, b): return c * (1.0 - y**a)**b + U_s * y

candidates = [
    {"id": 1, "func": fit_u_1, "name": "1P Power Law: U0 - (U0 - Us)*y^d", "p0": [1.0]},
    {"id": 2, "func": fit_u_2, "name": "2P Power Law: c - (c - Us)*y^d", "p0": [U_0, 1.0]},
    {"id": 3, "func": fit_u_3, "name": "3P Power Law: c - a*y^b", "p0": [U_0, U_0 - U_s, 1.0]},
    {"id": 4, "func": fit_u_4, "name": "2P Rational: c*(1-y)/(1+d*y) + Us*y", "p0": [U_0, 1.0]},
    {"id": 5, "func": fit_u_5, "name": "3P Singular: c*(1-y^a)*y^-b + Us*y", "p0": [U_0, 1.0, 0.3]},
    {"id": 6, "func": fit_u_6, "name": "3P Gen Power: c*(1-y^a)^b + Us*y", "p0": [U_0, 1.0, 1.0]}
]

print("\n--- Shock Velocity Fitting Comparison ---")
best_fit = None
min_avg_err = float("inf")

for cand in candidates:
    try:
        popt, _ = curve_fit(cand["func"], y_valid, U_valid, p0=cand["p0"], maxfev=10000)
        U_fit = cand["func"](y_valid, *popt)
        rel_err = np.abs((U_fit - U_valid) / (U_valid + 1e-15)) * 100
        avg_err = np.mean(rel_err)
        max_err = np.max(rel_err)
        print(f"Fit {cand['id']} ({cand['name']}): popt={popt}, Avg Err={avg_err:.4f}%, Max Err={max_err:.4f}%")
        
        if avg_err < min_avg_err:
            min_avg_err = avg_err
            best_fit = {"cand": cand, "popt": popt, "avg_err": avg_err, "max_err": max_err}
    except Exception as e:
        print(f"Fit {cand['id']} failed: {e}")

if best_fit:
    print(f"\nOptimal Velocity Fit: Fit {best_fit['cand']['id']} ({best_fit['cand']['name']})")
    print(f"popt: {best_fit['popt']}, Avg Err: {best_fit['avg_err']:.4f}%, Max Err: {best_fit['max_err']:.4f}%")

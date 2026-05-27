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
from project3_code.rad_hydro_sim.problems.presets_config import PRESET_FIG_8_CONSTANT_TEMPERATURE
from project3_code.rad_hydro_sim.verification.menahem_comparison import _heat_kwargs_from_case
from subsonic_heat_wave import SubsonicHeatWave

case, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
heat_kwargs = _heat_kwargs_from_case(case)
solver = SubsonicHeatWave(**heat_kwargs).find_xsi_f()

y_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
xsi_vec = y_grid * solver.xsi_f
profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)

V_val = profiles["V"]
U_val = profiles["U"]
P_val = profiles["P"]
T_val = profiles["T"]

valid_idx = (y_grid > 0.005) & (y_grid < 0.99) & np.isfinite(V_val) & np.isfinite(U_val) & np.isfinite(P_val) & np.isfinite(T_val)
y_valid = y_grid[valid_idx]
U_valid = U_val[valid_idx]

# Subsonic Velocity fits
def fit_u_1(y, c, d): return c * (1.0 - y)**d
def fit_u_2(y, c, b): return c * (1.0 - y) / (1.0 + b * y)
def fit_u_3(y, c, b): return c * (1.0 - y) * (y**(-b))
def fit_u_4(y, c, a, b): return c * (1.0 - y**a) * (y**(-b))
def fit_u_5(y, c, a, b): return c * (1.0 - y) * (y**a + y**(-b))
def fit_u_6(y, c, a, b): return c * (1.0 - y**a) / (1.0 + b * y)

candidates = [
    {"id": 1, "func": fit_u_1, "name": "Power Law: c*(1-y)^d", "p0": [U_valid[0], 0.5]},
    {"id": 2, "func": fit_u_2, "name": "Rational: c*(1-y)/(1+b*y)", "p0": [U_valid[0], 0.5]},
    {"id": 3, "func": fit_u_3, "name": "Singular 2P: c*(1-y)*y^-b", "p0": [-1.0, 0.3]},
    {"id": 4, "func": fit_u_4, "name": "Singular 3P: c*(1-y^a)*y^-b", "p0": [-1.0, 1.0, 0.3]},
    {"id": 5, "func": fit_u_5, "name": "User 3P: c*(1-y)*(y^a + y^-b)", "p0": [-1.0, 1.0, 0.3]},
    {"id": 6, "func": fit_u_6, "name": "Rational Frac: c*(1-y^a)/(1+b*y)", "p0": [U_valid[0], 1.0, 1.0]}
]

print("\n--- Subsonic Velocity Fitting Comparison (y < 0.99) ---")
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

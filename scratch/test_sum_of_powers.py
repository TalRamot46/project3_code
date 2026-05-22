import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

_REPO_ROOT = Path(r"c:\Users\TLP-001\Documents\GitHub\project3_code")
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

_MENAHEM_DIR = _REPO_ROOT / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import PRESET_FIG_8_CONSTANT_TEMPERATURE
from project3_code.rad_hydro_sim.verification.menahem_comparison import _heat_kwargs_from_case
# pyrefly: ignore [missing-import]
from subsonic_heat_wave_fixed import SubsonicHeatWave

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + 1e-30)

def main():
    case, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    heat_kwargs = _heat_kwargs_from_case(case)
    solver = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
    
    y_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
    xsi_vec = y_grid * solver.xsi_f
    profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    U_val = profiles["U"]
    
    u0 = U_val[0]
    print(f"U(0) = {u0:.6f}")
    
    # 1. Sum of powers with fixed U(0)
    def fit_double_fixed(y, c, a, b):
        return c * (1.0 - y**a) + (u0 - c) * (1.0 - y)**b
        
    popt, _ = curve_fit(fit_double_fixed, y_grid, U_val, p0=[u0 * 0.5, 0.5, 5.0], maxfev=20000)
    y_pred = fit_double_fixed(y_grid, *popt)
    r2 = compute_r2(U_val, y_pred)
    print(f"Fixed-U(0) Sum-of-powers | R^2 = {r2:.6f} | Params: c={popt[0]:.4f}, a={popt[1]:.4f}, b={popt[2]:.4f}")
    
    # 2. Sum of powers with free U(0)
    def fit_double_free(y, u0_fit, c, a, b):
        return c * (1.0 - y**a) + (u0_fit - c) * (1.0 - y)**b
        
    popt_free, _ = curve_fit(fit_double_free, y_grid, U_val, p0=[u0, u0 * 0.5, 0.5, 5.0], maxfev=20000)
    y_pred_free = fit_double_free(y_grid, *popt_free)
    r2_free = compute_r2(U_val, y_pred_free)
    print(f"Free-U(0) Sum-of-powers  | R^2 = {r2_free:.6f} | Params: U0={popt_free[0]:.4f}, c={popt_free[1]:.4f}, a={popt_free[2]:.4f}, b={popt_free[3]:.4f}")

    # 3. Rational Shift model: c * (1 - y^a) / (y^b + d)
    def fit_rational_shift(y, c, a, b, d):
        return c * (1.0 - y**a) / (y**b + d)
        
    popt_rat, _ = curve_fit(fit_rational_shift, y_grid, U_val, p0=[u0, 0.5, 0.5, 0.1], maxfev=20000)
    y_pred_rat = fit_rational_shift(y_grid, *popt_rat)
    r2_rat = compute_r2(U_val, y_pred_rat)
    print(f"Rational Shift model     | R^2 = {r2_rat:.6f} | Params: c={popt_rat[0]:.4f}, a={popt_rat[1]:.4f}, b={popt_rat[2]:.4f}, d={popt_rat[3]:.4f}")

if __name__ == "__main__":
    main()

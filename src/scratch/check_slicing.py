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
    
    imin = profiles["imin"]
    print(f"imin = {imin} out of 500")
    
    y_valid = y_grid[imin:]
    U_valid = profiles["U"][imin:]
    
    # Let's fit on the valid region
    def fit_u_1(y, c, d):
        return c * (1.0 - y)**d
    def fit_u_2(y, c, b):
        return c * (1.0 - y) / (1.0 + b * y)
    def fit_u_3(y, c, a):
        return c * (1.0 - y**a)
    def fit_u_4(y, c, a, b):
        return c * (1.0 - y**a) / (1.0 + b * y)
    def fit_u_5(y, c, a, b):
        return c * (1.0 - y**a)**b
        
    popt_U1, _ = curve_fit(fit_u_1, y_valid, U_valid, p0=[U_valid[0], 0.5])
    popt_U2, _ = curve_fit(fit_u_2, y_valid, U_valid, p0=[U_valid[0], 1.0])
    popt_U3, _ = curve_fit(fit_u_3, y_valid, U_valid, p0=[U_valid[0], 1.0])
    popt_U4, _ = curve_fit(fit_u_4, y_valid, U_valid, p0=[U_valid[0], 1.0, 1.0])
    popt_U5, _ = curve_fit(fit_u_5, y_valid, U_valid, p0=[U_valid[0], 1.0, 1.0])
    
    r2_U1 = compute_r2(U_valid, fit_u_1(y_valid, *popt_U1))
    r2_U2 = compute_r2(U_valid, fit_u_2(y_valid, *popt_U2))
    r2_U3 = compute_r2(U_valid, fit_u_3(y_valid, *popt_U3))
    r2_U4 = compute_r2(U_valid, fit_u_4(y_valid, *popt_U4))
    r2_U5 = compute_r2(U_valid, fit_u_5(y_valid, *popt_U5))
    
    print("\n--- Valid Sliced R2 ---")
    print(f"Option 1: R2 = {r2_U1:.6f} | Params: {popt_U1}")
    print(f"Option 2: R2 = {r2_U2:.6f} | Params: {popt_U2}")
    print(f"Option 3: R2 = {r2_U3:.6f} | Params: {popt_U3}")
    print(f"Option 4: R2 = {r2_U4:.6f} | Params: {popt_U4}")
    print(f"Option 5: R2 = {r2_U5:.6f} | Params: {popt_U5}")

if __name__ == "__main__":
    main()

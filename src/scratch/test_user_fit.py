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
    
    # Exclude y=0 to avoid division by zero for negative powers
    y_fit = y_grid[1:]
    U_fit = U_val[1:]
    
    candidates = {}
    
    # A. User suggestion: c * (1 - y) * (y^a + y^-b)
    # Since U_fit is negative, we expect c to be negative.
    candidates["c*(1-y)*(y^a + y^-b)"] = {
        "func": lambda y, c, a, b: c * (1.0 - y) * (y**a + y**(-b)),
        "p0": [-1.0, 0.5, 0.5]
    }
    
    # B. User suggestion variation: c * (1 - y^a) * (y^b + y^-d)
    candidates["c*(1-y^a)*(y^b + y^-d)"] = {
        "func": lambda y, c, a, b, d: c * (1.0 - y**a) * (y**b + y**(-d)),
        "p0": [-1.0, 0.5, 0.5, 0.5]
    }
    
    # C. Rational user suggestion: c * (1 - y) / (y^a + y^-b)
    # Wait, c * (1 - y) / (y^a + y^-b) is 0 at y=1 and c * y^b / (y^(a+b) + 1) -> drops near y=0, which is clever!
    candidates["c*(1-y)/(y^a + y^-b)"] = {
        "func": lambda y, c, a, b: c * (1.0 - y) / (y**a + y**(-b)),
        "p0": [-1.0, 0.5, 0.5]
    }

    # D. Sum of powers with negative term: c * (1 - y^a) - d * y^-b - wait, doesn't go to 0 at y=1.
    # What about c * (1 - y) * (y^a - y^-b)?
    candidates["c*(1-y)*(y^a - y^-b)"] = {
        "func": lambda y, c, a, b: c * (1.0 - y) * (y**a - y**(-b)),
        "p0": [-1.0, 0.5, 0.5]
    }

    print("Testing user-inspired fits:")
    print("-" * 65)
    for name, info in candidates.items():
        try:
            popt, _ = curve_fit(info["func"], y_fit, U_fit, p0=info["p0"], maxfev=20000)
            y_pred = info["func"](y_fit, *popt)
            r2 = compute_r2(U_fit, y_pred)
            p_str = ", ".join([f"{p:.4f}" for p in popt])
            print(f"{name:<30} | R^2 = {r2:.6f} | Params: [{p_str}]")
        except Exception as e:
            print(f"{name:<30} | FAILED: {e}")

if __name__ == "__main__":
    main()

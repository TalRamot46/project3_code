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
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_9_CONSTANT_FLUX
)
from project3_code.rad_hydro_sim.verification.menahem_comparison import _heat_kwargs_from_case
from subsonic_heat_wave_fixed import SubsonicHeatWave

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + 1e-30)

def test_preset_sliced(preset_name, label):
    case, _ = get_preset(preset_name)
    heat_kwargs = _heat_kwargs_from_case(case)
    solver = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
    
    y_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
    xsi_vec = y_grid * solver.xsi_f
    profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    imin = profiles["imin"]
    y_valid = y_grid[imin:]
    U_valid = profiles["U"][imin:]
    
    u0 = U_valid[0]
    print(f"\n=== Sliced Fits for {label} (imin={imin}, U0={U_valid[0]:.4f} to U[-1]={U_valid[-1]:.4f}) ===")
    
    # 1. Baseline: c * (1-y)^d (2 params)
    def fit_u_1(y, c, d):
        return c * (1.0 - y)**d
    # 2. Rational simple: c * (1-y) / (1 + b*y) (2 params)
    def fit_u_2(y, c, b):
        return c * (1.0 - y) / (1.0 + b * y)
    # 3. Singular Power-Law: c * (1-y) * y^-b (2 params)
    def fit_u_singular_2p(y, c, b):
        return c * (1.0 - y) * (y**(-b))
    # 4. User suggestion: c * (1-y) * (y^a + y^-b) (3 params)
    def fit_user(y, c, a, b):
        return c * (1.0 - y) * (y**a + y**(-b))
    # 5. Singular Fractional: c * (1-y^a) * y^-b (3 params)
    def fit_u_singular_3p(y, c, a, b):
        return c * (1.0 - y**a) * (y**(-b))
    # 6. Rational singular: c * (1-y) / (y^a + y^-b) (3 params)
    def fit_user_rational(y, c, a, b):
        return c * (1.0 - y) / (y**a + y**(-b))

    funcs = {
        "c*(1-y)^d [Baseline]": (fit_u_1, [-5.0, 2.0]),
        "c*(1-y)/(1+b*y) [Rational]": (fit_u_2, [-5.0, 3.0]),
        "c*(1-y)*y^-b [Singular 2-Param]": (fit_u_singular_2p, [-2.0, 0.3]),
        "c*(1-y)*(y^a + y^-b) [User 3-Param]": (fit_user, [-1.0, 0.3, 0.3]),
        "c*(1-y^a)*y^-b [Singular 3-Param]": (fit_u_singular_3p, [-2.0, 1.0, 0.3]),
        "c*(1-y)/(y^a + y^-b) [Rational 3-Param]": (fit_user_rational, [-2.0, 0.3, 0.3]),
    }
    
    for name, (func, p0) in funcs.items():
        try:
            popt, _ = curve_fit(func, y_valid, U_valid, p0=p0, maxfev=50000)
            y_pred = func(y_valid, *popt)
            r2 = compute_r2(U_valid, y_pred)
            p_str = ", ".join([f"{p:.5f}" for p in popt])
            print(f"{name:<45} | R^2 = {r2:.7f} | Params: [{p_str}]")
        except Exception as e:
            print(f"{name:<45} | FAILED: {e}")

def main():
    test_preset_sliced(PRESET_FIG_8_CONSTANT_TEMPERATURE, "Fig 8")
    test_preset_sliced(PRESET_FIG_9_CONSTANT_FLUX, "Fig 9")

if __name__ == "__main__":
    main()

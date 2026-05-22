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
    print(f"\n=== Sliced Fits for {label} (imin={imin}, U0={u0:.4f}) ===")
    
    # 1. Baseline: c * (1-y)^d
    def fit_u_1(y, c, d):
        return c * (1.0 - y)**d
    # 2. Rational simple: c * (1-y) / (1 + b*y)
    def fit_u_2(y, c, b):
        return c * (1.0 - y) / (1.0 + b * y)
    # 3. Fractional-power: c * (1-y^a)
    def fit_u_3(y, c, a):
        return c * (1.0 - y**a)
    # 4. Rational fractional-power: c * (1-y^a) / (1 + b*y)
    def fit_u_4(y, c, a, b):
        return c * (1.0 - y**a) / (1.0 + b * y)
    # 5. Dual power-law: c * (1-y^a)^b
    def fit_u_5(y, c, a, b):
        return c * (1.0 - y**a)**b
    # 6. User suggestion: c * (1-y) * (y^a + y^-b)
    def fit_user(y, c, a, b):
        return c * (1.0 - y) * (y**a + y**(-b))
    # 7. Clever alternative: c * (1-y) * (1 + a*y^b)
    def fit_clever(y, c, a, b):
        return c * (1.0 - y) * (1.0 + a * y**b)

    funcs = {
        "c*(1-y)^d": (fit_u_1, [u0, 0.5]),
        "c*(1-y)/(1+b*y)": (fit_u_2, [u0, 1.0]),
        "c*(1-y^a)": (fit_u_3, [u0, 1.0]),
        "c*(1-y^a)/(1+b*y)": (fit_u_4, [u0, 1.0, 1.0]),
        "c*(1-y^a)^b": (fit_u_5, [u0, 1.0, 1.0]),
        "c*(1-y)*(y^a + y^-b)": (fit_user, [u0 * 0.5, 0.5, 0.5]),
        "c*(1-y)*(1 + a*y^b)": (fit_clever, [u0, 1.0, 1.0]),
    }
    
    for name, (func, p0) in funcs.items():
        try:
            # Provide sensible bounds for parameters to avoid overflow or domain errors
            popt, _ = curve_fit(func, y_valid, U_valid, p0=p0, maxfev=50000)
            y_pred = func(y_valid, *popt)
            r2 = compute_r2(U_valid, y_pred)
            p_str = ", ".join([f"{p:.4f}" for p in popt])
            print(f"{name:<25} | R^2 = {r2:.6f} | Params: [{p_str}]")
        except Exception as e:
            print(f"{name:<25} | FAILED: {e}")

def main():
    test_preset_sliced(PRESET_FIG_8_CONSTANT_TEMPERATURE, "Fig 8")
    test_preset_sliced(PRESET_FIG_9_CONSTANT_FLUX, "Fig 9")

if __name__ == "__main__":
    main()

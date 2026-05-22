import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

# Set up paths to import from repository
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

def main():
    # 1) Get the case for Fig 8
    case, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    heat_kwargs = _heat_kwargs_from_case(case)
    
    # 2) Solve subsonic heat wave similarity ODEs
    solver = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
    
    y_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
    xsi_vec = y_grid * solver.xsi_f
    profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    U_val = profiles["U"]
    
    # Define candidate fit formulas
    candidates = {}
    
    # 1. Baseline: c * (1-y)^d
    candidates["c*(1-y)^d"] = {
        "func": lambda y, c, d: c * (1.0 - y)**d,
        "p0": [U_val[0], 0.5]
    }
    
    # 2. Rational simple: c * (1-y) / (1 + b*y)
    candidates["c*(1-y)/(1+b*y)"] = {
        "func": lambda y, c, b: c * (1.0 - y) / (1.0 + b * y),
        "p0": [U_val[0], 1.0]
    }
    
    # 3. Fractional-power: c * (1 - y^a)
    candidates["c*(1-y^a)"] = {
        "func": lambda y, c, a: c * (1.0 - y**a),
        "p0": [U_val[0], 1.0]
    }
    
    # 4. Rational fractional-power: c * (1 - y^a) / (1 + b*y)
    candidates["c*(1-y^a)/(1+b*y)"] = {
        "func": lambda y, c, a, b: c * (1.0 - y**a) / (1.0 + b * y),
        "p0": [U_val[0], 1.0, 1.0]
    }
    
    # 5. Dual power-law: c * (1 - y^a)^b
    candidates["c*(1-y^a)^b"] = {
        "func": lambda y, c, a, b: c * (1.0 - y**a)**b,
        "p0": [U_val[0], 1.0, 1.0]
    }
    
    # 6. y-power combination: c * (y^a - y^b)
    candidates["c*(y^a - y^b)"] = {
        "func": lambda y, c, a, b: c * (y**a - y**b),
        "p0": [U_val[0], 0.1, 1.1]
    }
    
    # 7. Another clever form: c * (1 - y) * (1 + a*y + b*y^2)
    candidates["c*(1-y)*(1+a*y+b*y^2)"] = {
        "func": lambda y, c, a, b: c * (1.0 - y) * (1.0 + a * y + b * y**2),
        "p0": [U_val[0], 1.0, 1.0]
    }
    
    # 8. Fractional with power: c * (1 - y)^a * (1 + b*y)
    candidates["c*(1-y)^a*(1+b*y)"] = {
        "func": lambda y, c, a, b: c * (1.0 - y)**a * (1.0 + b * y),
        "p0": [U_val[0], 0.5, 1.0]
    }

    # 9. Exponential decay form: c * (1 - y^a) * exp(-b*y)
    candidates["c*(1-y^a)*exp(-b*y)"] = {
        "func": lambda y, c, a, b: c * (1.0 - y**a) * np.exp(-b * y),
        "p0": [U_val[0], 1.0, 0.5]
    }

    # 10. Rational fractional-power in numerator and denominator: c * (1 - y^a) / (1 + b * y^d)
    candidates["c*(1-y^a)/(1+b*y^d)"] = {
        "func": lambda y, c, a, b, d: c * (1.0 - y**a) / (1.0 + b * y**d),
        "p0": [U_val[0], 1.0, 1.0, 1.0]
    }
    
    # 11. User's specific suggestion: y^a + y^-b or y^a - y^b
    # Let's try: c * (y^a - y^b) or c * (1 - y^a + y^-b) - wait, U(1) = 0 is a strong constraint, and U(0) is finite.
    # Let's try c * (1 - y^a + b * (y^d - 1)) etc.
    # What about c * (y^a - 1) * (y^b - 1) ? That would be 0 at y=1.
    candidates["c*(1-y^a)*(1-b*y^d)"] = {
        "func": lambda y, c, a, b, d: c * (1.0 - y**a) * (1.0 - b * y**d),
        "p0": [U_val[0], 1.0, 0.5, 1.0]
    }

    print("Testing fits for velocity profile:")
    print("-" * 65)
    for name, info in candidates.items():
        try:
            popt, _ = curve_fit(info["func"], y_grid, U_val, p0=info["p0"], maxfev=10000)
            y_pred = info["func"](y_grid, *popt)
            
            # Compute R^2
            ss_res = np.sum((U_val - y_pred)**2)
            ss_tot = np.sum((U_val - np.mean(U_val))**2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-30)
            
            p_str = ", ".join([f"{p:.4f}" for p in popt])
            print(f"{name:<35} | R^2 = {r2:.6f} | Params: [{p_str}]")
        except Exception as e:
            print(f"{name:<35} | FAILED: {e}")

if __name__ == "__main__":
    main()

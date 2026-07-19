import sys
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(r"c:\Users\TLP-001\Documents\GitHub\project3_code")
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

_MENAHEM_DIR = _REPO_ROOT / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE
)
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _heat_kwargs_from_case,
    _shock_kwargs_from_case
)
from subsonic_heat_wave_fixed import SubsonicHeatWave
from piston_shock import PistonShock

def main():
    # 1. SUBSONIC HEAT WAVE (Fig 8: Constant Boundary Temperature, tau = 0)
    case_heat, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    heat_kwargs = _heat_kwargs_from_case(case_heat)
    solver_heat = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
    
    # Let's print out the scales at t = 1 ns (1e-9 s)
    t = 1e-9
    
    # Density scale
    # rho = 1 / v = 1 / (V * A^a1 * B^b1 * t^c1)
    # rho_scale = A^-a1 * B^-b1 * t^-c1
    rho_scale = (solver_heat.A**(-solver_heat.a1)) * (solver_heat.B**(-solver_heat.b1)) * (t**(-solver_heat.c1))
    
    # Velocity scale
    # u = U * A^a2 * B^b2 * t^c2
    u_scale = (solver_heat.A**solver_heat.a2) * (solver_heat.B**solver_heat.b2) * (t**solver_heat.c2)
    
    # Pressure scale
    # p = P * A^a3 * B^b3 * t^c3
    p_scale = (solver_heat.A**solver_heat.a3) * (solver_heat.B**solver_heat.b3) * (t**solver_heat.c3)
    
    # Temperature scale
    # T_scale = Tb * t^tau
    T_scale = solver_heat.Tb * (t**solver_heat.tau)
    
    print("SUBSONIC COEFFS AT t = 1 ns (seconds scale):")
    print(f"rho_scale(1 ns) = {rho_scale:.6f} g/cm^3")
    print(f"u_scale(1 ns) = {u_scale:.6e} cm/s = {u_scale/1e5:.6f} km/s")
    print(f"p_scale(1 ns) = {p_scale:.6e} Barye = {p_scale/1e12:.6f} MBar")
    print(f"T_scale(1 ns) = {T_scale:.6e} K = {T_scale/1.160452e6:.6f} HeV")
    
    # 2. SHOCK COEFFS
    case_shock, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    shock_kwargs = _shock_kwargs_from_case(case_shock)
    solver_shock = PistonShock(
        rho0=float(case_shock.rho0),
        omega=0.0,
        p0=shock_kwargs["p0"],
        tau=shock_kwargs["tau"],
        gamma=float(case_shock.r) + 1.0
    )
    
    # Shock scales at t = 1 ns
    # p = P * p0 * t^tau
    p_scale_s = solver_shock.p0 * (t**solver_shock.tau)
    # u = U * (v0 * p0 * t^(omega+tau))^(1/(2-omega))
    # for omega=0: u = U * (p0 * t^tau / rho0)^0.5
    u_scale_s = np.sqrt(p_scale_s / solver_shock.rho0)
    # rho = rho0 * R
    rho_scale_s = solver_shock.rho0
    # T = T_scale * T(y)
    T_scale_s = p_scale_s / (solver_shock.r * solver_shock.rho0)
    
    print("\nSHOCK COEFFS AT t = 1 ns:")
    print(f"rho0 = {rho_scale_s:.6f} g/cm^3")
    print(f"u_scale_s(1 ns) = {u_scale_s:.6e} cm/s = {u_scale_s/1e5:.6f} km/s")
    print(f"p_scale_s(1 ns) = {p_scale_s:.6e} Barye = {p_scale_s/1e12:.6f} MBar")
    print(f"T_scale_s(1 ns) = {T_scale_s:.6e} K = {T_scale_s/1.160452e6:.6f} HeV")

if __name__ == "__main__":
    main()

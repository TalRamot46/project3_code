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
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE
)
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _heat_kwargs_from_case,
    _shock_kwargs_from_case
)
from subsonic_heat_wave_fixed import SubsonicHeatWave
from piston_shock import PistonShock

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + 1e-30)

def main():
    # 1. SUBSONIC HEAT WAVE (Fig 8: Constant Boundary Temperature, tau = 0)
    print("====================================================")
    print("SUBSONIC ABLATION REGIME (tau = 0, T0 = 1 HeV)")
    print("====================================================")
    case_heat, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    heat_kwargs = _heat_kwargs_from_case(case_heat)
    solver_heat = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
    
    y_grid = np.linspace(0.0, 1.0 - 1e-6, 500)
    xsi_vec = y_grid * solver_heat.xsi_f
    profiles_heat = solver_heat.get_self_similar_profiles(xsi_vec=xsi_vec)
    
    imin = profiles_heat["imin"]
    y_valid = y_grid[imin:]
    T_valid = profiles_heat["T"][imin:]
    P_valid = profiles_heat["P"][imin:]
    U_valid = profiles_heat["U"][imin:]
    # Calculate density rho = 1 / V
    V_valid = profiles_heat["V"][imin:]
    rho_valid = 1.0 / V_valid
    
    print(f"xsi_f = {solver_heat.xsi_f:.6f}")
    print(f"A = {solver_heat.A:.6e}, B = {solver_heat.B:.6e}")
    print(f"Powers: a = {solver_heat.a:.6f}, b = {solver_heat.b:.6f}, c = {solver_heat.c:.6f}")
    print(f"a1={solver_heat.a1:.6f}, b1={solver_heat.b1:.6f}, c1={solver_heat.c1:.6f}")
    print(f"a2={solver_heat.a2:.6f}, b2={solver_heat.b2:.6f}, c2={solver_heat.c2:.6f}")
    print(f"a3={solver_heat.a3:.6f}, b3={solver_heat.b3:.6f}, c3={solver_heat.c3:.6f}")
    
    # Let's fit T, P, u, rho in subsonic
    # T = T_f * (1 - y)^d. Note: T(y=0) should be 1.0.
    def power_law_front(y, c, d):
        return c * (1.0 - y)**d
    
    popt_T, _ = curve_fit(power_law_front, y_valid, T_valid, p0=[1.0, 0.5])
    print(f"T(y) fit: c*(1-y)^d -> c = {popt_T[0]:.6f}, d = {popt_T[1]:.6f} | R^2 = {compute_r2(T_valid, power_law_front(y_valid, *popt_T)):.6f}")
    
    # P = P0_sim * y^c + P1_sim * y^(c+d)
    def power_law_origin(y, a, b, c, d):
        return a * (y)**(c) + b*y**(c+d)
    
    popt_P, _ = curve_fit(power_law_origin, y_valid, P_valid, p0=[0.355, 0.5, 0.04, 2.3])
    print(f"P(y) fit: a*y^c + b*y^(c+d) -> a={popt_P[0]:.6f}, b={popt_P[1]:.6f}, c={popt_P[2]:.6f}, d={popt_P[3]:.6f} | R^2 = {compute_r2(P_valid, power_law_origin(y_valid, *popt_P)):.6f}")
    
    # u (dimensionless U). Let's fit U using: c * (1 - y) * y^-b (the Clever 2-parameter singular power law)
    # and c * (1 - y) * (y^a + y^-b) (User 3-parameter singular sum of powers)
    def fit_u_singular_2p(y, c, b):
        return c * (1.0 - y) * (y**(-b))
    def fit_user(y, c, a, b):
        return c * (1.0 - y) * (y**a + y**(-b))
        
    popt_u2, _ = curve_fit(fit_u_singular_2p, y_valid, U_valid, p0=[-2.0, 0.3])
    popt_u3, _ = curve_fit(fit_user, y_valid, U_valid, p0=[-1.0, 0.3, 0.3])
    print(f"U(y) Clever 2-Param fit: c*(1-y)*y^-b -> c = {popt_u2[0]:.6f}, b = {popt_u2[1]:.6f} | R^2 = {compute_r2(U_valid, fit_u_singular_2p(y_valid, *popt_u2)):.6f}")
    print(f"U(y) User 3-Param fit: c*(1-y)*(y^a + y^-b) -> c = {popt_u3[0]:.6f}, a = {popt_u3[1]:.6f}, b = {popt_u3[2]:.6f} | R^2 = {compute_r2(U_valid, fit_user(y_valid, *popt_u3)):.6f}")
    
    # rho = 1/V. Since V goes to 0 near y=1 as Vstar*(1-y)^delta where delta = 1/q.
    # So rho = rho_origin * (1-y)^-d. Let's fit rho(y) using a power law: c * (1-y)^-d
    def fit_rho(y, c, d):
        return c * (1.0 - y)**(-d)
    
    popt_rho, _ = curve_fit(fit_rho, y_valid, rho_valid, p0=[1.0, 0.5])
    print(f"rho(y) fit: c*(1-y)^-d -> c = {popt_rho[0]:.6f}, d = {popt_rho[1]:.6f} | R^2 = {compute_r2(rho_valid, fit_rho(y_valid, *popt_rho)):.6f}")
    
    # 2. PISTON SHOCK (Fig 7: tau = -0.447, P0 = 2.71 MBar)
    print("\n====================================================")
    print("SHOCK REGIME (tau = -0.447, P0 = 2.71 MBar)")
    print("====================================================")
    case_shock, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    shock_kwargs = _shock_kwargs_from_case(case_shock)
    # In plot_piston_shock.py, the omega is 0.0 and p0 is rescaled
    from piston_shock import PistonShock
    solver_shock = PistonShock(
        rho0=float(case_shock.rho0),
        omega=0.0,
        p0=shock_kwargs["p0"],
        tau=shock_kwargs["tau"],
        gamma=float(case_shock.r) + 1.0
    )
    
    y_grid_s = np.linspace(0.0, 1.0, 500)
    xsi_vec_s = y_grid_s * solver_shock.xsi_s
    xsi_vec_s[0] = 1e-10
    V_val_s, U_val_s, P_val_s = solver_shock.get_self_similar_profiles(xsi_vec=xsi_vec_s)
    
    R_val_s = 1.0 / V_val_s
    T_val_s = P_val_s * V_val_s / solver_shock.r
    
    valid_idx_s = (y_grid_s > 0.005) & np.isfinite(V_val_s) & np.isfinite(U_val_s) & np.isfinite(P_val_s)
    y_valid_s = y_grid_s[valid_idx_s]
    R_valid_s = R_val_s[valid_idx_s]
    U_valid_s = U_val_s[valid_idx_s]
    P_valid_s = P_val_s[valid_idx_s]
    T_valid_s = T_val_s[valid_idx_s]
    
    print(f"xsi_s = {solver_shock.xsi_s:.6f}")
    print(f"rho0 = {solver_shock.rho0:.6f}, p0 = {solver_shock.p0:.6e}, tau = {solver_shock.tau:.6f}")
    
    # Boundary values at front (shock, y=1) and origin (piston, y=0)
    R_s = R_valid_s[-1]
    U_s = U_valid_s[-1]
    P_s = P_valid_s[-1]
    T_s = T_valid_s[-1]
    U_0 = U_valid_s[0]
    
    print(f"At Piston (y=0): R0={R_valid_s[0]:.6f}, U0={U_0:.6f}, P0={P_valid_s[0]:.6f}, T0={T_valid_s[0]:.6f}")
    print(f"At Shock (y=1): Rs={R_s:.6f}, Us={U_s:.6f}, Ps={P_s:.6f}, Ts={T_s:.6f}")
    
    # Fitting Shock Profiles
    # R(y) = R_s * y^-d
    def power_law_R(y, d):
        return R_s * (y**(-d))
    popt_s_R, _ = curve_fit(power_law_R, y_valid_s, R_valid_s, p0=[0.3])
    print(f"R(y) fit: Rs*y^-d -> d = {popt_s_R[0]:.6f} | R^2 = {compute_r2(R_valid_s, power_law_R(y_valid_s, *popt_s_R)):.6f}")
    
    # P(y) = 1.0 - (1.0 - P_s)*y^d
    def power_law_P(y, d):
        return 1.0 - (1.0 - P_s) * (y**d)
    popt_s_P, _ = curve_fit(power_law_P, y_valid_s, P_valid_s, p0=[1.0])
    print(f"P(y) fit: 1.0 - (1-Ps)*y^d -> d = {popt_s_P[0]:.6f} | R^2 = {compute_r2(P_valid_s, power_law_P(y_valid_s, *popt_s_P)):.6f}")
    
    # U(y) = U_0 - (U_0 - U_s)*y^d
    def power_law_U(y, d):
        return U_0 - (U_0 - U_s) * (y**d)
    popt_s_U, _ = curve_fit(power_law_U, y_valid_s, U_valid_s, p0=[1.0])
    print(f"U(y) fit: U0 - (U0-Us)*y^d -> d = {popt_s_U[0]:.6f} | R^2 = {compute_r2(U_valid_s, power_law_U(y_valid_s, *popt_s_U)):.6f}")
    
    # T(y) = T_s * y^d
    def power_law_T(y, d):
        return T_s * (y**d)
    popt_s_T, _ = curve_fit(power_law_T, y_valid_s, T_valid_s, p0=[0.3])
    print(f"T(y) fit: Ts*y^d -> d = {popt_s_T[0]:.6f} | R^2 = {compute_r2(T_valid_s, power_law_T(y_valid_s, *popt_s_T)):.6f}")

if __name__ == "__main__":
    main()

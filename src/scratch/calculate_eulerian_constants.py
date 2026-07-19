# calculate_eulerian_constants.py
import sys
from pathlib import Path
import numpy as np

# Ensure proper imports
_REPO_ROOT = Path("c:/Users/TLP-001/Documents/GitHub/project3_code")
_REPO_PARENT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_PARENT))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "menahem_new"))
sys.path.insert(0, str(_REPO_ROOT / "ictt29"))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE
)
import sub_fitting
import shock_fitting

def main():
    case_heat, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    solver_heat = sub_fitting.get_cached_sub_solver(case_heat, "heat_const_T")
    params_heat = sub_fitting.perform_subsonic_fitting(solver_heat)
    
    case_shock, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    solver_shock = shock_fitting.get_cached_shock_solver(case_shock, "shock_const_T")
    params_shock = shock_fitting.perform_shock_fitting(solver_shock)
    
    t_val = 1e-9 # 1 ns
    
    # 1) Subsonic Scaling
    hs = solver_heat
    ss = solver_shock
    
    m_f = hs.ablated_mass(time=t_val)
    m_s = ss.shocked_mass(time=t_val)
    
    # Shock scales
    pos_scale = ss._position_temporal_factor(time=t_val)
    q1 = 1.0 - ss.omega
    q2 = (2.0 - ss.omega) / (ss.tau + 2.0)
    
    x_p_fit = pos_scale * q2 * ss.U0
    
    xsi_mf = m_f * ss.xsi_over_m(time=t_val)
    y_mf = xsi_mf / ss.xsi_s
    
    _, _, U_fit_sh_mf, rho_fit_sh_mf = shock_fitting.fit_by_params(np.array([y_mf]), params_shock)
    V_fit_sh_mf = 1.0 / rho_fit_sh_mf[0]
    
    x_af_fit = pos_scale * (q1 * xsi_mf * V_fit_sh_mf + q2 * U_fit_sh_mf[0])
    
    # Subsonic position coefficient
    C_pos = hs._position_temporal_factor(time=t_val)
    
    # Boundary position
    x_b_fit = hs.boundary_position(time=t_val) + x_af_fit
    
    # Shock front position
    x_s_fit = ss.shock_position(time=t_val)
    
    # Print out values in CGS (cm) and microns
    print(f"--- AT t = 1 ns ---")
    print(f"m_f_1ns = {m_f:.8e}")
    print(f"m_s_1ns = {m_s:.8e}")
    print(f"C_pos_1ns = {C_pos:.8e}")
    print(f"pos_scale_1ns = {pos_scale:.8e}")
    print(f"x_af_fit_1ns = {x_af_fit:.8e}")
    print(f"x_b_fit_1ns = {x_b_fit:.8e}")
    print(f"x_p_fit_1ns = {x_p_fit:.8e}")
    print(f"x_s_fit_1ns = {x_s_fit:.8e}")
    print(f"U0_sub = {hs.U0:.8f}")
    print(f"U0_shock = {ss.U0:.8f}")
    print(f"xsi_mf = {xsi_mf:.8f}")
    print(f"y_mf = {y_mf:.8f}")
    print(f"V_fit_sh_mf = {V_fit_sh_mf:.8f}")
    print(f"U_fit_sh_mf_0 = {U_fit_sh_mf[0]:.8f}")
    print(f"q1 = {q1:.8f}")
    print(f"q2 = {q2:.8f}")
    
    # Let's also get the exponents
    print(f"c2 subsonic (c_u) = {hs.c2:.8f}")
    print(f"c_v subsonic (c1) = {hs.c1:.8f}")
    print(f"tau subsonic (c3) = {hs.c3:.8f}")
    print(f"tau shock = {ss.tau:.8f}")
    print(f"omega shock = {ss.omega:.8f}")
    print(f"xsi_s = {ss.xsi_s:.8f}")

if __name__ == "__main__":
    main()

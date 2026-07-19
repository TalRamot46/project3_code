# ictt29/build_table.py
import sys
from pathlib import Path
import numpy as np

# Setup paths so all project modules are resolvable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT.parent))  # Enables "project3_code" package resolution
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "menahem_new"))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE
)
import data_loader

def main():
    # Load heat solver
    case_heat, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    solver_heat = data_loader.get_sub_similarity_solver(case_heat, "heat_const_T")
    
    # Load shock solver
    case_shock, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    solver_shock = data_loader.get_shock_similarity_solver(case_shock, "shock_const_T")
    
    # Evaluate at y = 0.0, 0.1, ..., 1.0
    y_vals = np.linspace(0.0, 0.9, 10)
    y_vals = np.append(y_vals, 0.999)
    
    # Subsonic Heat Wave
    xsi_vec_h = y_vals * solver_heat.xsi_f
    # Set the last one slightly less than xsi_f to avoid numerical/asymptotic boundary issues,
    # or handle analytically. Let's see how the solver behaves at xsi_f exactly.
    # We can evaluate at y_vals
    profiles_h = solver_heat.get_self_similar_profiles(xsi_vec=xsi_vec_h)
    
    T_h = profiles_h["T"]
    P_h = profiles_h["P"]
    U_h = profiles_h["U"]
    V_h = profiles_h["V"]
    
    print("Subsonic Heat Solver results:")
    for y, T, P, U, V in zip(y_vals, T_h, P_h, U_h, V_h):
        # density is 1/V
        rho = 1.0 / V if (V is not None and V > 0 and np.isfinite(V)) else float('inf')
        if y < 0.95:
            print(f"y_h = {y:.1f}: T = {T:.5e}, rho = {rho:.5e}, P = {P:.5e}, U = {U:.5e}")
        else:
            print(f"y_h = {y:.3f}: T = {T:.5e}, rho = {rho:.5e}, P = {P:.5e}, U = {U:.5e}")

    # Piston Shock
    y_vals[-1] = 1.0
    xsi_vec_s = y_vals * solver_shock.xsi_s
    # Avoid division by zero at y = 0
    xsi_vec_s_eval = np.copy(xsi_vec_s)
    xsi_vec_s_eval[0] = 1e-10

    
    V_s, U_s, P_s = solver_shock.get_self_similar_profiles(xsi_vec=xsi_vec_s_eval)
    rho_s = 1.0 / V_s
    T_s = (P_s * V_s**(1.0 - 0.14))**(1.0 / 1.6)
    
    print("\nShock Solver results:")
    for y, T, rho, P, U in zip(y_vals, T_s, rho_s, P_s, U_s):
        print(f"y_s = {y:.1f}: T = {T:.5e}, rho = {rho:.5e}, P = {P:.5e}, U = {U:.5e}")

if __name__ == "__main__":
    main()
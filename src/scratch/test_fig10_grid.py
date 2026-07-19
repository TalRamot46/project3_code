# scratch/test_fig10_grid.py
import sys
from pathlib import Path
from dataclasses import replace

# Ensure proper package imports
_REPO_ROOT = Path(r"c:\Users\TLP-001\Documents\GitHub\project3_code")
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))
_MENAHEM_DIR = _REPO_ROOT / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.menahem_new.ablation_solver import AblationSolver

def test_run():
    case, config = get_preset("fig_10_comparison")
    # Set x_max to 1.5e-2 so that total mass is 1.5e-3
    case_fixed = replace(case, x_max=1.5e-2)
    
    config = replace(config, show_plot=False, show_slider=False)
    print("\nRunning simulation with x_max = 1.5e-2, rho0 = 0.1...")
    try:
        _, _, _, history = simulate_rad_hydro(rad_hydro_case=case_fixed, simulation_config=config)
        print("  Simulation completed successfully.")
        
        ablation_solver = AblationSolver(
            Tb=case_fixed.T0_Kelvin,
            tau=case_fixed.tau,
            g=case_fixed.g_Kelvin,
            alpha=case_fixed.alpha,
            lambdap=case_fixed.lambda_,
            f_heat=case_fixed.f_Kelvin,
            beta_heat=case_fixed.beta_Rosen,
            mu_heat=case_fixed.mu,
            gamma_heat=case_fixed.r + 1.0,
            rho0=case_fixed.rho0,
            omega=case_fixed.omega if hasattr(case_fixed, "omega") else 0.0,
            f_shock=case_fixed.f_Kelvin,
            beta_shock=case_fixed.beta_Rosen,
            mu_shock=case_fixed.mu,
            gamma_shock=case_fixed.r + 1.0
        )
        
        t_max = max(history.t)
        t_target = t_max
        idx = np.argmin(np.abs(np.array(history.t) - t_target))
        t_actual = history.t[idx]
        m_sim = history.m[idx]
        m_max = m_sim[-1]
        
        m_f = ablation_solver.heat_solver.ablated_mass(time=t_actual)
        m_s = ablation_solver.shock_solver.shocked_mass(time=t_actual)
        
        print(f"Time: {t_actual*1e9:.3f} ns")
        print(f"  Grid Max Mass: {m_max:.6e} g/cm^2")
        print(f"  Subsonic front m_f: {m_f:.6e} g/cm^2")
        print(f"  Shock front m_s: {m_s:.6e} g/cm^2")
        
        if m_f > m_max:
            print(f"  WARNING: Subsonic front has exited the grid! (m_f > m_max)")
        elif m_s > m_max:
            print(f"  WARNING: Shock front has exited the grid! (m_s > m_max)")
        else:
            print(f"  All fronts are inside the grid!")
            
    except Exception as e:
        print("  Simulation failed:", e)

if __name__ == "__main__":
    import numpy as np
    test_run()

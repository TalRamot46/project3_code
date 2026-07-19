# scratch/test_fig10_run.py
import sys
from pathlib import Path
from dataclasses import replace

# Ensure proper package imports
_REPO_ROOT = Path(r"c:\Users\TLP-001\Documents\GitHub\project3_code")
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro

def test_run():
    case, config = get_preset("fig_10_comparison")
    print("Default Case parameters:")
    print("  rho0:", case.rho0)
    print("  x_max:", case.x_max)
    print("  t_sec_end:", case.t_sec_end)
    
    # Run with default rho0 (0.1)
    config = replace(config, show_plot=False, show_slider=False)
    print("\nRunning simulation with default case...")
    try:
        _, _, _, history = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)
        print("  Simulation completed. Final t:", history.t[-1])
        print("  Final x bounds:", history.x[-1][0], "to", history.x[-1][-1])
    except Exception as e:
        print("  Simulation failed:", e)

    # Run with rho0 = 19.32
    print("\nRunning simulation with rho0 = 19.32...")
    case_fixed = replace(case, rho0=19.32)
    try:
        _, _, _, history_fixed = simulate_rad_hydro(rad_hydro_case=case_fixed, simulation_config=config)
        print("  Simulation completed. Final t:", history_fixed.t[-1])
        print("  Final x bounds:", history_fixed.x[-1][0], "to", history_fixed.x[-1][-1])
    except Exception as e:
        print("  Simulation failed:", e)

if __name__ == "__main__":
    test_run()

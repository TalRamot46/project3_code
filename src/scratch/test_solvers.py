# scratch/test_solvers.py
import sys
from pathlib import Path

_REPO_ROOT = Path(r"c:\Users\TLP-001\Documents\GitHub\project3_code")
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_9_CONSTANT_FLUX,
    PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE
)
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _heat_kwargs_from_case,
    _ablation_kwargs_from_case
)
from project3_code.menahem_new.subsonic_heat_wave_og import SubsonicHeatWave
from project3_code.menahem_new.piston_shock_og import PistonShock
from project3_code.menahem_new.ablation_solver_og import AblationSolver

def test_preset(preset_name, label):
    print(f"\n--- Testing Preset: {label} ({preset_name}) ---")
    case, config = get_preset(preset_name)
    
    # 1) Subsonic Solver
    print("Solving subsonic similarity...")
    heat_kwargs = _heat_kwargs_from_case(case)
    sub_solver = SubsonicHeatWave(**heat_kwargs).find_xsi_f()
    print(f"  Subsonic solved. xsi_f = {sub_solver.xsi_f:.6f}")
    
    # 2) Ablation Solver (patched)
    print("Solving patched AblationSolver...")
    ablation_kwargs = _ablation_kwargs_from_case(case)
    ablation_solver = AblationSolver(**ablation_kwargs)
    print("  AblationSolver initialized successfully.")
    
    print(f"Preset {label} passed basic checks!")

if __name__ == "__main__":
    test_preset(PRESET_FIG_9_CONSTANT_FLUX, "Fig 9")
    test_preset(PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE, "Fig 10")

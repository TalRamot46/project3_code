import sys
from pathlib import Path

# Set up paths
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT.parent))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "ictt29"))
sys.path.insert(0, str(_REPO_ROOT / "menahem_new"))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE
)

import sub_fitting
import shock_fitting

def check_sub():
    print("=== Subsonic Fitting ===")
    case, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    solver = sub_fitting.get_cached_sub_solver(case, "heat_const_T")
    params = sub_fitting.perform_subsonic_fitting(solver)
    
    print("Best Pressure:", params["best_p"]["id"], params["best_p"]["name"])
    print("Popt P:", params["best_p"]["popt"])
    print("Best Velocity:", params["best_u"]["id"], params["best_u"]["name"])
    print("Popt U:", params["best_u"]["popt"])
    print("Smith R for Temp:", params["popt_T"])

def check_shock():
    print("=== Shock Fitting ===")
    case, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    solver = shock_fitting.get_cached_shock_solver(case, "shock_const_T")
    params = shock_fitting.perform_shock_fitting(solver)
    
    print("Best Temperature:", params["best_T"]["id"], params["best_T"]["name"])
    print("Popt T:", params["best_T"]["popt"])
    print("Best Velocity:", params["best_u"]["id"], params["best_u"]["name"])
    print("Popt U:", params["best_u"]["popt"])
    print("Popt P (pressure power law):", params["popt_P"])
    print("Popt rho low:", params["popt_rho_low"])

if __name__ == "__main__":
    check_sub()
    check_shock()

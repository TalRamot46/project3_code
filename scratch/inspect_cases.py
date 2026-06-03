# scratch/inspect_cases.py
import sys
from pathlib import Path
import numpy as np

# Ensure proper package imports
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_MENAHEM_DIR = _REPO_PARENT / "project3_code" / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

import pickle
import scipy.integrate

def inspect_case(case_label: str):
    print("=" * 60)
    print(f"INSPECTING CASE: {case_label}")
    print("=" * 60)
    cache_path = Path("results/ictt/cache") / f"{case_label}_full_fitting_cache.pkl"
    if not cache_path.exists():
        print("Cache file not found.")
        return
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    history = data["history"]
    solver = data["solver"]
    
    if solver is not None:
        if hasattr(solver.heat_solver, "fode"):
            solver.heat_solver.ode_solver = scipy.integrate.ode(solver.heat_solver.fode).set_integrator(solver.heat_solver.ode_scheme)
        if hasattr(solver.shock_solver, "fode"):
            solver.shock_solver.ode_solver = scipy.integrate.ode(solver.shock_solver.fode).set_integrator(solver.shock_solver.ode_scheme)
            
    print("Simulation times count:", len(history.t))
    t_max = max(history.t)
    target_times = [0.5 * t_max, 0.75 * t_max, t_max]
    
    for t_target in target_times:
        idx = np.argmin(np.abs(np.array(history.t) - t_target))
        t_actual = history.t[idx]
        m_sim = history.m[idx]
        m_max = m_sim[-1]
        
        m_f = solver.heat_solver.ablated_mass(time=t_actual)
        m_s = solver.shock_solver.shocked_mass(time=t_actual)
        
        print(f"Time: {t_target*1e9:.3f} ns (actual: {t_actual*1e9:.3f} ns)")
        print(f"  Grid Max Mass: {m_max:.6e} g/cm^2")
        print(f"  Subsonic front m_f: {m_f:.6e} g/cm^2")
        print(f"  Shock front m_s: {m_s:.6e} g/cm^2")
        
        # Check if subsonic front exceeds the grid mass
        if m_f > m_max:
            print(f"  WARNING: Subsonic front has exited the grid! (m_f > m_max)")
        elif m_s > m_max:
            print(f"  WARNING: Shock front has exited the grid! (m_s > m_max)")
        else:
            print(f"  All fronts are inside the grid.")

if __name__ == "__main__":
    inspect_case("const_S")
    inspect_case("const_P_shock")

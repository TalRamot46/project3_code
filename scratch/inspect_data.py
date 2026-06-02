# scratch/inspect_data.py
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

cache_path = Path("results/ictt/cache/full_fitting_cache.pkl")
if cache_path.exists():
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
    idx = np.argmin(np.abs(np.array(history.t) - 1.0e-9))
    print("  Simulation time:", history.t[idx])
    print("  Simulation x bounds (cm):", history.x[idx][0], "to", history.x[idx][-1])
    print("  Simulation m bounds (g/cm^2):", history.m[idx][0], "to", history.m[idx][-1])
    
    # Solver test
    m_sim = history.m[idx]
    sol = solver.solve(mass=m_sim, time=history.t[idx])
    print("  Solver boundary_position:", sol["boundary_position"])
    print("  Solver heat_position (ablation front):", sol["heat_position"])
    print("  Solver piston_position:", sol["piston_position"])
    print("  Solver shock_position:", sol["shock_position"])
    print("  Solver position bounds (cm):", sol["position"][0], "to", sol["position"][-1])
else:
    print("Cache file not found.")

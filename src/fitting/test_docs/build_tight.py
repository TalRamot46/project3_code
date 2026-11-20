import bootstrap  # noqa: F401  (sys.path)
import os
from paths import CACHE
import sys, pickle, logging; logging.disable(50)
from tight import make_tight_heat_solver
from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.verification.menahem_comparison import _heat_kwargs_from_case
from build_solvers import PRESETS
for k in map(int, sys.argv[1:]):
    case,_ = get_preset(PRESETS[k]); h = make_tight_heat_solver(_heat_kwargs_from_case(case))
    print(k, f"xsi_f={h.xsi_f:.8f} Pf={h.Pf:.8f} U0={h.U0:.5f} V0={h.V0:.5g} P0={h.P0:.3g} T0={h.T0:.6f} S0={h.S0:.5f}", flush=True)
    del h.ode_solver
    pickle.dump(h, open(os.path.join(CACHE, f"heat_tight_test{k}.pkl"), "wb"))

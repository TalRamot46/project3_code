import os
from paths import CACHE
import pickle, numpy as np, scipy.integrate, warnings
warnings.filterwarnings("ignore")
if not hasattr(scipy.integrate, "simps"): scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(np, "trapz"): np.trapz = np.trapezoid
import project3_code.menahem_new.ablation_solver_og  # noqa  (class path for unpickle)
def load(k):
    s = pickle.load(open(os.path.join(CACHE, f"cache_test{k}.pkl"), "rb"))
    for o in (s.heat_solver, s.shock_solver):
        o.ode_solver = scipy.integrate.ode(o.fode).set_integrator(o.ode_scheme)
    return s

def load_heat_tight(k):
    from tight import refresh
    kk = 1 if k in (3, 4) else k   # Tests 3, 4 share Test 1's subsonic solution (no omega, rho0 dependence)
    h = pickle.load(open(os.path.join(CACHE, f"heat_tight_test{kk}.pkl"), "rb"))
    return refresh(h)

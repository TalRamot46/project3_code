import bootstrap  # noqa: F401  (sys.path)
import os
from paths import CACHE
import sys, time, pickle, numpy as np, scipy.integrate
if not hasattr(scipy.integrate, "simps"): scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(np, "trapz"): np.trapz = np.trapezoid
from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
import project3_code.rad_hydro_sim.problems.presets_config as pc
from project3_code.rad_hydro_sim.verification.menahem_comparison import _ablation_kwargs_from_case
from project3_code.menahem_new.ablation_solver_og import AblationSolver
PRESETS = {1: pc.PRESET_FIG_8_CONSTANT_TEMPERATURE, 2: pc.PRESET_FIG_9_CONSTANT_FLUX,
           3: pc.CONSTANT_TEMPERATURE_OMEGA_0_5_FULL, 4: pc.CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL,
           5: pc.PRESET_COPPER_CONST_TEMPERATURE, 6: pc.PRESET_ALUMINUM_CONST_TEMPERATURE}
if __name__ == "__main__":
    for k in map(int, sys.argv[1:]):
        case, _ = get_preset(PRESETS[k])
        kw = _ablation_kwargs_from_case(case)
        t0 = time.time()
        s = AblationSolver(**kw)
        h, sh = s.heat_solver, s.shock_solver
        print(k, {kk: (round(v,6) if isinstance(v,float) else v) for kk,v in kw.items()})
        print(f"  built in {time.time()-t0:.1f}s xsi_f={h.xsi_f:.6g} Pf={h.Pf:.6g} U0={h.U0:.6g} T0={h.T0:.6g} S0={h.S0:.6g} | xsi_s={sh.xsi_s:.6g} p0={sh.p0:.4g} tau_s={sh.tau:.6g} omega={sh.omega} U0s={sh.U0:.5g} V0s={sh.V0:.4g} Vs={sh.Vs:.4g}")
        print("  m_f(1ns)=%.5g m_s(1ns)=%.5g" % (h.ablated_mass(time=1e-9), sh.shocked_mass(time=1e-9)))
        for o in (h, sh):
            if hasattr(o, "ode_solver"): del o.ode_solver
        pickle.dump(s, open(os.path.join(CACHE, f"cache_test{k}.pkl"), "wb"))

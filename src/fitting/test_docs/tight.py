"""Re-shoot the subsonic heat-wave solution with tight ODE tolerances.

The original solver integrates with dopri5 defaults (rtol=1e-6, atol=1e-12). Near the
ablation surface (xi -> 0) the solution is singular (V -> inf, P -> 0) and the default
tolerance leaves P(0), U(0), V near the surface unconverged; see README / doc notes.
"""
import numpy as np, scipy.integrate, scipy.optimize, logging
from project3_code.menahem_new.subsonic_heat_wave_og import SubsonicHeatWave, NumericalParameters

RTOL, ATOL = 1e-11, 1e-15

def tight_ode(h, rtol=RTOL, atol=ATOL):
    return scipy.integrate.ode(h.fode).set_integrator("dopri5", rtol=rtol, atol=atol, nsteps=10**7)

def make_tight_heat_solver(kw, rtol=RTOL, atol=ATOL):
    h = SubsonicHeatWave(**kw)
    h.ode_solver = tight_ode(h, rtol, atol)
    h.find_xsi_f()
    return h

def refresh(h, rtol=RTOL, atol=ATOL):
    h.ode_solver = tight_ode(h, rtol, atol)
    return h

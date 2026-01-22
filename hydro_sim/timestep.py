import numpy as np
from eos import sound_speed

def compute_dt(state, gamma, CFL):
    c = sound_speed(state.rho, state.p, gamma)     # cells
    dr = state.r[1:] - state.r[:-1]                # cells
    dt = CFL * np.min(dr / (c + 1e-30))
    return dt

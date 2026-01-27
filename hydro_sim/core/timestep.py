import numpy as np
from .eos import sound_speed
from core.state import HydroState

def compute_dt_acoustic(state, gamma, CFL):
    c = sound_speed(state.rho, state.p, gamma)     # cells
    dr = state.x[1:] - state.x[:-1]                # cells
    dt = CFL * np.min(dr / (c + 1e-30))
    return dt

def compute_dt_crossing(x_nodes, u_nodes, CFL):
    du = u_nodes[:-1] - u_nodes[1:]                     # per cell
    dx = x_nodes[1:] - x_nodes[:-1]
    collapsing = du > 0.0
    if np.any(collapsing):
        dt_cross = CFL * np.min(dx[collapsing] / (du[collapsing] + 1e-30))
    else:
        dt_cross = np.inf
    return dt_cross

def compute_dt_cfl(x_nodes, u_nodes, rho_cells, p_cells, gamma, CFL):
    # 1) acoustic CFL
    dt_acoustic = compute_dt_acoustic(
        HydroState(t=0.0, x=x_nodes, u=u_nodes, a=None,
                   V=None, rho=rho_cells, e=None, p=p_cells, q=None),
        gamma, CFL)
    
    # 2) mesh-crossing limiter (prevents x_{i+1} < x_i after update)
    # cell is collapsing if u_left > u_right
    dt_cross = compute_dt_crossing(x_nodes, u_nodes, CFL)
    return min(dt_acoustic, dt_cross)
    
import numpy as np
from .eos import sound_speed
from core.state import HydroState

def compute_dt_acoustic(state, gamma, CFL):
    c = sound_speed(state.rho, state.p, gamma)     # cells
    dx = state.x[1:] - state.x[:-1]                # cells

    u_cell = 0.5 * (state.u[:-1] + state.u[1:])  # cell-centered velocity
    wave_speed = np.abs(u_cell) + c
    dt = CFL * np.min(dx / (wave_speed + 1e-30))
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

def compute_dt_volchange(x_nodes, u_nodes, eta=0.1):
    dx = x_nodes[1:] - x_nodes[:-1]
    dudx = u_nodes[1:] - u_nodes[:-1]   # per cell: du
    # avoid division by zero
    dt = np.min(eta * dx / (np.abs(dudx) + 1e-30))
    return dt


def compute_dt_cfl(x_nodes, u_nodes, rho_cells, p_cells, gamma, CFL):
    # 1) acoustic CFL
    dt_acoustic = compute_dt_acoustic(
        HydroState(t=0.0, x=x_nodes, u=u_nodes, a=None,
                   V=None, rho=rho_cells, e=None, p=p_cells, q=None),
        gamma, CFL)
    
    # 2) mesh-crossing limiter (prevents x_{i+1} < x_i after update)
    # cell is collapsing if u_left > u_right
    dt_cross = compute_dt_crossing(x_nodes, u_nodes, CFL)

    dt_vol_change = compute_dt_volchange(x_nodes, u_nodes, eta=0.1)
    return min(dt_acoustic, dt_cross, dt_vol_change)
    
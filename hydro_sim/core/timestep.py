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


# ============================================================================
# Radiation timestep (for rad-hydro)
# ============================================================================

def update_dt_relchange(dt, new_E, E, new_UR, UR, *, dtfac=0.05, dtmin=2e-15, dtmax=5e-13, growth_cap=1.1):
    """
    Adaptive dt based on max relative change in E and UR.

    dtfac: target relative change per step (~0.05 means ~5%)
    dtmax: absolute cap on dt
    growth_cap: allow dt to increase by at most 10% per step (1.1)
    """
    # Protect from division by tiny numbers
    E_min = np.max(np.abs(E)) * 1e-3 + 1e-30
    dE = np.max(np.abs(new_E - E) / (np.abs(E) + E_min))

    U_min = np.max(np.abs(UR)) * 1e-3 + 1e-30
    dU = np.max(np.abs(new_UR - UR) / (np.abs(UR) + U_min))

    # Avoid blow-ups if change is ~0
    dE = max(dE, 1e-16)
    dU = max(dU, 1e-16)

    dttag1 = dt / dE * dtfac
    dttag2 = dt / dU * dtfac

    dt_new = min(dttag1, dttag2, growth_cap * dt, dtmax)
    dt_new = max(dt_new, dtmin)
    return dt_new


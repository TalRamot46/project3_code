import numpy as np
from geometry import Geometry
from eos import pressure_ideal_gas, sound_speed
from viscosity import artificial_viscosity
from grid import cell_volumes
from state import HydroState
from boundary import apply_velocity_bc

def compute_acceleration_nodes(x_nodes: np.ndarray,
                               p_cells: np.ndarray,
                               q_cells: np.ndarray,
                               m_cells: np.ndarray,
                               geom: Geometry) -> np.ndarray:
    """
    PDF Eq.(18) in node form:
      a_i = -2 * beta * r_i^alpha * [ (p+q)_{i+1/2} - (p+q)_{i-1/2} ] / (m_{i+1/2}+m_{i-1/2})

    For boundaries (i=0 and i=N-1), we use one-sided gradients:
      a_0    uses (p+q)_{1/2} - (p+q)_{1/2} => 0  OR a one-sided approx
      a_{N-1} similarly.
    To be "outflow-like", we do one-sided:
      a_0     = -2*beta*x0^alpha * [ (p+q)_{1/2} - (p+q)_{1/2} ]/(m_{1/2}) = 0
      a_{N-1} = 0
    and rely on large enough domain ([-1,1], t=0.25).
    """
    Nnodes = x_nodes.size
    a = np.zeros(Nnodes)

    pq = p_cells + q_cells

    # interior
    left = pq[:-1]     # i-1/2 for i=1..N-2
    right = pq[1:]     # i+1/2 for i=1..N-2
    denom = m_cells[:-1] + m_cells[1:]
    a[1:-1] = -2.0 * geom.beta * (x_nodes[1:-1] ** geom.alpha) * (right - left) / denom

    # boundaries: keep 0 (transmissive-ish; avoids wall reflection)
    a[0] = 0.0
    a[-1] = 0.0
    return a


def compute_dt_cfl(x_nodes, u_nodes, rho_cells, p_cells, gamma, CFL):
    # 1) acoustic CFL
    c = sound_speed(rho_cells, p_cells, gamma)           # cell-centered
    dx = x_nodes[1:] - x_nodes[:-1]
    dt_sound = CFL * np.min(dx / (c + 1e-30))

    # 2) mesh-crossing limiter (prevents x_{i+1} < x_i after update)
    # cell is collapsing if u_left > u_right
    du = u_nodes[:-1] - u_nodes[1:]                     # per cell
    collapsing = du > 0.0
    if np.any(collapsing):
        dt_cross = CFL * np.min(dx[collapsing] / (du[collapsing] + 1e-30))
    else:
        dt_cross = np.inf
    return min(dt_sound, dt_cross)

def step_lagrangian(state: HydroState,
                    m_cells: np.ndarray,
                    geom: Geometry,
                    gamma: float,
                    sigma_visc: float,
                    bc_left: str = "outflow",
                    bc_right: str = "outflow",
                    dt: float = 1e-6) -> HydroState:
    """
    One time step implementing PDF Eqs.(11)-(19).
    """
    # (11) half-step velocity
    u_half = state.u + 0.5 * dt * state.a
    u_half = apply_velocity_bc(u_half, bc_left=bc_left, bc_right=bc_right) 

    # (12) update nodes
    x_new = state.x + dt * u_half

    # (13) new volumes
    V_new = cell_volumes(x_new, geom)
    if not np.all(V_new > 0):
        raise RuntimeError("Negative/zero cell volume encountered. Reduce dt or add crossing limiter.")

    # (14) new density
    rho_new = m_cells / V_new

    # (15) new viscosity
    q_new = artificial_viscosity(rho_new, u_half, sigma_visc)

    # (16) energy update (as written in PDF)
    dV = V_new - state.V
    num = state.e - 0.5 * (state.p + state.q + q_new) * (dV / m_cells)
    den = 1.0 + 0.5 * (gamma - 1.0) * rho_new * (dV / m_cells)
    e_new = num / den

    # (17) pressure EOS
    p_new = pressure_ideal_gas(rho_new, e_new, gamma)

    # (18) acceleration from new (p,q)
    a_new = compute_acceleration_nodes(x_new, p_new, q_new, m_cells, geom)

    # (19) full-step velocity
    u_new = u_half + 0.5 * dt * a_new

    return HydroState(
        t=state.t + dt,
        x=x_new,
        u=u_new,
        a=a_new,
        V=V_new,
        rho=rho_new,
        e=e_new,
        p=p_new,
        q=q_new
    )

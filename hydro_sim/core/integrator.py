import numpy as np
from .geometry import Geometry
from .eos import pressure_ideal_gas, sound_speed
from .viscosity import artificial_viscosity
from .grid import cell_volumes
from .state import HydroState
from .boundary import apply_velocity_bc, apply_pressure_bc

def compute_acceleration_nodes(x_nodes, p_cells, q_cells, m_cells, geom,
                               *, p_left=None, q_left=0.0, p_right=None, q_right=0.0):
    Nnodes = x_nodes.size
    a = np.zeros(Nnodes)
    pq = p_cells + q_cells

    # interior nodes 1..N-2
    left  = pq[:-1]
    right = pq[1:]
    denom = m_cells[:-1] + m_cells[1:]
    x_mid = x_nodes[1:-1]
    a[1:-1] = -2.0 * geom.beta * (x_mid ** geom.alpha) * (right - left) / denom

    # left boundary node
    if p_left is not None:
        a[0] = -2.0 * geom.beta * (x_nodes[0] ** geom.alpha) * (pq[0] - (p_left + q_left)) / (m_cells[0])
    else:
        a[0] = 0.0  # transmissive

    # right boundary node
    if p_right is not None:
        a[-1] = -2.0 * geom.beta * (x_nodes[-1] ** geom.alpha) * ((p_right + q_right) - pq[-1]) / (m_cells[-1])
    else:
        a[-1] = 0.0

    return a


def step_lagrangian(state: HydroState,
                    m_cells: np.ndarray,
                    geom: Geometry,
                    gamma: float,
                    sigma_visc: float,
                    bc_left="outflow",
                    bc_right="outflow",
                    dt: float = 1e-6) -> HydroState:
    """
    One time step implementing PDF Eqs.(11)-(19),
    with generalized boundary conditions.
    """

    t_half = state.t + 0.5 * dt

    # (11) half-step velocity
    u_half = state.u + 0.5 * dt * state.a
    # u_half = apply_velocity_bc(u_half, bc_left, bc_right, t_half)

    # (12) update nodes
    x_new = state.x + dt * u_half

    # (13) new volumes
    V_new = cell_volumes(x_new, geom)
    if not np.all(V_new > 0):
        raise RuntimeError("Negative/zero cell volume encountered. Reduce dt or add limiter.")

    # (14) new density
    rho_new = m_cells / V_new

    # (15) artificial viscosity
    q_new = artificial_viscosity(rho_new, u_half, sigma_visc)

    # (16) energy update
    dV = V_new - state.V
    num = state.e - 0.5 * (state.p + state.q + q_new) * (dV / m_cells)
    den = 1.0 + 0.5 * (gamma - 1.0) * rho_new * (dV / m_cells)
    e_new = num / den

    # (17) pressure EOS
    p_new = pressure_ideal_gas(rho_new, e_new, gamma)

    # (18) acceleration from new (p,q)
    p_left = bc_left["p"] if isinstance(bc_left, dict) and bc_left["type"]=="pressure" else None

    a_new = compute_acceleration_nodes(x_new, p_new, q_new, m_cells, geom, p_left=p_left)


    # (19) full-step velocity
    u_new = u_half + 0.5 * dt * a_new
    #  u_new = apply_velocity_bc(u_new, bc_left, bc_right, state.t + dt)

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

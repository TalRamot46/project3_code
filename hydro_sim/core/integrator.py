import numpy as np
from .geometry import Geometry
from .eos import pressure_ideal_gas, sound_speed
from .viscosity import artificial_viscosity
from .grid import cell_volumes
from .state import HydroState
from .boundary import apply_velocity_bc, apply_pressure_bc
import matplotlib.pyplot as plt

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
        a[0] = -2.0 * geom.beta * (x_nodes[0] ** geom.alpha) * (pq[0] - (p_left + q_left)) / (2 * m_cells[0])
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
    
    Supported boundary conditions:
        - "outflow": Zero-gradient (transmissive) boundary
        - "none": No boundary treatment (for Riemann problems)
        - "reflective": Reflecting wall (u=0 at boundary, for Sedov origin)
        - {"type": "pressure", "p": value}: Pressure-driven boundary
    """

    # (11) half-step velocity
    u_half = state.u + 0.5 * dt * state.a
    
    # Apply velocity boundary conditions at half-step
    u_half = _apply_velocity_bc_half(u_half, bc_left, bc_right)

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
    if not np.all(p_new >= 0):
        pass

    # (18) acceleration from new (p,q)
    # Determine boundary pressures for acceleration calculation
    p_left, p_right = _get_boundary_pressures(bc_left, bc_right, p_new, state.t + dt)
    a_new = compute_acceleration_nodes(x_new, p_new, q_new, m_cells, geom, 
                                        p_left=p_left, p_right=p_right)

    # (19) full-step velocity
    u_new = u_half + 0.5 * dt * a_new
    
    # Apply velocity boundary conditions at full step
    u_new = _apply_velocity_bc_full(u_new, bc_left, bc_right)

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


def _apply_velocity_bc_half(u, bc_left, bc_right):
    """Apply velocity boundary conditions at half-step."""
    u = u.copy()
    
    # Left boundary
    if bc_left == "reflective":
        # Reflective: velocity at origin is zero (symmetric/wall)
        u[0] = 0.0
    elif bc_left == "outflow":
        u[0] = u[1]
    elif bc_left == "none":
        pass
    elif isinstance(bc_left, dict):
        if bc_left["type"] == "pressure":
            # Pressure BC: don't constrain velocity at half-step
            pass
    
    # Right boundary
    if bc_right == "reflective":
        u[-1] = 0.0
    elif bc_right == "outflow":
        u[-1] = u[-2]
    elif bc_right == "none":
        pass
    elif isinstance(bc_right, dict):
        pass
    
    return u


def _apply_velocity_bc_full(u, bc_left, bc_right):
    """Apply velocity boundary conditions at full step."""
    u = u.copy()
    
    # Left boundary
    if bc_left == "reflective":
        u[0] = 0.0
    elif bc_left == "outflow":
        u[0] = u[1]
    elif bc_left == "none":
        pass
    elif isinstance(bc_left, dict):
        pass
    
    # Right boundary
    if bc_right == "reflective":
        u[-1] = 0.0
    elif bc_right == "outflow":
        u[-1] = u[-2]
    elif bc_right == "none":
        pass
    elif isinstance(bc_right, dict):
        pass
    
    return u


def _get_boundary_pressures(bc_left, bc_right, p_cells, t):
    """
    Extract boundary pressures for acceleration calculation.
    
    Returns:
        (p_left, p_right): Boundary pressures, or None for transmissive/none
    """
    p_left = None
    p_right = None
    
    # Left boundary
    if isinstance(bc_left, dict) and bc_left.get("type") == "pressure":
        p_left = bc_left["p"]
    elif bc_left == "reflective":
        # For reflective BC, use ghost cell with same pressure (dp/dn = 0)
        p_left = p_cells[0]
    
    # Right boundary
    if isinstance(bc_right, dict) and bc_right.get("type") == "pressure":
        p_right = bc_right["p"]
    elif bc_right == "reflective":
        p_right = p_cells[-1]
    
    return p_left, p_right

import numpy as np
from .geometry import Geometry
from .eos import pressure_ideal_gas, sound_speed
from .viscosity import artificial_viscosity
from .grid import cell_volumes
from .state import HydroState
from .boundary import apply_velocity_bc, apply_pressure_bc
from project3_code.rad_hydro_sim.plotting import mpl_style 
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
    num = state.e_material - 0.5 * (state.p + state.q + q_new) * (dV / m_cells)
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
    T_new = np.zeros_like(e_new)

    return HydroState(
        t=state.t + dt,
        x=x_new,
        u=u_new,
        a=a_new,
        V=V_new,
        rho=rho_new,
        e_material=e_new,
        p=p_new,
        q=q_new,
        m_cells=m_cells,
        T_material=T_new,
    )


def _apply_velocity_bc_half(u, bc_left, bc_right):
    """Apply velocity boundary conditions at half-step."""
    u = u.copy()
    
    # Left boundary
    if bc_left == "reflective":
        u[0] = 0.0
    elif bc_left in ("outflow", "vacuum") or isinstance(bc_left, (int, float)):
        u[0] = u[1]  # zero-gradient (outflow / pressure drive)
    elif bc_left in ("none", None):
        pass
    
    # Right boundary
    if bc_right == "reflective":
        u[-1] = 0.0
    elif bc_right in ("outflow", "vacuum") or isinstance(bc_right, (int, float)):
        u[-1] = u[-2]
    elif bc_right in ("none", None):
        pass
    
    return u


def _apply_velocity_bc_full(u, bc_left, bc_right):
    """Apply velocity boundary conditions at full step."""
    u = u.copy()
    
    # Left boundary
    if bc_left == "reflective":
        u[0] = 0.0
    elif bc_left in ("outflow", "vacuum") or isinstance(bc_left, (int, float)):
        u[0] = u[1]
    elif bc_left in ("none", None):
        pass
    
    # Right boundary
    if bc_right == "reflective":
        u[-1] = 0.0
    elif bc_right in ("outflow", "vacuum") or isinstance(bc_right, (int, float)):
        u[-1] = u[-2]
    elif bc_right in ("none", None):
        pass
    
    return u


def _get_boundary_pressures(bc_left, bc_right, p_cells, t):
    """
    Boundary pressures for acceleration: p_left, p_right.
    None = transmissive (outflow). Simple types only:

    bc_left / bc_right:
        None, "outflow", "none" -> None (transmissive)
        "vacuum", 0               -> 0.0  (ablation into vacuum at m=0)
        "reflective"              -> p_cells[0] or p_cells[-1]
        float                     -> that value (prescribed pressure)
    """
    def _resolve(bc, p_edge, side):
        if bc is None or bc in ("outflow", "none"):
            return None
        if bc in ("vacuum", 0) or (isinstance(bc, (int, float)) and bc == 0):
            return 0.0
        if bc == "reflective":
            return float(p_edge)
        if isinstance(bc, (int, float)):
            return float(bc)
        return None

    p_left = _resolve(bc_left, p_cells[0], "left")
    p_right = _resolve(bc_right, p_cells[-1], "right")
    return p_left, p_right

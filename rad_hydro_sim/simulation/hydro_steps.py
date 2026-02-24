import numpy as np

from project_3.hydro_sim.core.geometry import Geometry, planar
from project_3.hydro_sim.core.integrator import (
    _apply_velocity_bc_half,
    _apply_velocity_bc_full,
    _get_boundary_pressures,
    compute_acceleration_nodes,
)
from project_3.hydro_sim.core.state import RadHydroState
from project_3.hydro_sim.core.grid import cell_volumes
from project_3.hydro_sim.core.viscosity import artificial_viscosity
from project_3.hydro_sim.core.eos import pressure_ideal_gas
from project_3.hydro_sim.core.boundary import apply_velocity_bc, apply_pressure_bc
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase

def get_e_star_from_hydro(
    state: RadHydroState,
    geom: Geometry,
    r: float,
    sigma_visc: float,
    dt: float,
    bc_left="outflow",
    bc_right="outflow",
) -> RadHydroState:
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
    rho_new = state.m_cells / V_new

    # (15) artificial viscosity
    q_new = artificial_viscosity(rho_new, u_half, sigma_visc)

    # (16) energy update
    dV = V_new - state.V
    num = state.e - 0.5 * (state.p + state.q + q_new) * (dV / state.m_cells)
    den = 1.0 + 0.5 * r * rho_new * (dV / state.m_cells)
    e_star = num / den

    state_star = RadHydroState(
        t=state.t,
        x=x_new,
        u=u_half,
        a=state.a,  # acceleration will be updated after radiation step
        V=V_new,
        rho=rho_new,
        e=e_star,
        p=state.p,  # pressure will be updated after radiation step
        q=q_new,
        m_cells=state.m_cells,
        T=state.T,  # temperature will be updated after radiation step
        E_rad=state.E_rad # radiation energy will be updated after radiation step
    )

    return state_star

def update_nodes_from_pressure(state: RadHydroState, case: RadHydroCase, e_new, dt: float, bc_left="outflow", bc_right="outflow", t_old: float = 0.0) -> RadHydroState:
    # (17) pressure EOS
    p_new = pressure_ideal_gas(state.rho, e_new, gamma=case.r+1)

    # (18) acceleration from new (p,q)
    # Determine boundary pressures at the NEW time (t_old + dt)
    p_left_bc, p_right_bc = _get_boundary_pressures(bc_left, bc_right, p_new, t_old + dt)
    a_new = compute_acceleration_nodes(state.x, p_new, state.q, state.m_cells, planar(), 
                                        p_left=p_left_bc, p_right=p_right_bc)

    
    # (19) full-step velocity
    # state.u here is u_half from get_e_star_from_hydro
    u_new = state.u + 0.5 * dt * a_new  # Complete the leapfrog: u_half + 0.5*dt*a_new
    
    # Apply velocity boundary conditions at full step
    u_new = _apply_velocity_bc_full(u_new, bc_left, bc_right)

    # Time is advanced in the caller (step_rad_hydro) so we do not add dt here.
    new_state = RadHydroState(
        t=state.t,
        x=state.x,
        u=u_new,
        a=a_new, # acceleration updated with new pressure
        V=state.V,
        rho=state.rho,
        e=e_new,
        p=p_new, # pressure updated with new energy (after radiation step)
        q=state.q,
        m_cells=state.m_cells,
        T=state.T, # new_state.T already calculated in radiation step.
        E_rad=state.E_rad # new_state.E_rad already calculated in radiation step.
    )

    return new_state
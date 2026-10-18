from __future__ import annotations

import numpy as np

from hydro_sim.core.geometry import Geometry, planar
from hydro_sim.core.integrator import (
    _apply_velocity_bc_half,
    _apply_velocity_bc_full,
    _get_boundary_pressures,
    compute_acceleration_nodes,
)
from hydro_sim.core.state import RadHydroState
from hydro_sim.core.grid import cell_volumes
from hydro_sim.core.viscosity import artificial_viscosity
from hydro_sim.core.eos import pressure_ideal_gas
from hydro_sim.core.boundary import apply_velocity_bc, apply_pressure_bc
from rad_hydro_sim.problems.RadHydroCase import RadHydroCase

def get_e_star_from_hydro(
    state: RadHydroState,
    geom: Geometry,
    r: float,
    sigma_visc: float,
    dt: float,
    bc_left: str | float = "outflow",
    bc_right: str | float = "outflow",
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
    
    # # Apply velocity boundary conditions at half-step
    # u_half = _apply_velocity_bc_half(u_half, bc_left, bc_right)

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
    num = state.e_material - 0.5 * (state.p + state.q + q_new) * (dV / state.m_cells)
    den = 1.0 + 0.5 * r * rho_new * (dV / state.m_cells)
    e_star = num / den
    

    state_star = RadHydroState(
        t=state.t,
        x=x_new,
        u=u_half,
        a=state.a,
        V=V_new,
        rho=rho_new,
        e_material=e_star,
        p=state.p,
        q=q_new,
        m_cells=state.m_cells,
        T_material=state.T_material,
        T_rad=state.T_rad,
        E_rad=state.E_rad,
        F_rad=state.F_rad,
    )

    return state_star

def update_nodes_from_pressure(
    state: RadHydroState,
    case: RadHydroCase,
    new_e_material,
    dt: float,
    bc_left: str | float = "outflow",
    bc_right: str | float = "outflow",
    t_old: float = 0.0,
) -> RadHydroState:
    # (17) pressure EOS
    p_new = pressure_ideal_gas(state.rho, new_e_material, gamma=case.r+1)

    # (18) acceleration from new (p,q)
    # p_left=0 at m=0 for ablation (vacuum); pressure drive uses bc_left as prescribed value
    p_left_bc, p_right_bc = _get_boundary_pressures(bc_left, bc_right, p_new, t_old + dt)
    a_new = compute_acceleration_nodes(state.x, p_new, state.q, state.m_cells, planar(), 
                                        p_left=p_left_bc, p_right=p_right_bc)

    
    # (19) full-step velocity
    # state.u here is u_half from get_e_star_from_hydro
    u_new = state.u + 0.5 * dt * a_new  # Complete the leapfrog: u_half + 0.5*dt*a_new
    
    # Apply velocity boundary conditions at full step
    # u_new = _apply_velocity_bc_full(u_new, bc_left, bc_right)

    # Time is advanced in the caller (step_rad_hydro) so we do not add dt here.
    new_state = RadHydroState(
        t=state.t,
        x=state.x,
        u=u_new,
        a=a_new, # acceleration updated with new pressure
        V=state.V,
        rho=state.rho,
        e_material=new_e_material,
        p=p_new,
        q=state.q,
        m_cells=state.m_cells,
        T_material=state.T_material,
        T_rad=state.T_rad,
        E_rad=state.E_rad,
    )

    return new_state


# ---------------------------------------------------------------------------
# Phase B, plan item A1 (md/phase_b_diffusion_changes.md): an energy-
# conserving ("compatible") alternative to update_nodes_from_pressure, and the
# energy bookkeeping both schemes are measured with.  Nothing above changes.
# ---------------------------------------------------------------------------

def node_masses(m_cells: np.ndarray) -> np.ndarray:
    """Node masses implied by compute_acceleration_nodes (planar).

    Interior nodes carry half of each neighbour; the right node half of its
    cell.  The left node is accelerated by (p_left - P_0) / m_0, i.e. it
    carries the full first cell (not half), so it gets m_0 here.  Energy
    conservation holds for any node masses as long as the kinetic energy uses
    the same ones as the momentum update.
    """
    M = np.empty(m_cells.size + 1)
    M[0] = m_cells[0]
    M[1:-1] = 0.5 * (m_cells[:-1] + m_cells[1:])
    M[-1] = 0.5 * m_cells[-1]
    return M


def kinetic_energy(u_nodes: np.ndarray, m_cells: np.ndarray) -> float:
    return float(0.5 * np.sum(node_masses(m_cells) * u_nodes**2))


def boundary_forces(a: np.ndarray, p: np.ndarray, q: np.ndarray, m_cells: np.ndarray):
    """External boundary pressures implied by the node accelerations.

    Left node: M_0 a_0 = p_L - P_0;  right node: M_N a_N = P_{N-1} - p_R,
    with P = p + q.  Recovering p_L, p_R from the accelerations makes the work
    consistent with whatever the momentum update actually used (including the
    initial state, whose accelerations use p_left = p[0]).
    """
    M = node_masses(m_cells)
    P = p + q
    return M[0] * a[0] + P[0], P[-1] - M[-1] * a[-1]


def boundary_work(state_old: RadHydroState, state_new: RadHydroState, dt: float) -> float:
    """Work done on the slab by the boundary pressures during the step."""
    FL0, FR0 = boundary_forces(state_old.a, state_old.p, state_old.q, state_old.m_cells)
    FL1, FR1 = boundary_forces(state_new.a, state_new.p, state_new.q, state_new.m_cells)
    u_bar = 0.5 * (state_old.u + state_new.u)
    return float(dt * (u_bar[0] * 0.5 * (FL0 + FL1) - u_bar[-1] * 0.5 * (FR0 + FR1)))


def update_nodes_compatible(
    state_old: RadHydroState,
    state_star: RadHydroState,
    case: RadHydroCase,
    de_radiation: np.ndarray,
    dt: float,
    bc_left: str | float = "outflow",
    bc_right: str | float = "outflow",
    t_old: float = 0.0,
    tol: float = 1e-13,
    max_iter: int = 100,
) -> RadHydroState:
    """Close the step so that total energy is conserved to round-off.

    The leapfrog momentum update is the original one,
        u^{n+1} = u^{n+1/2} + dt/2 a^{n+1},   a^{n+1} from p^{n+1} = r rho e^{n+1},
    so M (u^{n+1} - u^n) = dt (F^n + F^{n+1}) / 2.  The internal energy is then
    updated with the same time-centred pressure and velocity,
        m_j (e^{n+1} - e^n) = -dt Pbar_j (ubar_{j+1} - ubar_j) + m_j de_radiation_j,
        Pbar = (P^n + P^{n+1}) / 2,  ubar = (u^n + u^{n+1}) / 2,
    which makes the cells' work equal the nodes' kinetic-energy change (summation
    by parts).  e^{n+1} enters p^{n+1}, so it is found by fixed-point iteration
    (contraction ~ (c_s dt / dx)^2).  de_radiation is the radiation-matter
    exchange computed from e* (as in the original scheme).
    """
    if case.geom.alpha != 0:
        raise NotImplementedError("hydro_energy='compatible' is implemented for planar geometry only.")
    m = state_star.m_cells
    P_old = state_old.p + state_old.q
    e_new = state_star.e_material + de_radiation           # the standard scheme's value
    for _ in range(max_iter):
        p_new = pressure_ideal_gas(state_star.rho, e_new, gamma=case.r + 1)
        p_left_bc, p_right_bc = _get_boundary_pressures(bc_left, bc_right, p_new, t_old + dt)
        a_new = compute_acceleration_nodes(state_star.x, p_new, state_star.q, m, planar(),
                                           p_left=p_left_bc, p_right=p_right_bc)
        u_new = state_star.u + 0.5 * dt * a_new
        u_bar = 0.5 * (state_old.u + u_new)
        P_bar = 0.5 * (P_old + p_new + state_star.q)
        e_next = state_old.e_material - dt * P_bar * (u_bar[1:] - u_bar[:-1]) / m + de_radiation
        change = np.max(np.abs(e_next - e_new)) / max(np.max(np.abs(e_next)), 1e-300)
        e_new = e_next
        if change <= tol:
            break
    else:
        raise RuntimeError("Compatible energy update did not converge; reduce dt.")

    # Final state, consistent with the converged energy.
    p_new = pressure_ideal_gas(state_star.rho, e_new, gamma=case.r + 1)
    p_left_bc, p_right_bc = _get_boundary_pressures(bc_left, bc_right, p_new, t_old + dt)
    a_new = compute_acceleration_nodes(state_star.x, p_new, state_star.q, m, planar(),
                                       p_left=p_left_bc, p_right=p_right_bc)
    return RadHydroState(
        t=state_star.t, x=state_star.x, u=state_star.u + 0.5 * dt * a_new, a=a_new,
        V=state_star.V, rho=state_star.rho, e_material=e_new, p=p_new, q=state_star.q,
        m_cells=m, T_material=state_star.T_material, T_rad=state_star.T_rad, E_rad=state_star.E_rad,
    )

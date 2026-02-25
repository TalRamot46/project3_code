from project_3.hydro_sim.core.geometry import Geometry
from project_3.rad_hydro_sim.simulation.hydro_steps import (
    get_e_star_from_hydro,
    update_nodes_from_pressure,
)
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.rad_hydro_sim.simulation.radiation_step import radiation_step, calculate_temperature_from_specific_energy
from project_3.hydro_sim.problems.simulation_config import SimulationConfig
from project_3.hydro_sim.core.state import RadHydroState
import numpy as np

S_PER_NS = 1e-9

def step_rad_hydro(state: RadHydroState, dt: float, case: RadHydroCase, config: SimulationConfig) -> RadHydroState:
    """
    Pseudo-code for Lagrangian step with radiation-hydro coupling:
    1. Perform half-step velocity update using current acceleration.
    2. Update all cell hydrodynamic variables (x, V, rho, e_star, p, m_cells) using half-step velocities.
    3. Take e_star, rho, and m_cells to perform radiation-matter coupling and get updated new_e_material and E_rad.
    4. Update pressure profile using new_e_material.
    """
    # Determine boundary conditions based on case
    if case.P0 is not None:
        # Pressure-driven boundary
        p_drive = case.P0 * ((state.t / S_PER_NS) ** case.tau) if state.t > 0 else 0.0
        bc_left = {"type": "pressure", "p": p_drive}
        bc_right = "outflow"
    else:
        bc_left = "outflow"
        bc_right = "outflow"
    
    if case.scenario == "hydro_only":
        state_star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, dt, bc_left, bc_right)
        new_e_material = state_star.e # Attempting to bypass the radiation step for now to isolate hydro behavior
        new_T = calculate_temperature_from_specific_energy(new_e_material, state_star.rho, case.f_HeV, case.gamma, case.mu)
        new_state = update_nodes_from_pressure(state_star, case, new_e_material, dt, bc_left, bc_right, t_old=state.t)
        new_E_rad = np.zeros_like(state_star.rho) 
        new_state.T, new_state.E_rad = new_T, new_E_rad
    elif case.scenario == "radiation_only":
        new_e_material, new_T, new_E_rad = radiation_step(state, dt, case)
        new_T = calculate_temperature_from_specific_energy(new_e_material, state.rho, case.f_HeV, case.gamma, case.mu)
        new_state = state._replace(e=new_e_material, T=new_T, E_rad=new_E_rad)
    elif case.scenario == "full_rad_hydro":
        state_star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, dt, bc_left, bc_right)
        new_e_material, new_T, new_E_rad = radiation_step(state_star, dt, case)
        new_T = calculate_temperature_from_specific_energy(new_e_material, state_star.rho, case.f_HeV, case.gamma, case.mu)
        new_state = update_nodes_from_pressure(state_star, case, new_e_material, dt, bc_left, bc_right, t_old=state.t)
        new_state.T, new_state.E_rad = new_T, new_E_rad
    new_state.t += dt
    return new_state
    


    

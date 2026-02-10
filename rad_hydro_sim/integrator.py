from project_3.hydro_sim.core.geometry import Geometry
from project_3.rad_hydro_sim.hydro_steps import (
    get_e_star_from_hydro,
    update_nodes_from_pressure,
)
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.rad_hydro_sim.radiation_step import radiation_step
from project_3.hydro_sim.problems.simulation_config import SimulationConfig
from project_3.hydro_sim.core.state import RadHydroState

def step_rad_hydro(state: RadHydroState, dt: float, rad_hydro_case: RadHydroCase, simulation_config: SimulationConfig) -> RadHydroState:
    """
    Pseudo-code for Lagrangian step with radiation-hydro coupling:
    1. Perform half-step velocity update using current acceleration.
    2. Update all cell hydrodynamic variables (x, V, rho, e_star, p, m_cells) using half-step velocities.
    3. Take e_star, rho, and m_cells to perform radiation-matter coupling and get updated new_e_material and E_rad.
    4. Update pressure profile using new_e_material.
    """
    state_star = get_e_star_from_hydro(state, state.geom, simulation_config.r, simulation_config.sigma_visc, dt)
    new_e_material, new_T, new_E_rad = radiation_step(state_star, dt, rad_hydro_case) # FIX!
    new_state = update_nodes_from_pressure(state, new_e_material, dt)
    new_state.T, new_state.E_rad = new_T, new_E_rad
    return new_state
    


    

from project_3.hydro_sim.core.geometry import Geometry
from project_3.rad_hydro_sim.hydro_steps import (
    get_e_star_from_hydro,
    update_nodes_from_pressure,
)
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.rad_hydro_sim.radiation_step import radiation_step, calculate_temperature_from_specific_energy
from project_3.hydro_sim.problems.simulation_config import SimulationConfig
from project_3.hydro_sim.core.state import RadHydroState
import numpy as np

def step_rad_hydro(state: RadHydroState, dt: float, case: RadHydroCase, config: SimulationConfig) -> RadHydroState:
    """
    Pseudo-code for Lagrangian step with radiation-hydro coupling:
    1. Perform half-step velocity update using current acceleration.
    2. Update all cell hydrodynamic variables (x, V, rho, e_star, p, m_cells) using half-step velocities.
    3. Take e_star, rho, and m_cells to perform radiation-matter coupling and get updated new_e_material and E_rad.
    4. Update pressure profile using new_e_material.
    """
    state_star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, dt)
    # new_e_material, new_T, new_E_rad = radiation_step(state_star, dt, case)
    new_e_material = state_star.e # Attempting to bypass the radiation step for now to isolate hydro behavior
    new_T = calculate_temperature_from_specific_energy(new_e_material, state_star.rho, case.f, case.gamma, case.mu)
    new_state = update_nodes_from_pressure(state, case, new_e_material, dt)
    new_E_rad = np.zeros_like(state_star.rho) 
    new_state.T, new_state.E_rad = new_T, new_E_rad
    return new_state
    


    

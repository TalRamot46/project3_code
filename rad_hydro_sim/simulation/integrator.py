from project3_code.hydro_sim.core.geometry import Geometry
from project3_code.rad_hydro_sim.simulation.hydro_steps import (
    get_e_star_from_hydro,
    update_nodes_from_pressure,
)
from project3_code.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project3_code.rad_hydro_sim.simulation.radiation_step import radiation_step, calculate_temperature_from_specific_energy
from project3_code.hydro_sim.problems.simulation_config import SimulationConfig
from project3_code.hydro_sim.core.state import RadHydroState
import numpy as np

S_PER_NS = 1e-9

def step_rad_hydro(
    state: RadHydroState, 
    dt: float, 
    case: RadHydroCase, 
    config: SimulationConfig,
    T_left: float,
) -> RadHydroState:
    """
    Pseudo-code for Lagrangian step with radiation-hydro coupling:
    1. Perform half-step velocity update using current acceleration.
    2. Update all cell hydrodynamic variables (x, V, rho, e_star, p, m_cells) using half-step velocities.
    3. Take e_star, rho, and m_cells to perform radiation-matter coupling and get updated new_e_material and E_rad.
    4. Update pressure profile using new_e_material.
    """
    # Subsonic ablation: p=0 at m=0 (ablated matter expands into vacuum).
    # Pressure drive: p_left = P0 * t^tau.
    if case.P0_Barye is not None:
        p_drive = case.P0_Barye * ((state.t / S_PER_NS) ** case.tau) if state.t > 0 else 0.0
        bc_left = p_drive
        bc_right = "outflow"
    else:
        bc_left = "vacuum"   # p=0 at m=0 (radiation/temperature drive)
        bc_right = "outflow"
    
    if case.scenario == "hydro_only":
        # Retry mechanism for stability: if hydro update fails due to
        # negative/zero cell volumes, reduce dt and retry a few times.
        retries = 5
        # CFL-like safeguard: cap dt so nodes don't cross (limit displacement
        # to at most half a cell width in a single step).
        dx = np.min(np.diff(state.x)) if len(state.x) > 1 else 1.0
        max_speed = max(np.max(np.abs(state.u)), 1e-12)
        cfl_dt = 0.5 * dx / max_speed
        attempt_dt = min(dt, float(cfl_dt))
        while True:
            try:
                state_star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, attempt_dt, bc_left, bc_right)
                break
            except RuntimeError as e:
                if retries <= 0:
                    raise
                retries -= 1
                attempt_dt *= 0.5
        dt_used = attempt_dt
        new_e_material = state_star.e_material
        new_T_material = calculate_temperature_from_specific_energy(new_e_material, state_star.rho, case.f_Kelvin, case.beta_Rosen, case.mu)
        new_state = update_nodes_from_pressure(state_star, case, new_e_material, dt, bc_left, bc_right, t_old=state.t)
        new_E_rad = np.zeros_like(state_star.rho) 
        new_T_rad = state_star.T_rad if state_star.T_rad is not None else np.zeros_like(state_star.rho)
        new_state.T_material = new_T_material
        new_state.T_rad = new_T_rad
        new_state.E_rad = new_E_rad
    elif case.scenario == "radiation_only":
        try:
            new_T_material, new_e_material, new_T_rad, new_E_rad, new_F = radiation_step(state, dt, case, T_left)
            new_state = state._replace(T_material=new_T_material, e_material=new_e_material, T_rad=new_T_rad, E_rad=new_E_rad)
        except ValueError as e:
            if "infs or NaNs" in str(e):
                # If the radiation solver fails due to NaNs/Infs, keep the state unchanged
                new_state = state._replace()
            else:
                raise
    elif case.scenario == "full_rad_hydro":
        state_star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, dt, bc_left, bc_right)
        new_T_material, new_e_material, new_T_rad, new_E_rad, new_F_rad = radiation_step(state_star, dt, case, T_left)
        new_state = update_nodes_from_pressure(state_star, case, new_e_material, dt, bc_left, bc_right, t_old=state.t)
        new_state.T_material = new_T_material
        new_state.T_rad = new_T_rad
        new_state.E_rad = new_E_rad
        new_state.F_rad = new_F_rad
    else:
        raise ValueError(f"Unknown scenario: {case.scenario}")
    new_state.t += dt
    return new_state
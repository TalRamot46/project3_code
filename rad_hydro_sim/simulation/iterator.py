import numpy as np
from tqdm import tqdm

from project_3.hydro_sim.core.eos import internal_energy_from_prho
from project_3.hydro_sim.core.geometry import planar
from project_3.hydro_sim.core.grid import cell_volumes, masses_from_initial_rho
from project_3.hydro_sim.core.integrator import compute_acceleration_nodes
from project_3.hydro_sim.core.state import RadHydroState
from project_3.hydro_sim.core.timestep import compute_dt_cfl, update_dt_relchange
from project_3.rad_hydro_sim.simulation.integrator import step_rad_hydro
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.rad_hydro_sim.plotting.RadHydroHistory import RadHydroHistory
from project_3.rad_hydro_sim.simulation.radiation_step import (
    calculate_temperature_from_specific_energy,
    a_Hev
)
from project_3.hydro_sim.problems.simulation_config import SimulationConfig

def initialize_problem(case: RadHydroCase, config: SimulationConfig) -> tuple:
    """Initialize the problem state based on the case and simulation type."""
    # Unpack parameters from case & config
    geom = planar()
    x_nodes = np.linspace(case.x_min, case.x_max, num=config.N + 1)  # Initial grid
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])  # Cell centers

    rho = np.zeros_like(x_cells)
    p = np.zeros_like(x_cells)
    u = np.zeros_like(x_nodes)
    e = np.zeros_like(x_cells)
    T = np.zeros_like(x_cells)
    E_rad= np.zeros_like(x_cells)

    if case.initial_condition == "temperature, density":
        # If initial condition is given in terms of temperature, convert to specific energy
        T = np.full_like(x_cells, case.T_initial)
        e = case.f * T**case.gamma * case.rho0**(-case.mu)
        rho = np.full_like(x_cells, case.rho0)
        p = (case.r + 1) * rho * e  # Ideal gas EOS
        E_rad=a_Hev * T**4
    elif case.initial_condition == "pressure, velocity, density":
        rho = np.full_like(x_cells, case.rho0)
        p = np.full_like(x_cells, case.p0)
        u = np.full_like(x_nodes, case.u0)
        e = internal_energy_from_prho(p, rho, case.r+1)
        T = calculate_temperature_from_specific_energy(e, rho, case.f, case.gamma, case.mu)  # Initial temperature from Rosen's model
        E_rad = np.zeros_like(x_cells)  # Assuming no initial radiation energy
        
    q = np.zeros_like(x_cells)

    # Initialize mass cells and node positions
    V = cell_volumes(x_nodes, geom)
    m_cells = masses_from_initial_rho(x_nodes, rho, geom)
    
    a = np.zeros_like(x_nodes) # Initial acceleration will be computed in a matching step.


    state = RadHydroState(t=0.0, x=x_nodes, u=u, a=a, V=V, rho=rho, e=e, p=p, q=q, m_cells=m_cells, T=T, E_rad=E_rad)
    return state

def simulate_rad_hydro(
    rad_hydro_case: RadHydroCase,
    simulation_config: SimulationConfig,
) -> tuple:
    """This method activates the rad-hydro integrator for a given problem case and simulation type. It initializes the problem, runs the time integration loop, and returns the final state and history."""
    # Unpack parameters from case & config
    # Initialize problem
    state = initialize_problem(rad_hydro_case, simulation_config)
    state.a = compute_acceleration_nodes(state.x, state.p, state.q, state.m_cells, rad_hydro_case.geom, p_left=state.p[0], p_right=state.p[-1])
    
    t_end = rad_hydro_case.t_end
    
    # ---- history buffers ----
    times = []
    x_history, m_history, RHOs, Ps, u_history, Es, T_history, E_rad_history = [], [], [], [], [], [], [], []

    def store_frame():
        x_cells = 0.5 * (state.x[:-1] + state.x[1:])
        u_cells = 0.5 * (state.u[:-1] + state.u[1:])
        m_coordinate = np.cumsum(state.m_cells)
        times.append(state.t)
        x_history.append(x_cells.copy())
        m_history.append(m_coordinate.copy())
        RHOs.append(state.rho.copy())
        Ps.append(state.p.copy())
        u_history.append(u_cells.copy())
        Es.append(state.e.copy())
        T_history.append(state.T.copy())
        E_rad_history.append(state.E_rad.copy())
    store_frame()
    dt_prev = np.inf
    step = 0
    
    # Main time integration loop
    with tqdm(total=t_end) as pbar:
        while state.t < t_end:
            # Adaptive timestep
            if step > 2:
                if rad_hydro_case.scenario == "hydro_only":
                    dt_cfl = compute_dt_cfl(state.x, state.u, state.rho, state.p, rad_hydro_case.r+1, simulation_config.CFL)
                    if np.isnan(dt_cfl):
                        dt_cfl = min(0.05 * t_end, dt_prev * 1.1, t_end - state.t, 1e-12)
                    dt = min(dt_cfl, 0.05 * t_end, dt_prev * 1.1, t_end - state.t)
                if rad_hydro_case.scenario == "radiation_only":
                    dt_rel = update_dt_relchange(dt_prev, state.E_rad, E_rad_history[-1], state.T, T_history[-1])
                    dt = min(dt_rel, 0.05 * t_end, dt_prev * 1.1, t_end - state.t, 1e-12)
                elif rad_hydro_case.scenario == "full_rad_hydro":
                    dt_cfl = compute_dt_cfl(state.x, state.u, state.rho, state.p, rad_hydro_case.r+1, simulation_config.CFL)
                    dt_rel = update_dt_relchange(dt_prev, state.E_rad, E_rad_history[-1], state.T, T_history[-1])
                    dt = min(dt_cfl, dt_rel, 0.05 * t_end, dt_prev * 1.1, t_end - state.t)
                    if np.isnan(dt):
                        dt = min(0.05 * t_end, dt_prev * 1.1, t_end - state.t, 1e-12)
            else:
                # Small initial timestep for stability
                dt = min(1e-13, 1e-6 * t_end, t_end - state.t)
            # print(dt)
            dt_prev = dt


            # Get boundary conditions for current state
            # bc_left, bc_right = _get_boundary_conditions(rad_hydro_case, simulation_config, state)
            
            # Lagrangian step
            new_state = step_rad_hydro(
                state, dt, rad_hydro_case, simulation_config
            )
            state = new_state

            # storing the new state
            step += 1
            if (step % simulation_config.store_every) == 0:
                store_frame()
            
            pbar.update(dt)

    # Ensure last frame stored
    if times[-1] != state.t:
        store_frame()

    history = RadHydroHistory(
        t=np.array(times),
        x=np.stack(x_history, axis=0),
        m=np.stack(m_history, axis=0),
        rho=np.stack(RHOs, axis=0),
        p=np.stack(Ps, axis=0),
        u=np.stack(u_history, axis=0),
        e=np.stack(Es, axis=0),
        T=np.stack(T_history, axis=0),
        E_rad=np.stack(E_rad_history, axis=0)
    )

    x_cells = 0.5 * (state.x[:-1] + state.x[1:])
    meta = dict(case=rad_hydro_case, sim_type=simulation_config, geometry=rad_hydro_case.geom)
    
    return x_cells, state, meta, history
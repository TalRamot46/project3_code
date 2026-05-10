import os
from functools import lru_cache
from typing import Any, Optional
from dataclasses import replace

import numpy as np
from tqdm import tqdm

from project3_code.menahem_new.subsonic_heat_wave import SubsonicHeatWave
from project3_code.hydro_sim.core.eos import internal_energy_from_prho
from project3_code.hydro_sim.core.geometry import planar
from project3_code.hydro_sim.core.grid import cell_volumes, masses_from_initial_rho
from project3_code.hydro_sim.core.integrator import compute_acceleration_nodes
from project3_code.hydro_sim.core.state import RadHydroState
from project3_code.hydro_sim.core.timestep import compute_dt_cfl, update_dt_relchange
from project3_code.rad_hydro_sim.simulation.integrator import step_rad_hydro
from project3_code.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project3_code.rad_hydro_sim.plotting.RadHydroHistory import RadHydroHistory
from project3_code.rad_hydro_sim.simulation.radiation_step import (
    calculate_temperature_from_specific_energy,
    a_Kelvin
)
from project3_code.hydro_sim.problems.simulation_config import SimulationConfig

def initialize_problem(case: RadHydroCase, config: SimulationConfig) -> RadHydroState:
    """Initialize the problem state based on the case and simulation type."""
    # Unpack parameters from case & config
    geom = planar()
    x_nodes = np.linspace(case.x_min, case.x_max, num=config.N + 1)  # Initial grid
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])  # Cell centers

    rho = np.zeros_like(x_cells)
    p = np.zeros_like(x_cells)
    u = np.zeros_like(x_nodes)
    e = np.zeros_like(x_cells)
    T_material = np.zeros_like(x_cells) # will be changed according to the initial condition!
    T_rad = np.zeros_like(x_cells)
    E_rad= np.zeros_like(x_cells)

    if case.initial_condition == "temperature, density":
        T_material = np.full_like(x_cells, case.T_initial_Kelvin)
        e = case.f_Kelvin * T_material**case.beta_Rosen * case.rho0**(-case.mu)
        rho = np.full_like(x_cells, case.rho0)
        p = (case.r + 1) * rho * e  # Ideal gas EOS
        T_rad = T_material.copy()
        E_rad = a_Kelvin * T_rad**4
    elif case.initial_condition == "pressure, velocity, density":
        rho = np.full_like(x_cells, case.rho0)
        p = np.full_like(x_cells, case.p0)
        u = np.full_like(x_nodes, case.u0)
        e = internal_energy_from_prho(p, rho, case.r+1)
        T_material = calculate_temperature_from_specific_energy(e, rho, case.f_Kelvin, case.beta_Rosen, case.mu)
        T_rad = T_material.copy()
        E_rad = a_Kelvin * T_rad**4
        
    q = np.zeros_like(x_cells)

    # Initialize mass cells and node positions
    V = cell_volumes(x_nodes, geom)
    m_cells = masses_from_initial_rho(x_nodes, rho, geom)
    
    a = np.zeros_like(x_nodes)

    state = RadHydroState(
        t=0.0, x=x_nodes, u=u, a=a, V=V, rho=rho,
        e_material=e, p=p, q=q, m_cells=m_cells,
        T_material=T_material, T_rad=T_rad, E_rad=E_rad,
    )
    return state

def initialize_bc(case: RadHydroCase) -> float:
    bc_type = getattr(case, "bc_type", "Dirichlet")  # default to "Dirichlet" if not specified
    if bc_type == "Marshak":
        # Use the improved bath temperature calculation based on subsonic heat wave
        T_bath = get_T_bath(case, time=0.0)
        T_left = T_bath
    else:
        # Use the simple time-dependent boundary temperature
        t_drive = 0.0 if case.T0_Kelvin is None else 1e-9
        T0_left = case.T0_Kelvin if case.T0_Kelvin is not None else 0.0
        T_left = T0_left * (t_drive/(10**-9))**case.tau if t_drive > 0 else 0.0
    
    return T_left

def _build_mass_grid_uniform(case, omega: float, num_cells: int) -> np.ndarray:
    """Build uniform Lagrangian mass grid."""
    coordinate = np.array(list(sorted(set(
        list(np.linspace(0., float(case.x_max), num_cells+1))
    ))))
    dx = coordinate[1:] - coordinate[:-1]
    density = case.rho0 / (1.-omega) * (coordinate[1:]**(1.-omega) - coordinate[:-1]**(1.-omega))/(coordinate[1:] - coordinate[:-1])
    mass_cells = density * dx
    mass = np.cumsum(mass_cells)
    mass = np.array([1e-30, 1e-7*mass[0]] + list(mass))
    return mass


def _subsonic_heat_wave_cache_key(case: RadHydroCase) -> tuple[float, float, float, float, float, float, float, float, float]:
    return (
        float(case.T0_Kelvin if case.T0_Kelvin is not None else 0.0),
        float(case.tau),
        float(case.g_Kelvin),
        float(case.alpha),
        float(case.lambda_),
        float(case.f_Kelvin),
        float(case.beta_Rosen),
        float(case.mu),
        float(case.r),
    )


@lru_cache(maxsize=None)
def _get_subsonic_heat_wave_solver(cache_key: tuple[float, float, float, float, float, float, float, float, float]) -> SubsonicHeatWave:
    Tb, tau, g, alpha, lambdap, f, beta, mu, r = cache_key
    solver = SubsonicHeatWave(
        Tb=Tb,
        tau=tau,
        g=g,
        alpha=alpha,
        lambdap=lambdap,
        f=f,
        beta=beta,
        mu=mu,
        gamma=r + 1.0,
    )
    solver.find_xsi_f()
    return solver

def get_T_bath(case, time: float) -> float:
    """Compute bath temperature using only the SubsonicHeatWave dimensionless flux S.

    This function creates a `SubsonicHeatWave` solver, finds the correct
    self-similar front, evaluates the dimensionless profiles on a Lagrangian
    mass grid, extracts the dimensionless boundary flux `S[0]` and converts
    it to the physical bath temperature via
    `calc_T_bath_from_dimensionless_boundary_flux`.
    """
    solver = _get_subsonic_heat_wave_solver(_subsonic_heat_wave_cache_key(case))
 
    # build a smallLagrangian mass grid and evaluate the self-similar profiles
    mass = _build_mass_grid_uniform(case, omega=0.0, num_cells=200)
    t_eval = max(float(time), 1e-300)
    xsi_vec = mass * solver.xsi_over_m(time=t_eval)

    profiles = solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    S = profiles.get("S", None)
    dimensionless_boundary_flux = float(S[0]) if (S is not None and len(S) > 0) else 0.0

    T_bath = solver.calc_T_bath_from_dimensionless_boundary_flux(
        dimensionless_boundary_flux=dimensionless_boundary_flux,
        time=t_eval,
    )
    return float(T_bath)

def simulate_rad_hydro(
    rad_hydro_case: RadHydroCase,
    simulation_config: SimulationConfig,
    *,
    tqdm_line: int | None = None,
    tqdm_desc: str | None = None,
    mp_progress: Any = None,
    mp_progress_index: int | None = None,
) -> tuple:
    """This method activates the rad-hydro integrator for a given problem case and simulation type. It initializes the problem, runs the time integration loop, and returns the final state and history.

    If ``tqdm_line`` is set (e.g. sweep index in parallel runs), the progress bar uses that
    row ``position`` so multiple bars do not overwrite the same terminal line.

    If ``mp_progress`` and ``mp_progress_index`` are set (e.g. ``multiprocessing.Array``),
    each step writes ``min(1, state.t / t_end)`` into that slot for a parent-process aggregate
    bar; the in-process tqdm is disabled in that mode.
    """
    # Unpack parameters from case & config
    # Initialize problem
    state = initialize_problem(rad_hydro_case, simulation_config)
    state.a = compute_acceleration_nodes(state.x, state.p, state.q, state.m_cells, rad_hydro_case.geom, p_left=state.p[0], p_right=state.p[-1])
    T_left = initialize_bc(rad_hydro_case)
    # Create a new case object with T_left set
    rad_hydro_case = replace(rad_hydro_case, T_left=T_left)

    t_end = rad_hydro_case.t_sec_end
    
    # ---- history buffers ----
    times = []
    x_history, m_history = [], []
    RHOs, Ps, u_history, Es = [], [], [], []
    T_material_history, T_rad_history, E_rad_history = [], [], []

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
        Es.append(state.e_material.copy())
        T_material_history.append(state.T_material.copy() if state.T_material is not None else np.zeros_like(state.rho))
        T_rad_history.append(state.T_rad.copy() if state.T_rad is not None else np.zeros_like(state.rho))
        E_rad_history.append(state.E_rad.copy() if state.E_rad is not None else np.zeros_like(state.rho))
    store_frame()
    dt_prev = np.inf
    step = 0

    use_mp_progress = mp_progress is not None and mp_progress_index is not None
    if use_mp_progress:
        assert mp_progress_index is not None
        prog_slot = int(mp_progress_index)
    else:
        prog_slot = -1

    # Main time integration loop
    pbar_kw: dict = {"total": t_end}
    if use_mp_progress:
        pbar_kw["disable"] = True
    elif os.environ.get("TQDM_DISABLE", "0") == "1":
        pbar_kw["disable"] = True
    if tqdm_line is not None and not use_mp_progress:
        pbar_kw["position"] = int(tqdm_line)
        pbar_kw["leave"] = True
        pbar_kw["mininterval"] = 0.25
        if tqdm_desc:
            pbar_kw["desc"] = tqdm_desc
    with tqdm(**pbar_kw) as pbar:
        while state.t < t_end:
            # Adaptive timestep
            dt = 0.0
            if step > 2:
                if rad_hydro_case.scenario == "hydro_only":
                    dt_cfl = compute_dt_cfl(state, rad_hydro_case.r+1, simulation_config.CFL)
                    if np.isnan(dt_cfl):
                        dt_cfl = min(0.05 * t_end, dt_prev * 1.1, t_end - state.t, 1e-12)
                    dt = min(dt_cfl, 0.05 * t_end, dt_prev * 1.1, t_end - state.t)
                elif rad_hydro_case.scenario == "radiation_only":
                    dt_rel = update_dt_relchange(dt_prev, state.E_rad, E_rad_history[-1], state.T_rad, T_rad_history[-1])
                    dt = min(dt_rel, 0.05 * t_end, dt_prev * 1.1, t_end - state.t, 1e-12)
                elif rad_hydro_case.scenario == "full_rad_hydro":
                    dt_cfl = compute_dt_cfl(state, rad_hydro_case.r+1, simulation_config.CFL)
                    dt_rel = update_dt_relchange(dt_prev, state.E_rad, E_rad_history[-1], state.T_rad, T_rad_history[-1])
                    dt = min(dt_cfl, dt_rel, 0.05 * t_end, dt_prev * 1.1, t_end - state.t)
                    if step % 1000 == 0:
                        pass

                    if np.isnan(dt):
                        dt = min(0.05 * t_end, dt_prev * 1.1, t_end - state.t, 1e-12)
            else:
                # Small initial timestep for stability
                dt = min(1e-13, 1e-6 * t_end, t_end - state.t)
            # print(dt)
            dt_prev = dt   # pyright: ignore[reportPossiblyUnboundVariable]

            # Get boundary conditions for current state
            # bc_left, bc_right = _get_boundary_conditions(rad_hydro_case, simulation_config, state)
            
            # Recalculate T_left for current time step for Marshak BC
            if rad_hydro_case.bc_type == "Marshak":
                T_left = get_T_bath(rad_hydro_case, time=state.t)
                # Update the case with the new T_left
                rad_hydro_case = replace(rad_hydro_case, T_left=T_left)
            
            # Lagrangian step
            new_state = step_rad_hydro(
                state, dt, rad_hydro_case, simulation_config, T_left
            )
            # Progress bar: use actual time advanced (step_rad_hydro may advance by more than dt in some code paths)
            pbar.update(new_state.t - state.t)
            state = new_state
            if use_mp_progress:
                mp_progress[prog_slot] = min(1.0, float(state.t) / float(t_end))

            # storing the new state
            step += 1
            if (step % simulation_config.store_every) == 0:
                store_frame()

            # if state.t > 0.5e-9:
            #     import matplotlib.pyplot as plt
            #     plt.plot(state.m_cells, state.rho)
            #     plt.plot(state.m_cells, state.q)
            #     plt.show()

        if use_mp_progress:
            mp_progress[prog_slot] = 1.0

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
        T=np.stack(T_rad_history, axis=0),
        E_rad=np.stack(E_rad_history, axis=0),
        T_material=np.stack(T_material_history, axis=0),
    )

    x_cells = 0.5 * (state.x[:-1] + state.x[1:])
    meta = dict(case=rad_hydro_case, sim_type=simulation_config, geometry=rad_hydro_case.geom)
    
    return x_cells, state, meta, history

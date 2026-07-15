import csv
import os
from functools import lru_cache
from typing import Any, Optional, Tuple
from dataclasses import replace

import numpy as np
from tqdm import tqdm

from project3_code.menahem_new.subsonic_heat_wave_og import SubsonicHeatWave
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
    a_Kelvin,
    c as radiation_c,
)
from project3_code.rad_hydro_sim.output_paths import get_rad_hydro_results_dir
from project3_code.hydro_sim.problems.simulation_config import SimulationConfig
# from project3_code.verification import _build_mass_grid

def initialize_problem(case: RadHydroCase, config: SimulationConfig) -> RadHydroState:
    """Initialize the problem state based on the case and simulation type."""
    # Unpack parameters from case & config
    geom = planar()
    omega = case.omega
    assert omega < 1.0, "Error: omega must be less than 1.0 to avoid singularity at x=0"

    # Option B: Uniform spatial grid (dx = const) with exact cell mass integration
    x_nodes = np.linspace(case.x_min, case.x_max, num=config.N + 1)
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    dx = x_nodes[1:] - x_nodes[:-1]

    if omega != 1.0:
        m_cells_analytic = (case.rho0 / (1.0 - omega)) * (x_nodes[1:]**(1.0 - omega) - x_nodes[:-1]**(1.0 - omega))
    else:
        x_nodes_non_zero = np.where(x_nodes == 0.0, 1e-30, x_nodes)
        m_cells_analytic = case.rho0 * np.log(x_nodes_non_zero[1:] / x_nodes_non_zero[:-1])

    rho = m_cells_analytic / dx
    p = np.zeros_like(x_cells)
    u = np.zeros_like(x_nodes)
    e = np.zeros_like(x_cells)
    T_material = np.zeros_like(x_cells) # will be changed according to the initial condition!
    T_rad = np.zeros_like(x_cells)
    E_rad= np.zeros_like(x_cells)

    if case.initial_condition == "temperature, density":
        assert case.T_initial_Kelvin is not None
        assert case.rho0 is not None
        T_material = np.full_like(x_cells, case.T_initial_Kelvin)
        e = case.f_Kelvin * T_material**case.beta_Rosen * rho**(-case.mu)
        p = (case.r + 1) * rho * e  # Ideal gas EOS
        T_rad = T_material.copy()
        E_rad = a_Kelvin * T_rad**4
    elif case.initial_condition == "pressure, velocity, density":
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
        Tb=Tb * (1./1e-9)**tau,
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

def get_dimensionless_boundary_flux(case) -> Tuple[SubsonicHeatWave, float]:
    """Compute dimensionless boundary flux using only the SubsonicHeatWave dimensionless flux S.

    This function creates a `SubsonicHeatWave` solver, finds the correct
    self-similar front, evaluates the dimensionless profiles on a Lagrangian
    mass grid, extracts the dimensionless boundary flux `S[0]` and converts
    it to the physical bath temperature via
    `calc_T_bath_from_dimensionless_boundary_flux`.
    """
    solver = _get_subsonic_heat_wave_solver(_subsonic_heat_wave_cache_key(case))
    assert solver.xsi_f != None
    assert solver.Pf != None

    xsi_vec = solver.get_xsi_grid(xsi_f=solver.xsi_f, fac=20.)    
    result= solver.get_self_similar_profiles(xsi_vec=xsi_vec)
    V, P, U, S = result["V"], result["P"], result["U"], result["S"]
    if S is None:
        dimensionless_boundary_flux = 0.0
    else:
        S_arr = np.asarray(S)
        dimensionless_boundary_flux = float(S_arr[0]) if S_arr.size > 0 else 0.0
    return solver, dimensionless_boundary_flux

def check_marshak(E_rad: np.ndarray, T_bath: float, F0_solver: float, F0_sim: float) -> dict[str, float]:
    """Return Marshak boundary quantities for monitoring."""
    E_bath = a_Kelvin * T_bath**4
    E_rad0 = float(E_rad[0]) if len(E_rad) > 0 else np.nan
    rhs = 0.5 * radiation_c * (E_bath - E_rad0)
    residual_solver = F0_solver - rhs
    residual_sim = F0_sim - rhs
    residual_solver_pct = 100.0 * abs(residual_solver) / max(abs(rhs), 1e-300)
    residual_sim_pct = 100.0 * abs(residual_sim) / max(abs(rhs), 1e-300)
    return {
        "F0_solver": float(F0_solver),
        "F0_sim": float(F0_sim),
        "E_bath": float(E_bath),
        "E_rad0": E_rad0,
        "residual_solver_pct": float(residual_solver_pct),
        "residual_sim_pct": float(residual_sim_pct),
    }


class MarshakMonitor:
    """Helper context manager for optional logging of Marshak boundary condition metrics."""

    def __init__(
        self,
        case_name: str,
        enabled: bool = False,
        full_diagnostics: bool = False,
    ):
        self.enabled = enabled
        self.full_diagnostics = full_diagnostics
        self._file = None
        self._writer = None

        if self.enabled:
            monitor_dir = get_rad_hydro_results_dir() / "monitor"
            monitor_dir.mkdir(parents=True, exist_ok=True)
            safe_case_name = (
                case_name.replace(" ", "_")
                .replace("=", "")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace(":", "")
                .replace("$", "")
                .replace("\\", "")
                .replace("/", "_")
            )
            monitor_path = monitor_dir / f"{safe_case_name}_marshak_monitor.csv"
            self._file = open(monitor_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)

            if self.full_diagnostics:
                self._writer.writerow([
                    "time",
                    "T_left",
                    "T_surface",
                    "F0_solver",
                    "F0_sim",
                    "E_bath",
                    "E_rad0",
                    "residual_solver_pct",
                    "residual_sim_pct",
                    "LHS",
                ])
            else:
                self._writer.writerow(["time", "T_bath", "T_surface"])

    def record(
        self,
        t: float,
        T_bath: float,
        T_surface: float,
        *,
        state: Optional[RadHydroState] = None,
        solver: Optional[Any] = None,
        dimensionless_boundary_flux: float = 0.0,
        LHS: float = 0.0,
    ) -> None:
        if not self.enabled or self._writer is None or self._file is None:
            return

        if self.full_diagnostics and state is not None and solver is not None:
            F0_solver, _, _, _, _, _ = solver.get_boundary_flux(
                dimensionless_boundary_flux=dimensionless_boundary_flux, time=t
            )
            F0_sim = state.F_rad[1] if state.F_rad is not None and len(state.F_rad) > 0 else 0.0
            marshak = check_marshak(state.E_rad, T_bath, F0_solver, F0_sim)
            self._writer.writerow([
                f"{t:.16e}",
                f"{T_bath:.16e}",
                f"{T_surface:.16e}",
                f"{marshak['F0_solver']:.16e}",
                f"{marshak['F0_sim']:.16e}",
                f"{marshak['E_bath']:.16e}",
                f"{marshak['E_rad0']:.16e}",
                f"{marshak['residual_solver_pct']:.16e}",
                f"{marshak['residual_sim_pct']:.16e}",
                f"{LHS:.16e}",
            ])
        else:
            self._writer.writerow([
                f"{t:.16e}",
                f"{T_bath:.16e}",
                f"{T_surface:.16e}",
            ])
        self._file.flush()

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def compute_adaptive_dt(
    step: int,
    t: float,
    t_end: float,
    dt_prev: float,
    state: RadHydroState,
    scenario: str,
    CFL: float,
    gamma: float,
    E_rad_last: np.ndarray,
    T_rad_last: np.ndarray,
) -> float:
    """Compute adaptive timestep for hydro, radiation, or coupled rad-hydro simulation steps."""
    if step <= 2:
        return min(1e-13, 1e-6 * t_end, t_end - t)

    if scenario == "hydro_only":
        dt_cfl = compute_dt_cfl(state, gamma, CFL)
        if np.isnan(dt_cfl):
            dt_cfl = min(0.05 * t_end, dt_prev * 1.1, t_end - t, 1e-12)
        dt = min(dt_cfl, 0.05 * t_end, dt_prev * 1.1, t_end - t)
    elif scenario == "radiation_only":
        dt_rel = update_dt_relchange(dt_prev, state.E_rad, E_rad_last, state.T_rad, T_rad_last)
        dt = min(dt_rel, 0.05 * t_end, dt_prev * 1.1, t_end - t, 1e-12)
    elif scenario == "full_rad_hydro":
        dt_cfl = compute_dt_cfl(state, gamma, CFL)
        dt_rel = update_dt_relchange(dt_prev, state.E_rad, E_rad_last, state.T_rad, T_rad_last)
        dt = min(dt_cfl, dt_rel, 0.05 * t_end, dt_prev * 1.1, t_end - t)
        if np.isnan(dt):
            dt = min(0.05 * t_end, dt_prev * 1.1, t_end - t, 1e-12)
    else:
        dt = min(0.05 * t_end, dt_prev * 1.1, t_end - t, 1e-12)

    return dt


def compute_boundary_temperatures(
    state: RadHydroState,
    rad_hydro_case: RadHydroCase,
    solver: Optional[SubsonicHeatWave] = None,
) -> Tuple[float, float, float, RadHydroCase]:
    """Compute boundary conditions and surface temperatures for current simulation step.

    Returns (T_left, T_surface, T_bath, updated_rad_hydro_case).
    """
    T_bath = 0.0

    if rad_hydro_case.bc_type == "Marshak":
        assert solver is not None, "SubsonicHeatWave solver required for Marshak BC"
        T_bath = solver.Tbath(time=state.t + 1e-20)
        T_surface = (state.E_rad[0] / a_Kelvin) ** 0.25
        T_left = T_bath
        updated_case = replace(rad_hydro_case, T_left=T_left)
    else: # Dirichlet
        if rad_hydro_case.T0_Kelvin is None or rad_hydro_case.T0_Kelvin == 0.0:
            T_surface = 0.0
        else:
            t_ns = state.t / 1e-9
            if t_ns == 0.0:
                T_surface = 0.0
            else:
                T_surface = rad_hydro_case.T0_Kelvin * (t_ns) ** rad_hydro_case.tau
        T_left = T_surface
        updated_case = rad_hydro_case

    return T_left, T_surface, T_bath, updated_case


def simulate_rad_hydro(
    rad_hydro_case: RadHydroCase,
    simulation_config: SimulationConfig,
    *,
    tqdm_line: int | None = None,
    tqdm_desc: str | None = None,
    mp_progress: Any = None,
    mp_progress_index: int | None = None,
    enable_monitoring: bool = False,
    full_monitoring: bool = False,
) -> tuple:
    """Activates the rad-hydro integrator for a given problem case and simulation type.

    Initializes the problem, executes the time integration loop, and returns final state and history.

    Parameters:
        rad_hydro_case: Physical case specification.
        simulation_config: Solver configuration parameters.
        tqdm_line: Line offset for multi-progress bars.
        tqdm_desc: Description for progress bar.
        mp_progress: Multiprocessing array for parallel progress updates.
        mp_progress_index: Index slot in mp_progress array.
        enable_monitoring: Flag to enable Marshak boundary monitoring file export (default False).
        full_monitoring: Flag to compute full diagnostics (residuals, fluxes) when monitoring is enabled.
    """
    state = initialize_problem(rad_hydro_case, simulation_config)
    state.a = compute_acceleration_nodes(
        state.x, state.p, state.q, state.m_cells, rad_hydro_case.geom, p_left=state.p[0], p_right=state.p[-1]
    )
    t_end = rad_hydro_case.t_sec_end

    solver = None
    dimensionless_boundary_flux = 0.0
    if rad_hydro_case.bc_type in ["Marshak", "Marshak_Menahem"]:
        solver, dimensionless_boundary_flux = get_dimensionless_boundary_flux(rad_hydro_case)

    case_name = rad_hydro_case.title or rad_hydro_case.scenario or "rad_hydro_run"

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
    prog_slot = int(mp_progress_index) if use_mp_progress and mp_progress_index is not None else -1

    pbar_kw: dict = {"total": t_end}
    if use_mp_progress or os.environ.get("TQDM_DISABLE", "0") == "1":
        pbar_kw["disable"] = True
    if tqdm_line is not None and not use_mp_progress:
        pbar_kw["position"] = int(tqdm_line)
        pbar_kw["leave"] = True
        pbar_kw["mininterval"] = 0.25
        if tqdm_desc:
            pbar_kw["desc"] = tqdm_desc

    with MarshakMonitor(case_name, enabled=enable_monitoring, full_diagnostics=full_monitoring) as monitor, tqdm(**pbar_kw) as pbar:
        while state.t < t_end:
            dt = compute_adaptive_dt(
                step=step,
                t=state.t,
                t_end=t_end,
                dt_prev=dt_prev,
                state=state,
                scenario=rad_hydro_case.scenario,
                CFL=simulation_config.CFL,
                gamma=rad_hydro_case.r + 1.0,
                E_rad_last=E_rad_history[-1],
                T_rad_last=T_rad_history[-1],
            )
            dt_prev = dt

            T_left, T_surface, T_bath, current_case = compute_boundary_temperatures(
                state=state, rad_hydro_case=rad_hydro_case, solver=solver
            )

            new_state, LHS = step_rad_hydro(state, dt, current_case, simulation_config, T_left)

            monitor.record(
                t=state.t,
                T_bath=T_bath,
                T_surface=T_surface,
                state=state,
                solver=solver,
                dimensionless_boundary_flux=dimensionless_boundary_flux,
                LHS=LHS,
            )

            pbar.update(new_state.t - state.t)
            state = new_state
            if use_mp_progress:
                mp_progress[prog_slot] = min(1.0, float(state.t) / float(t_end))

            step += 1
            if (step % simulation_config.store_every) == 0:
                store_frame()

        if use_mp_progress:
            mp_progress[prog_slot] = 1.0

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


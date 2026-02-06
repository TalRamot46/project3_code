"""
Unified Lagrangian hydrodynamics simulation.

Supports:
  - Riemann shock tube problems (transmissive BCs)
  - Driven shock problems (pressure-driven left BC)
  - Sedov-Taylor point explosions (reflective origin BC, spherical/cylindrical geometry)

All problem types use the same integration loop with configurable boundary conditions.
"""
from tqdm import tqdm
from core.geometry import planar, spherical, cylindrical, Geometry
from core.integrator import step_lagrangian, compute_acceleration_nodes
from core.timestep import compute_dt_cfl
from problems.driven_shock_problem import init_planar_driven_shock_case, DrivenShockCase
from problems.riemann_problem import init_planar_riemann_case, RiemannCase
from problems.sedov_problem import init_sedov_explosion, SedovExplosionCase
from simulations.riemann_exact import sample_solution
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, Callable


class SimulationType(str, Enum):
    """Supported simulation types."""
    RIEMANN = "riemann"
    DRIVEN_SHOCK = "driven_shock"
    SEDOV = "sedov"


@dataclass
class SimulationHistory:
    """
    Time history of simulation fields (for all simulation types).
    
    Attributes:
        t: Time values at each snapshot (K,)
        x: Cell centers at each snapshot (K, Ncells)
        m: Mass coordinate at each snapshot (K, Ncells)
        rho: Density at each snapshot (K, Ncells)
        p: Pressure at each snapshot (K, Ncells)
        u: Cell-centered velocity at each snapshot (K, Ncells)
        e: Internal energy at each snapshot (K, Ncells)
    """
    t: np.ndarray          # (K,)
    x: np.ndarray          # (K, Ncells) cell centers
    m: np.ndarray          # (K, Ncells) mass coordinate
    rho: np.ndarray        # (K, Ncells)
    p: np.ndarray          # (K, Ncells)
    u: np.ndarray          # (K, Ncells)
    e: np.ndarray          # (K, Ncells)


# Alias for backward compatibility
ShockHistory = SimulationHistory


def _determine_geometry(case, sim_type: SimulationType) -> Geometry:
    """
    Determine the appropriate geometry based on simulation type and case.
    
    For Sedov problems, infers geometry from case title or uses spherical as default.
    """
    if sim_type == SimulationType.SEDOV:
        title_lower = case.title.lower() if case.title else ""
        if "cylindrical" in title_lower:
            return cylindrical()
        elif "planar" in title_lower:
            return planar()
        else:
            # Default Sedov is spherical
            return spherical()
    else:
        # Riemann and Driven Shock are planar
        return planar()


def _get_boundary_conditions(
    case,
    sim_type: SimulationType,
    state,
) -> tuple:
    """
    Get boundary conditions based on simulation type.
    
    Returns:
        (bc_left, bc_right) tuple for step_lagrangian
    """
    if sim_type == SimulationType.RIEMANN:
        # Transmissive boundaries on both sides
        return "none", "none"
    
    elif sim_type == SimulationType.DRIVEN_SHOCK:
        # Pressure-driven left, outflow right
        t = state.t
        if hasattr(case, 'P0') and hasattr(case, 'tau'):
            p_drive = case.P0 * (t ** case.tau) if t > 0 else 0.0
        else:
            p_drive = 0.0
        return {"type": "pressure", "p": p_drive}, "outflow"
    
    elif sim_type == SimulationType.SEDOV:
        # Reflective at origin (r=0), outflow at far boundary
        # For Sedov: u=0 at r=0 (reflective/symmetric), free at outer boundary
        return "reflective", "outflow"
    
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")


def _initialize_problem(
    case,
    sim_type: SimulationType,
    geom: Geometry,
    gamma: float,
    Ncells: int,
) -> tuple:
    """
    Initialize state and masses for the given problem type.
    
    Returns:
        (state, m_cells, x_nodes)
    """
    if sim_type == SimulationType.RIEMANN:
        x_nodes = np.linspace(case.x_min, case.x_max, Ncells + 1)
        x0 = getattr(case, 'x0', 0.0)
        state, m_cells = init_planar_riemann_case(x_nodes, geom, gamma, case, x0=x0)
        
    elif sim_type == SimulationType.DRIVEN_SHOCK:
        x_nodes = np.linspace(case.x_min, case.x_max, Ncells + 1)
        state, m_cells = init_planar_driven_shock_case(x_nodes, geom, gamma, case)
        
    elif sim_type == SimulationType.SEDOV:
        # For Sedov, start slightly away from r=0 to avoid singularity
        r_min = case.x_min if case.x_min > 0 else 1e-6 * case.x_max
        x_nodes = np.linspace(r_min, case.x_max, Ncells + 1)
        state, m_cells = init_sedov_explosion(x_nodes, geom, gamma, case)
        
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")
    
    return state, m_cells, x_nodes


def simulate_lagrangian(
    case: Union[RiemannCase, DrivenShockCase, SedovExplosionCase],
    sim_type: SimulationType,
    *,
    Ncells: int,
    gamma: float,
    CFL: float,
    sigma_visc: float,
    store_every: int = 1000,
    geom: Optional[Geometry] = None,
) -> tuple:
    """
    Unified Lagrangian hydrodynamics simulation.
    
    This is the main entry point for running any supported simulation type.
    
    Parameters:
        case: Problem case configuration (RiemannCase, DrivenShockCase, or SedovExplosionCase)
        sim_type: Type of simulation (RIEMANN, DRIVEN_SHOCK, or SEDOV)
        Ncells: Number of computational cells
        gamma: Adiabatic index (ratio of specific heats)
        CFL: CFL number for timestep control
        sigma_visc: Artificial viscosity coefficient
        store_every: Store history every N steps (default: 1000)
        geom: Optional geometry override (inferred from case if not provided)
        
    Returns:
        x_cells: Final cell center positions
        state: Final HydroState
        meta: Dictionary with simulation metadata
        history: SimulationHistory with time evolution data
    """

    # Initialize problem
    state, m_cells, x_nodes = _initialize_problem(case, sim_type, geom, gamma, Ncells)
    state.a = compute_acceleration_nodes(state.x, state.p, state.q, m_cells, geom)
    
    t_end = case.t_end
    
    # ---- history buffers ----
    times = []
    Xs, Ms, RHOs, Ps, Us, Es = [], [], [], [], [], []

    def store_frame():
        x_cells = 0.5 * (state.x[:-1] + state.x[1:])
        u_cells = 0.5 * (state.u[:-1] + state.u[1:])
        m_coordinate = np.cumsum(m_cells)
        times.append(state.t)
        Xs.append(x_cells.copy())
        Ms.append(m_coordinate.copy())
        RHOs.append(state.rho.copy())
        Ps.append(state.p.copy())
        Us.append(u_cells.copy())
        Es.append(state.e.copy())
    
    store_frame()
    dt_prev = np.inf
    step = 0
    
    # Main time integration loop
    with tqdm(total=t_end) as pbar:
        while state.t < t_end:
            # Adaptive timestep
            if step > 2:
                dt = compute_dt_cfl(state.x, state.u, state.rho, state.p, gamma, CFL)
                dt = min(dt, 0.05 * t_end, dt_prev * 1.1, t_end - state.t)
                if np.isnan(dt):
                    dt = min(0.05 * t_end, dt_prev * 1.1, t_end - state.t)
            else:
                # Small initial timestep for stability
                dt = 1e-12 if sim_type in (SimulationType.DRIVEN_SHOCK, SimulationType.SEDOV) else 1e-6 * t_end
            dt_prev = dt

            # Get boundary conditions for current state
            bc_left, bc_right = _get_boundary_conditions(case, sim_type, state)
            
            # Lagrangian step
            state = step_lagrangian(
                state, m_cells, geom, gamma, sigma_visc,
                bc_left=bc_left, 
                bc_right=bc_right,
                dt=dt
            )

            step += 1
            if (step % store_every) == 0:
                store_frame()
            
            pbar.update(dt)

    # Ensure last frame stored
    if times[-1] != state.t:
        store_frame()

    history = SimulationHistory(
        t=np.array(times),
        x=np.stack(Xs, axis=0),
        m=np.stack(Ms, axis=0),
        rho=np.stack(RHOs, axis=0),
        p=np.stack(Ps, axis=0),
        u=np.stack(Us, axis=0),
        e=np.stack(Es, axis=0),
    )

    x_cells = 0.5 * (state.x[:-1] + state.x[1:])
    meta = dict(case=case, sim_type=sim_type, geometry=geom)
    
    return x_cells, state, meta, history


# ============================================================================
# Convenience wrappers for specific problem types
# ============================================================================

def simulate_driven_shock(
    case: DrivenShockCase,
    *,
    Ncells: int,
    CFL: float,
    sigma_visc: float,
    store_every: int = 1000,
) -> tuple:
    """
    Simulate a driven shock problem.
    
    Backward-compatible wrapper around simulate_lagrangian.
    """
    return simulate_lagrangian(
        case,
        SimulationType.DRIVEN_SHOCK,
        Ncells=Ncells,
        gamma=case.gamma,
        CFL=CFL,
        sigma_visc=sigma_visc,
        store_every=store_every,
    )


def simulate_riemann(
    test_id: int,
    *,
    Ncells: int,
    gamma: float,
    CFL: float,
    sigma_visc: float,
) -> tuple:
    """
    Simulate a Riemann shock tube problem.
    
    Backward-compatible with the original riemann_sim interface.
    
    Returns:
        x_cells: Cell center positions
        numeric: Dictionary with numerical results (rho, p, u, e)
        exact: Dictionary with exact solution (rho, p, u, e)
        meta: Dictionary with simulation metadata
    """
    from problems.riemann_problem import RIEMANN_TEST_CASES
    
    case = RIEMANN_TEST_CASES[test_id]
    t_end = case.t_end
    x0 = getattr(case, 'x0', 0.0)
    
    # Use unified simulation
    x_cells, state, meta, history = simulate_lagrangian(
        case,
        SimulationType.RIEMANN,
        Ncells=Ncells,
        gamma=gamma,
        CFL=CFL,
        sigma_visc=sigma_visc,
        store_every=max(1, Ncells),  # Only store final
    )
    
    u_num_cells = 0.5 * (state.u[:-1] + state.u[1:])
    
    # Compute exact solution
    rhoL, uL, pL = case.left
    rhoR, uR, pR = case.right
    rho_ex, u_ex, p_ex, e_ex = sample_solution(x_cells, t_end, (rhoL, uL, pL), (rhoR, uR, pR), gamma)
    
    numeric = dict(rho=state.rho, p=state.p, u=u_num_cells, e=state.e)
    exact = dict(rho=rho_ex, p=p_ex, u=u_ex, e=e_ex)
    meta = dict(
        test_id=test_id,
        t_end=t_end,
        Ncells=Ncells,
        gamma=gamma,
        x_min=case.x_min,
        x_max=case.x_max,
        title_extra=case.title,
    )
    
    return x_cells, numeric, exact, meta


def simulate_sedov(
    case: SedovExplosionCase,
    *,
    Ncells: int,
    gamma: float,
    CFL: float,
    sigma_visc: float,
    store_every: int = 1000,
    geom: Optional[Geometry] = None,
) -> tuple:
    """
    Simulate a Sedov-Taylor point explosion.
    
    The Sedov problem models a strong explosion from a point source
    in a uniform ambient medium. Uses reflective BC at the origin
    and outflow BC at the outer boundary.
    
    Parameters:
        case: SedovExplosionCase configuration
        Ncells: Number of computational cells
        gamma: Adiabatic index
        CFL: CFL number for timestep control
        sigma_visc: Artificial viscosity coefficient
        store_every: Store history every N steps
        geom: Optional geometry (spherical, cylindrical, or planar)
        
    Returns:
        x_cells: Final cell center positions (radii)
        state: Final HydroState
        meta: Dictionary with simulation metadata
        history: SimulationHistory with time evolution data
    """
    return simulate_lagrangian(
        case,
        SimulationType.SEDOV,
        Ncells=Ncells,
        gamma=gamma,
        CFL=CFL,
        sigma_visc=sigma_visc,
        store_every=store_every,
        geom=geom,
    )

def simulate_lagrangian(
    rad_hydro_case: RadHydroCase,
    sim_type: SimulationType,
    *,
    Ncells: int,
    gamma: float,
    CFL: float,
    sigma_visc: float,
    store_every: int,
    geom: Geometry,
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
        store_every: Store history every N steps
        geom: Geometry of the simulation domain
        
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
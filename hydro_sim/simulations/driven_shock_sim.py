from tqdm import tqdm
from core.geometry import planar
from core.integrator import step_lagrangian, compute_acceleration_nodes
from core.timestep import compute_dt_cfl
from problems.driven_shock_problem import init_planar_driven_shock_case
import numpy as np
from dataclasses import dataclass

@dataclass
class ShockHistory:
    t: np.ndarray          # (K,)
    x: np.ndarray          # (K, Ncells) cell centers
    rho: np.ndarray        # (K, Ncells)
    p: np.ndarray          # (K, Ncells)
    u: np.ndarray          # (K, Ncells)
    e: np.ndarray          # (K, Ncells)

def simulate_driven_shock(case, *, Ncells, CFL, sigma_visc, store_every=1000):
    geom = planar()
    x_nodes = np.linspace(case.x_min, case.x_max, Ncells + 1)

    state, m_cells = init_planar_driven_shock_case(x_nodes, geom, case.gamma, case)
    state.a = compute_acceleration_nodes(state.x, state.p, state.q, m_cells, geom)

    # ---- history buffers ----
    times = []
    Xs, RHOs, Ps, Us, Es = [], [], [], [], []

    def store_frame():
        x_cells = 0.5 * (state.x[:-1] + state.x[1:])
        u_cells = 0.5 * (state.u[:-1] + state.u[1:])
        times.append(state.t)
        Xs.append(x_cells.copy())
        RHOs.append(state.rho.copy())
        Ps.append(state.p.copy())
        Us.append(u_cells.copy())
        Es.append(state.e.copy())

    store_frame()

    dt_prev = np.inf
    step = 0
    # use a tqdl progress bar
    with tqdm(total=case.t_end) as pbar:
        while state.t < case.t_end:
            if step > 2:
                
                dt = compute_dt_cfl(state.x, state.u, state.rho, state.p, case.gamma, CFL)
                dt = min(dt, 0.05 * case.t_end, dt_prev * 1.1, case.t_end - state.t)
                if np.isnan(dt):
                    dt = min(0.05 * case.t_end, dt_prev * 1.1, case.t_end - state.t)
            else:
                dt = 1e-12
            dt_prev = dt

            state = step_lagrangian(
                state, m_cells, geom, case.gamma, sigma_visc,
                bc_left={"type": "pressure", "p": case.P0},   # keep constant drive for now
                bc_right="outflow",
                dt=dt
            )

            step += 1
            if (step % store_every) == 0:
                store_frame()
        # update progress bar
            pbar.update(dt)

    # ensure last frame stored
    if times[-1] != state.t:
        store_frame()

    history = ShockHistory(
        t=np.array(times),
        x=np.stack(Xs, axis=0),
        rho=np.stack(RHOs, axis=0),
        p=np.stack(Ps, axis=0),
        u=np.stack(Us, axis=0),
        e=np.stack(Es, axis=0),
    )

    x_cells = 0.5 * (state.x[:-1] + state.x[1:])
    return x_cells, state, dict(case=case), history

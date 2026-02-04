import numpy as np
from core.geometry import planar
from core.integrator import step_lagrangian, compute_acceleration_nodes
from core.timestep import compute_dt_cfl
from simulations.riemann_exact import sample_solution
from problems.riemann_problem import RIEMANN_TEST_CASES, init_planar_riemann_case

# @nb.njit(no_python=True) # <-- uncommented because numba can't handle dataclasses, 
# should be solved by extracting the needed arrays as arguments and decorating 
# a separate function that only applies calculations on the arrays.
def simulate_riemann(test_id: int, *, Ncells: int, gamma: float, CFL: float, sigma_visc: float):
    geom = planar()
    case = RIEMANN_TEST_CASES[test_id]

    x_min, x_max, x0 = case.x_min, case.x_max, case.x0
    t_end = case.t_end

    x_nodes = np.linspace(x_min, x_max, Ncells + 1)

    state, m_cells = init_planar_riemann_case(x_nodes, geom, gamma, case, x0=x0)
    state.a = compute_acceleration_nodes(state.x, state.p, state.q, m_cells, geom)

    while state.t < t_end:
        dt = compute_dt_cfl(state.x, state.u, state.rho, state.p, gamma, CFL)
        if state.t + dt > t_end:
            dt = t_end - state.t

        state = step_lagrangian(
            state, m_cells, geom, gamma, sigma_visc,
            bc_left="none", bc_right="none", dt=dt
        )

    x_cells = 0.5 * (state.x[:-1] + state.x[1:])
    u_num_cells = 0.5 * (state.u[:-1] + state.u[1:])

    # exact on same x
    rhoL, uL, pL = case.left
    rhoR, uR, pR = case.right
    rho_ex, u_ex, p_ex, e_ex = sample_solution(x_cells, t_end, (rhoL,uL,pL), (rhoR,uR,pR), gamma)

    numeric = dict(rho=state.rho, p=state.p, u=u_num_cells, e=state.e)
    exact   = dict(rho=rho_ex, p=p_ex, u=u_ex, e=e_ex)
    meta    = dict(test_id=test_id, t_end=t_end, Ncells=Ncells, gamma=gamma,
                   x_min=x_min, x_max=x_max, title_extra=case.title)
    return x_cells, numeric, exact, meta

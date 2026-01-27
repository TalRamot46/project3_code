# problems/run_riemann.py
import numpy as np
import sys
from pathlib import Path
import argparse
import numba as nb
ji = nb.jit(nopython=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.geometry import planar
from core.integrator import step_lagrangian, compute_acceleration_nodes
from core.timestep import compute_dt_cfl
from problems.riemann_exact import sample_solution
from problems.riemann_problem import RIEMANN_CASES, init_planar_riemann_case
from problems.riemann_plots import plot_riemann_comparison

@nb.jit(nopython=True)
def simulate_riemann(test_id: int, *, Ncells: int, gamma: float, CFL: float, sigma_visc: float):
    geom = planar()
    case = RIEMANN_CASES[test_id]

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=1, choices=[1,2,3,4])
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--CFL", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=None, help="Override viscosity sigma (otherwise uses case default)")
    ap.add_argument("--save", type=str, default=None, help="Save figure path (png)")
    args = ap.parse_args()

    case = RIEMANN_CASES[args.test]
    sigma = args.sigma if args.sigma is not None else case.sigma_visc

    x_cells, num, ex, meta = simulate_riemann(
        args.test, Ncells=args.N, gamma=args.gamma, CFL=args.CFL, sigma_visc=sigma
    )

    plot_riemann_comparison(
        x_cells=x_cells,
        rho_num=num["rho"], p_num=num["p"], u_num=num["u"], e_num=num["e"],
        rho_ex=ex["rho"],  p_ex=ex["p"],  u_ex=ex["u"],  e_ex=ex["e"],
        test_id=meta["test_id"], t_end=meta["t_end"], Ncells=meta["Ncells"], gamma=meta["gamma"],
        x_min=meta["x_min"], x_max=meta["x_max"],
        title_extra=meta["title_extra"],
        savepath=args.save,
        show=True
    )

if __name__ == "__main__":
    main()
    # run with:
    # python -m project_3.hydro_sim.problems.run_riemann --test 1 --N 1000
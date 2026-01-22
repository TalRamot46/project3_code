# problems/run_riemann.py
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from geometry import planar
from integrator import compute_dt_cfl, step_lagrangian, compute_acceleration_nodes
from riemann_exact import sample_solution
from problems.riemann_problem import RIEMANN_CASES, init_planar_riemann_case

def run_test(test_id: int, *, Ncells=1000, x_min=-1.0, x_max=1.0,
             gamma=1.4, CFL=0.5, sigma_visc=1.0):
    geom = planar()
    case = RIEMANN_CASES[test_id]
    t_end = case.t_end

    x_nodes = np.linspace(x_min, x_max, Ncells + 1)

    state, m_cells = init_planar_riemann_case(x_nodes, geom, gamma, case, x0=0.0)
    state.a = compute_acceleration_nodes(state.x, state.p, state.q, m_cells, geom)

    # Integrate
    while state.t < t_end:
        dt = compute_dt_cfl(state.x, state.u, state.rho, state.p, gamma, CFL)
        if state.t + dt > t_end:
            dt = t_end - state.t
        state = step_lagrangian(
            state, m_cells, geom, gamma, sigma_visc,
            bc_left="none", bc_right="none", dt=dt   # <-- important for Riemann tests
        )

    # Plot
    x_cells = 0.5 * (state.x[:-1] + state.x[1:])
    rhoL, uL, pL = case.left
    rhoR, uR, pR = case.right
    rho_ex, u_ex, p_ex, e_ex = sample_solution(x_cells, t_end, (rhoL,uL,pL), (rhoR,uR,pR), gamma)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    ax_rho, ax_p, ax_u, ax_e = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    ax_rho.plot(x_cells, rho_ex, linewidth=2)
    ax_rho.plot(x_cells, state.rho, linestyle="None", marker="+")
    ax_rho.set_ylabel("density")

    ax_p.plot(x_cells, p_ex, linewidth=2)
    ax_p.plot(x_cells, state.p, linestyle="None", marker="+")
    ax_p.set_ylabel("pressure")

    u_num_cells = 0.5*(state.u[:-1] + state.u[1:])
    ax_u.plot(x_cells, u_ex, linewidth=2)
    ax_u.plot(x_cells, u_num_cells, linestyle="None", marker="+")
    ax_u.set_ylabel("velocity")

    ax_e.plot(x_cells, e_ex, linewidth=2)
    ax_e.plot(x_cells, state.e, linestyle="None", marker="+")
    ax_e.set_ylabel("energy")

    for ax in [ax_rho, ax_p, ax_u, ax_e]:
        ax.grid(True)
        ax.set_xlim(x_min, x_max)

    ax_u.set_xlabel("x (cm)")
    ax_e.set_xlabel("x (cm)")
    fig.suptitle(f"Planar Riemann Test {test_id} at t={t_end}, N={Ncells}, gamma={gamma}")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change this to 2,3,4 as needed
    run_test(1)

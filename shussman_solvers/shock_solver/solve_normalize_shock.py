import numpy as np
from project_3.shussman_solvers.shock_solver.utils import integrate_ode
from project_3.shussman_solvers.shock_solver.F_shock import F3
import matplotlib.pyplot as plt

def solve_normalize3(tau: float, r: float, iternum: int = 20, xi_f0: float = 4.0):
    first_change = int(np.floor(np.log(xi_f0)/np.log(2.0))) if xi_f0 > 0 else 0
    wm3 = 1.0 + 0.5 * tau

    a = np.zeros(iternum + 1, dtype=float)
    a[0] = float(xi_f0)

    t_out = None
    x_out = None
    best_i = 0

    for i in range(iternum):
        xi_f = a[i]

        # Hugoniot initial condition at the shock front (xi = xi_f)
        V0 = r / (r + 2.0)
        u0 = wm3 * xi_f * 2.0 / r * V0
        P0 = wm3 * xi_f * u0
        x0 = np.array([V0, P0, u0], dtype=float)
         
        sol = integrate_ode(
            lambda tt, yy: F3(tt, yy, tau, r),
            (xi_f, 1e-6),
            y0=x0,
            rtol=1e-9,
            atol=1e-9,
            max_step=0.05,          # optional, tune if needed
            stop_on_nonfinite=True,
            positivity_idx=(0, 1),  # V and P must stay positive
            positivity_eps=0.0,
        )

        if not sol.success or sol.t.size < 2:
            a[i+1] = xi_f + 2.0**(-i + first_change)
            continue

        t_out = sol.t
        x_out = sol.y.T
        best_i = i

        P_end = x_out[-1, 1]   # P(near 0)
        if P_end > 1.0:
            a[i+1] = xi_f - 2.0**(-i + first_change)
        else:
            a[i+1] = xi_f + 2.0**(-i + first_change)

    # print(t_out)
    # print(x_out)
    # plt.plot(t_out, x_out[:, 1], label='Ptilde (Pressure)')
    # plt.legend()
    # plt.show()

    if t_out is None or x_out is None:
        raise RuntimeError("Shooting failed to produce a solution.")

    return t_out, x_out, float(a[best_i])  # returns (xi, x, xi_f)
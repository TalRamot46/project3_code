# solve_normalize_super.py
"""
Binary-shooting normalization for the self-similar profile (MATLAB solve_normalize.m).

Role in solver structure:
    Finds the front coordinate xsi such that the ODE solution T(xi) satisfies T(0) = 1.
    Integrates the ODE (F_super.F) from xi = xsi down to 0; adjusts xsi iteratively.
    Called only by manager_super. Depends on F_super and utils_super.

Structure:
    - solve_normalize(alpha, beta, tau, iternum, xsi0, tol):
      Returns (t, x) where t = xi grid, x[:,0] = T, x[:,1] = dT/dxi.
"""
from __future__ import annotations
import numpy as np
from .F_super import F
from .utils_super import integrate_ode


def solve_normalize(
    alpha: float,
    beta: float,
    tau: float,
    iternum: int,
    xsi0: float,
    shooting_tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the solution using binary shooting: if T(0) > 1, decrease xsi_f; else increase.
    xsi0: initial guess (advised to use a power of 2 so solution asymptotically reaches between 2*xsi0 and 0).
    iternum: number of iterations.
    tol: convergence when |T(0)-1| < tol (MATLAB used 1e-5; 1e-4 is faster).
    Returns (t, x) where t is the similarity coordinate xi, x[:,0] = T, x[:,1] = dT/dxi.
    """
    first_change = int(np.floor(np.log(xsi0) / np.log(2.0))) if xsi0 > 0 else 0
    a = np.zeros(iternum + 1, dtype=float)
    a[1] = xsi0

    t_out = None
    x_out = None
    for i in range(1, iternum):
        xi_start = a[i]
        # Integrate from xi = xi_start down to 0 (MATLAB: ode45(..., [a(i), 0], [0.001, -1000]))
        sol = integrate_ode(
            lambda t, y: F(t, y, alpha, beta, tau),
            t_span=(xi_start, 0.0),
            y0=np.array([0.01, -1000.0], dtype=float),
            rtol=1e-6,
            atol=1e-8,
            max_step=1e-4,
            first_step=1e-10,
        )

        # Match MATLAB indexing: MATLAB uses i=1,2,... so step = 2^(-i+first_change).
        # Python i=0,1,... corresponds to MATLAB i=1,2,..., so use 2^(-(i+1)+first_change).
        step = 2.0 ** (-i + first_change)
        if not sol.success or sol.t.size < 2:
            a[i + 1] = a[i] + step
            continue
        # sol.y has shape (2, n_t); we want (n_t, 2) like MATLAB x
        t_out = np.asarray(sol.t, dtype=float)
        x_out = sol.y.T  # (n_t, 2): x_out[:,0]=T, x_out[:,1]=T'
        T_at_zero = float(x_out[-1, 0])
        if T_at_zero > 1.0:
            a[i + 1] = a[i] - step
        else:
            a[i + 1] = a[i] + step

        # Avoid negative or zero xsi for next iteration
        a[i + 1] = max(a[i + 1], 1e-6)
        if abs(T_at_zero - 1.0) < shooting_tol:
            break

    if t_out is None or x_out is None:
        raise RuntimeError("solve_normalize_super: shooting failed to produce a solution.")

    return t_out, x_out

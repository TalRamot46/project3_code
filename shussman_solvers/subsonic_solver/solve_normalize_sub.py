# solve_normalize_sub.py
"""
Binary-shooting normalization for the subsonic self-similar profile (MATLAB solve_normalize.m).
Finds (xsi, P(1)) so that the ODE solution integrated from xsi to 0 satisfies boundary conditions.
State: x = [V, V', P, P', u]. Called only by manager_sub.
"""
from __future__ import annotations
import numpy as np
from .F_sub import F
from .utils_sub import integrate_ode


def solve_normalize(
    alpha: float,
    beta: float,
    lambda_: float,
    mu: float,
    r: float,
    tau: float,
    iternum: int = 3000,
    xsi0: float = 1.0,
    P0: float = 4.0,
):
    """
    Normalize using binary shooting: adjust P(1) so integration completes; adjust xsi from calibrator.
    Returns (t, x) where t is similarity coordinate xi, x has columns [V, V', P, P', u].
    """
    first_change = int(np.floor(np.log(P0) / np.log(2.0))) if P0 > 0 else 0
    a = np.zeros(3, dtype=float)
    a[0] = xsi0
    b = np.zeros(iternum + 1, dtype=float)
    b[0] = P0

    t_out = None
    x_out = None

    for i in range(2):
        for j in range(iternum):
            y0 = np.array([0.002, 0.0, b[j], 0.0, 0.0], dtype=float)
            sol = integrate_ode(
                lambda t, y: F(t, y, alpha, beta, lambda_, mu, r, tau),
                (float(a[i]), 0.0),
                y0,
                rtol=5e-3,
                atol=5e-5,
                max_step=1e-4,
                first_step=1e-10,
            )
            if not sol.success or sol.t.size < 2:
                b[j + 1] = b[j] + 2.0 ** (-j + first_change)
            else:
                b[j + 1] = b[j] - 2.0 ** (-j + first_change)
            b[j + 1] = max(b[j + 1], 1e-6)
        t_out = np.asarray(sol.t, dtype=float)
        x_out = sol.y.T  # (n_t, 5)

        # Calibrator: update xsi for next outer iteration
        calibrator_helper = (4 + alpha - beta) / beta
        calibrator = 2 * (1 - calibrator_helper) / (
            lambda_ - mu + (2 - mu) * calibrator_helper
        )
        P_front = float(x_out[-1, 2])
        V_front = float(x_out[-1, 0])
        denom = P_front * (V_front ** (1 - mu))
        if denom > 1e-20:
            c = (1.0 / denom) ** (1.0 / (2 * calibrator + 2 - mu * calibrator))
        else:
            c = a[i]
        a[i + 1] = max(c, 1e-6)

    if t_out is None or x_out is None:
        raise RuntimeError("solve_normalize_sub: shooting failed to produce a solution.")
    return t_out, x_out

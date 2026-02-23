# utils_super.py
"""
Utility functions for the supersonic solver (no dependency on other solver modules).

Role in solver structure:
    - trapz: used by manager_super to compute the dimensionless energy z = -∫ T^beta d(xi).
    - mid: used when building spatial derivatives on grids (e.g. in report/profile code).
    - integrate_ode: thin wrapper around scipy.integrate.solve_ivp; used by
      solve_normalize_super to integrate the ODE from F_super (Radau, stiff-friendly).
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration: integrand y, abscissa x. Same signature as shussman_shock_solver.utils."""
    return float(np.trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def mid(x: np.ndarray) -> np.ndarray:
    """Midpoints of consecutive elements: (x[:-1] + x[1:])/2. MATLAB mid.m."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (x[:-1] + x[1:])


def integrate_ode(
    fun,
    t_span: tuple[float, float],
    y0: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: float | None = 1e-4,
    first_step: float | None = 1e-10,
    method: str = "Radau",
):
    """
    Wrapper around solve_ivp for the supersonic ODE. Uses Radau by default
    (ODE is stiff: T'' can be huge when T is small).
    Returns solution with .success, .t, .y (shape (n, n_t)).
    """
    y0 = np.asarray(y0, dtype=float)
    t0, tf = float(t_span[0]), float(t_span[1])
    span = abs(tf - t0)
    kwargs = dict(rtol=rtol, atol=atol, method=method)
    if max_step is not None:
        kwargs["max_step"] = max_step
    if first_step is not None and first_step <= span:
        kwargs["first_step"] = first_step
    # else: omit first_step when it would exceed bounds (e.g. xi_start very small)
    sol = solve_ivp(fun, (t0, tf), y0,
                method="RK45",
                rtol=1e-6, atol=1e-8,
                dense_output=False)
    return sol

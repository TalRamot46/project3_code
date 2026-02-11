# utils_sub.py
"""
Utility functions for the subsonic solver (trapz, mid, integrate_ode).
Mirrors utils_super; used by manager_sub and solve_normalize_sub.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration: integrand y, abscissa x. MATLAB trapz(t, y)."""
    return float(np.trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def mid(x: np.ndarray) -> np.ndarray:
    """Midpoints of consecutive elements. MATLAB mid.m: (x(1:end-1)+x(2:end))/2."""
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
    Wrapper around solve_ivp for the subsonic ODE (5-component state).
    Returns solution with .success, .t, .y (shape (5, n_t)).
    """
    y0 = np.asarray(y0, dtype=float)
    t0, tf = float(t_span[0]), float(t_span[1])
    kwargs = dict(rtol=rtol, atol=atol, method=method)
    if max_step is not None:
        kwargs["max_step"] = max_step
    if first_step is not None:
        kwargs["first_step"] = first_step
    sol = solve_ivp(fun, (t0, tf), y0, **kwargs)
    return sol

# F_super.py
"""
ODE right-hand side for the self-similar temperature profile (MATLAB F.m).

Role in solver structure:
    Defines the 2D ODE d[T, T']/d(xi) used to compute the dimensionless profile T(xi).
    Called only by solve_normalize_super (via integrate_ode). Depends only on utils_super
    for the integration wrapper; F itself has no internal imports from other solver modules.

Structure:
    - F(t, x, alpha, beta, tau): returns [dT/dxi, d²T/dxi²] with t = similarity variable xi,
      x = [T, T'] (state), alpha/beta/tau from material and power law.
"""
import numpy as np


def F(t: float, x: np.ndarray, alpha: float, beta: float, tau: float) -> np.ndarray:
    """
    Derivatives for the self-similar profile: x = [T, T'] (dimensionless T and dT/dxi).
    Used by solve_ivp in solve_normalize_super for numerical integration.
    """
    w3 = -0.5 + 0.5 * (beta - alpha - 4) * tau
    T, Tp = float(x[0]), float(x[1])
    # xp(1) = x(2); xp(2) = (x(1)^(beta-alpha-4))*[w3*t*x(2)+tau*x(1)] - (alpha+3)*(1/x(1))*(x(2)^2)
    exp = beta - alpha - 4
    xp0 = Tp
    base = (T ** exp)
    term1 = base * (w3 * t * Tp + tau * T)
    term2 = (alpha + 3) * (Tp**2) / T
    xp1 = float(term1 - term2)
    return np.array([xp0, xp1], dtype=float)

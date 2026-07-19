# solve_for_tau_sub.py
"""
Solve for temporal power-law exponent tau given conserved-quantity units etta (MATLAB solve_for_tau.m).
"""
from __future__ import annotations
import numpy as np


def solve_for_tau(etta1: float, etta2: float, etta3: float, mat) -> float:
    """
    Given a material mat and conserved quantity units etta1, etta2, etta3,
    solve for the temporal evolution exponent tau.
    """
    A = np.zeros((2, 2), dtype=float)
    B = np.zeros(2, dtype=float)
    A[0, 0] = 2 * mat.beta + mat.lambda_ * mat.beta - mat.mu * (4 + mat.alpha)
    A[0, 1] = mat.mu
    A[1, 0] = 3 * mat.mu * (4 + mat.alpha) - 2 * mat.alpha - 2 * mat.beta - 3 * mat.lambda_ * mat.beta - 8
    A[1, 1] = 2 - 3 * mat.mu
    B[0] = -mat.beta * etta1
    B[1] = -mat.beta * etta2
    x_sol = np.linalg.solve(A, B)
    tau = x_sol[0] * (8 - 3 * mat.beta + 2 * mat.alpha) + mat.beta * etta3
    tau = tau / x_sol[1]
    tau = (tau - 2) / mat.beta
    return float(tau)

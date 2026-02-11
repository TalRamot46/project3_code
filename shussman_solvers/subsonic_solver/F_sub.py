# F_sub.py
"""
ODE right-hand side for the subsonic self-similar profile (MATLAB F.m).
State x = [V, V', P, P', u] (5 components). Used by solve_normalize_sub.
"""
from __future__ import annotations
import numpy as np


def F(
    t: float,
    x: np.ndarray,
    alpha: float,
    beta: float,
    lambda_: float,
    mu: float,
    r: float,
    tau: float,
) -> np.ndarray:
    """
    Derivatives for the self-similar profile: x = [V, V', P, P', u].
    Used by solve_ivp in solve_normalize_sub.
    """
    mechane = 4 + 2 * lambda_ - 4 * mu
    wm3 = 2 + 2 * (4 + alpha - beta) * tau - mu * (3 + (4 + alpha) * tau) + lambda_ * (2 + beta * tau)
    wm3 = wm3 / mechane
    wu3 = mu + beta * (2 + lambda_) * tau - (4 + alpha) * mu * tau
    wu3 = wu3 / mechane
    wP3 = -1 + mu + (4 + alpha + beta * lambda_) * tau - (4 + alpha) * mu * tau
    wP3 = wP3 * 2 / mechane
    wV3 = 1 - (4 + alpha - 2 * beta) * tau
    wV3 = wV3 * 2 / mechane
    w3 = -wm3
    temp3 = (4 + alpha) / beta
    temp3 = 1.0

    x1, x2, x3, x4, x5 = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])

    ezerA = ((wV3 + wP3) * x1 * x3 + w3 * t * (x1 * x4 + x2 * x3)) / r
    ezerA2 = x3 * (wV3 * x1 + w3 * t * x2)
    temp2 = (1 - mu) * x3 * (x1 ** (-mu)) * x2 + (x1 ** (1 - mu)) * x4
    ezerB = temp2 * lambda_ * (x1 ** (lambda_ - 1)) * x2
    temp = x3 * (x1 ** (1 - mu))
    ezerB = ezerB * (temp ** ((4 + alpha - beta) / beta))

    ezerC = temp2**2 * (temp ** ((4 + alpha - 2 * beta) / beta)) * (x1**lambda_) * (4 + alpha - beta) / beta

    ezerD = 2 * (1 - mu) * (x1 ** (-mu)) * x2 * x4 - mu * (1 - mu) * x3 * (x1 ** (-mu - 1)) * (x2**2)

    temp4 = (x1**lambda_) * (temp ** ((4 + alpha - beta) / beta))
    ezerE = (wu3 + w3) * (wV3 * x1 + w3 * t * x2) + w3 * t * (wV3 * x2 + w3 * x2)
    ezerE = -ezerE
    ezerForP = ezerE
    ezerE = ezerE * (x1 ** (1 - mu))

    ezerF = (1 - mu) * x3 * (x1 ** (-mu)) - (x1 ** (1 - mu)) * (w3**2) * (t**2)

    xp = np.zeros(5, dtype=float)
    xp[4] = wV3 * x1 + w3 * t * x2  # u'
    xp[0] = x2  # V'
    xp[2] = x4  # P'
    xp[1] = (ezerA + ezerA2) / temp3
    xp[1] = (xp[1] - ezerB - ezerC) / temp4
    xp[1] = (xp[1] - ezerD - ezerE) / ezerF  # V"
    xp[3] = -(w3**2) * (t**2) * xp[1] + ezerForP  # P"
    return xp

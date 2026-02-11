# manager_sub.py
"""
Manager: orchestrates the subsonic self-similar solution (MATLAB manager.m).
Returns normalized profile and dimensional scaling constants (m0, e0, P0, V0, u0, power laws, etc.).
"""
from __future__ import annotations
import numpy as np
from .materials_sub import MaterialSub, HEV_IN_KELVIN
from .solve_normalize_sub import solve_normalize
from .utils_sub import trapz


def manager_sub(
    mat: MaterialSub,
    tau: float,
    *,
    iternum: int = 3000,
    xsi0: float = 1.0,
    P0: float = 4.0,
):
    """
    Self-similar solution for a material and temporal power law tau.
    Returns m0, mw, e0, ew, P0_out, Pw, V0, Vw, u0, uw, xsi, z, Ptilda, utilda, B, t, x.
    - m = m0 * T^mw(2) * t^mw(3) in g/cm^2
    - e = e0 * T^ew(2) * t^ew(3) in J/cm^2
    - P, V, u with analogous power laws; xsi = max(xi); z = dimensionless energy.
    """
    tempW = 4 + 2 * mat.lambda_ - 4 * mat.mu
    w1 = (mat.mu - 2) / tempW
    w2 = (-8 - 2 * mat.alpha + 2 * mat.beta - mat.beta * mat.lambda_ + (4 + mat.alpha) * mat.mu) / tempW
    w3 = (-2 - 2 * (4 + mat.alpha - mat.beta) * tau + mat.mu * (3 + (4 + mat.alpha) * tau) - mat.lambda_ * (2 + mat.beta * tau)) / tempW

    mw = np.zeros(3, dtype=float)
    mw[0] = -w1
    mw[1] = -w2
    mw[2] = -w3

    ew = np.zeros(3, dtype=float)
    ew[0] = (2 - 3 * mat.mu) / tempW
    ew[1] = (8 + 2 * mat.alpha + 2 * mat.beta + 3 * mat.beta * mat.lambda_ - 3 * mat.mu * (4 + mat.alpha)) / tempW
    ew[2] = (2 + 2 * (4 + mat.alpha + mat.beta) * tau + mat.mu * (-1 - 3 * (4 + mat.alpha) * tau) + mat.lambda_ * (2 + 3 * mat.beta * tau)) / tempW

    Pw = np.zeros(3, dtype=float)
    Pw[0] = (1 - mat.mu) * 2 / tempW
    Pw[1] = (4 + mat.alpha + mat.beta * mat.lambda_ - (4 + mat.alpha) * mat.mu) * 2 / tempW
    Pw[2] = (-1 + mat.mu + (4 + mat.alpha + mat.lambda_ * mat.beta) * tau - (4 + mat.alpha) * mat.mu * tau) * 2 / tempW

    Vw = np.zeros(3, dtype=float)
    Vw[0] = -2 / tempW
    Vw[1] = (-4 - mat.alpha + 2 * mat.beta) * 2 / tempW
    Vw[2] = (1 - (4 + mat.alpha - 2 * mat.beta) * tau) * 2 / tempW

    uw = np.zeros(3, dtype=float)
    uw[0] = -mat.mu / tempW
    uw[1] = (mat.beta * (2 + mat.lambda_) - (4 + mat.alpha) * mat.mu) / tempW
    uw[2] = (mat.mu + mat.beta * (2 + mat.lambda_) * tau - (4 + mat.alpha) * mat.mu * tau) / tempW

    A = 3 * mat.f * mat.beta / 16 / mat.sigma / mat.g
    B = (16 * mat.sigma * mat.g) / mat.beta / 3
    B = B * ((mat.r * mat.f) ** (-(4 + mat.alpha) / mat.beta))

    t, x = solve_normalize(
        mat.alpha, mat.beta, mat.lambda_, mat.mu, mat.r, tau,
        iternum=iternum, xsi0=xsi0, P0=P0,
    )

    z = -trapz(x[:, 0] * x[:, 2], t) / mat.r - 0.5 * trapz(x[:, 4] ** 2, t)
    xsi = float(np.max(t))
    Ptilda = float(x[0, 2])
    utilda = float(x[-1, 4])

    P0_out = (
        Ptilda
        * (B ** Pw[0])
        * ((mat.r * mat.f) ** (Pw[1] / mat.beta))
        * (10.0 ** (-9 * (-Pw[1] * tau)))
        / 1e12
        * (HEV_IN_KELVIN ** Pw[1])
        * (1e-9 ** Pw[2])
    )
    m0 = (
        xsi
        * (B ** mw[0])
        * ((mat.r * mat.f) ** (mw[1] / mat.beta))
        * (10.0 ** (-9 * (-mw[1] * tau)))
        * (HEV_IN_KELVIN ** mw[1])
        * (1e-9 ** mw[2])
    )
    e0 = (
        z
        * (B ** ew[0])
        * ((mat.r * mat.f) ** (ew[1] / mat.beta))
        * (10.0 ** (-9 * (-ew[1] * tau)))
        * (HEV_IN_KELVIN ** ew[1])
        * (1e-9 ** ew[2])
        / 1e9
        / 100.0
    )
    V0 = (
        (B ** Vw[0])
        * ((mat.r * mat.f) ** (Vw[1] / mat.beta))
        * (10.0 ** (-9 * (-Vw[1] * tau)))
        * (HEV_IN_KELVIN ** Vw[1])
        * (1e-9 ** Vw[2])
    )
    u0 = (
        utilda
        * (B ** uw[0])
        * ((mat.r * mat.f) ** (uw[1] / mat.beta))
        * (10.0 ** (-9 * (-uw[1] * tau)))
        / 1e5
        * (HEV_IN_KELVIN ** uw[1])
        * (1e-9 ** uw[2])
    )

    return m0, mw, e0, ew, P0_out, Pw, V0, Vw, u0, uw, xsi, z, Ptilda, utilda, B, t, x

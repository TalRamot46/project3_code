# manager_super.py
"""
Manager: orchestrates the supersonic self-similar solution (MATLAB manager.m).

Role in solver structure:
    Top-level routine that takes a material and temporal power-law exponent tau, and
    returns the normalized profile and dimensional scaling constants. Calls
    solve_normalize_super to get (t, x), then computes power-law exponents (mw, ew),
    constant A, dimensionless energy z, and dimensional m0, e0 (with unit conversions).

Structure:
    - manager_super(mat, tau, iternum, xsi0, tol):
      Returns (m0, mw, e0, ew, xsi, z, A, t, x). Used by profiles_for_report_super and
      run_super. Depends on materials_super, solve_normalize_super, utils_super.
"""
from __future__ import annotations
import numpy as np
from .materials_super import MaterialSuper, KELVIN_PRE_HEV
from .solve_normalize_super import solve_normalize
from .utils_super import trapz


def manager_super(
    mat: MaterialSuper,
    tau: float
) -> tuple[float, np.ndarray, float, np.ndarray, float, float, float, np.ndarray, np.ndarray]:
    """
    Provides a self-similar solution for a material and temporal power law tau.

    Returns
    -------
    m0, mw, e0, ew, xsi, z, A, t, x
    - m = m0 * T^mw(2) * t^mw(3)  in g/cm^2
    - e = e0 * T^ew(2) * t^ew(3)  in J/cm^2
    - xsi: self-similar front coordinate (max xi)
    - z: dimensionless energy (integral of -T^beta over xi)
    - A: parameter from the article (3*f*beta/(16*sigma*g))
    - t: similarity coordinate xi
    - x: x[:,0] = self-similar T, x[:,1] = dT/dxi
    """

    # ---- Power laws (MATLAB: Xsi goes like; m goes like; e0 goes like) ----
    w1 = 0.5
    w2 = (mat.beta - mat.alpha - 4) / 2.0
    w3 = -0.5 + 0.5 * (mat.beta - mat.alpha - 4) * tau

    mw = np.zeros(3, dtype=float)
    mw[0] = -w1
    mw[1] = -w2
    mw[2] = -w3

    ew = np.zeros(3, dtype=float)
    ew[0] = mw[0]
    ew[1] = mat.beta + mw[1]
    ew[2] = mat.beta * tau + mw[2]

    # ---- Constant A (MATLAB: A = 3*mat.f*mat.beta/16/mat.sigma/mat.g) ----
    A = 3.0 * mat.f * mat.beta / 16.0 / mat.sigma / mat.g

    # ---- Solve ODE and normalize ----
    t, x = solve_normalize(mat.alpha, mat.beta, tau, iternum=100, xsi0=1, shooting_tol=1e-5)

    # z = -integral of T^beta over xi (MATLAB: z = -trapz(t, x(:,1).^mat.beta))
    # trapz(y, x): first arg integrand, second abscissa
    z = -trapz(x[:, 0] ** mat.beta, t)

    xsi = float(np.max(t))

    # ---- Dimensional constants m0, e0 (MATLAB formulas with unit conversions) ----
    # m0 in g/cm^2. Sophisticated unit conversion (MATLAB author):
    # - 10^(-9*(-mw(2)*tau)) = 10^(9*mw(2)*tau): time (ns) scaling for the power law
    # - 1160500^mw(2): temperature in HeV (1 HeV = 1160500 K) scaling
    # - (1e-9)^mw(3): time in ns
    m0 = (
        xsi
        * (A ** mw[0])
        * (10.0 ** (-9.0 * (-mw[1] * tau)))
        * (KELVIN_PRE_HEV ** mw[1])
        * (1e-9 ** mw[2])
    )

    # e0 in J/cm^2. Same time/HeV scaling; /1e9/100 is an internal unit conversion
    # (MATLAB author): likely relating energy density or time units to J/cm^2.
    e0 = (
        z
        * mat.f
        * (A ** ew[0])
        * (10.0 ** (-9.0 * (-ew[1] * tau)))
        * (KELVIN_PRE_HEV ** ew[1])
        * (1e-9 ** ew[2])
        / 1e9
        / 100.0
    )

    return m0, mw, e0, ew, xsi, z, A, t, x

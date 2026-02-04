from __future__ import annotations
import numpy as np
from utils import trapz
from materials_shock import Material
from solve_normalize_shock import solve_normalize3


def manager_shock(mat: Material, tau: float,
                  *, iternum: int = 20, xi_f0: float = 4.0):
    """
    Python equivalent of MATLAB shock/manager.m:

      [m0,mw,e0,ew,u0,uw,xsi,z,utilda,ufront,t,x] = manager(mat,tau)

    Notes:
    - t is the similarity coordinate xi (often decreasing from xi_f -> 0).
    - x columns: [Vtilde, Ptilde, utilde]
    """

    # ---- Solve power laws (copied 1:1 from MATLAB) ----
    w1 = -0.5
    w2 =  0.5
    w3 = -1.0 - tau/2.0

    mw = np.zeros(3, dtype=float)
    ew = np.zeros(3, dtype=float)
    uw = np.zeros(3, dtype=float)

    # m goes like: m = m0 * T^(mw(2)) * t^(mw(3))  [g/cm^2]
    mw[0] = -w1  # P0 exponent
    mw[1] = -w2  # mat.V0 exponent
    mw[2] = -w3  # time exponent

    # e goes like: e = e0 * T^(ew(2)) * t^(ew(3))  [J/cm^2]
    ew[0] =  1.5
    ew[1] =  0.5
    ew[2] = -2.0 + (2.0 + tau) * 1.5

    # u goes like: u = u0 * T^(uw(2)) * t^(uw(3))  [cm/s]
    uw[0] = 0.5
    uw[1] = 0.5
    uw[2] = tau/2.0

    # ---- Solve the self-similar ODE + normalization (MATLAB solve_normalize3) ----
    t, x, _ = solve_normalize3(tau, mat.r, iternum=iternum, xi_f0=xi_f0)

    # In MATLAB:
    # z = -trapz(t, Vtilde*Ptilde)/r - 0.5*trapz(t, utilde^2)
    Vtilde = x[:, 0]
    Ptilde = x[:, 1]
    utilde = x[:, 2]

    z = -trapz(Vtilde * Ptilde, t) / mat.r
    z = z - 0.5 * trapz(utilde**2, t)

    xsi = float(np.max(t))  # xi_f

    # "front" values in MATLAB use the first row x(1,:)
    # because ode integrates from xi_f downwards, so x[0] is at xi=xi_f
    Vtilda_front = float(x[0, 0])
    Ptilda_front = float(x[0, 1])
    utilda = float(x[0, 2])  # self-similar front velocity

    # ufront = utilda * (mat.V0^uw(2))
    ufront = utilda * (mat.V0 ** uw[1])

    # ---- Dimensional constants (copied 1:1 from MATLAB) ----
    # m0 = (10^(12*mw(1))) * (10^(-9*(mw(3)-mw(1)*tau))) * xsi * (mat.V0^mw(2))
    # e0 = (10^(12*ew(1))) * (10^(-9*(ew(3)-ew(1)*tau))) * z   * (mat.V0^ew(2))
    # u0 = (10^(12*uw(1))) * (10^(-9*(uw(3)-uw(1)*tau))) * utilda*(mat.V0^uw(2))/1e5

    m0 = (10.0 ** (12.0 * mw[0])) * (10.0 ** (-9.0 * (mw[2] - mw[0] * tau))) * xsi * (mat.V0 ** mw[1])
    e0 = (10.0 ** (12.0 * ew[0])) * (10.0 ** (-9.0 * (ew[2] - ew[0] * tau))) * z   * (mat.V0 ** ew[1])
    u0 = (10.0 ** (12.0 * uw[0])) * (10.0 ** (-9.0 * (uw[2] - uw[0] * tau))) * utilda * (mat.V0 ** uw[1]) / 1e5

    # Return in the same order as MATLAB manager
    return m0, mw, e0, ew, u0, uw, xsi, z, utilda, ufront, t, x
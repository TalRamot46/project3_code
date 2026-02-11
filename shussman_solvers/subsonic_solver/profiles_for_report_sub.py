# profiles_for_report_sub.py
"""
Report profiles: builds m_heat, P_heat, T_heat, u_heat, rho_heat, etc. (MATLAB profiles_for_report_sub.m).
Post-processing: calls manager_sub once, then evaluates power-law formulas at a sequence of times.
"""
from __future__ import annotations
import numpy as np
from .materials_sub import MaterialSub
from .manager_sub import manager_sub
from .utils_sub import mid


def compute_profiles_for_report(
    mat: MaterialSub,
    tau: float,
    *,
    times: np.ndarray | None = None,
    T0: float = 7.0,
    iternum: int = 3000,
    xsi0: float = 1.0,
    P0: float = 4.0,
):
    """
    Compute mass, pressure, temperature, velocity, density and derivative profiles at a sequence of times.
    Returns dict with times, t, x, m_heat, P_heat, dPdm_heat, T_heat, u_heat, rho_heat,
    drhodm_heat, dPdx_heat, drhodx_heat, zeta, F_heat, dPdm_numeric_heat, drhodm_numeric_heat,
    dTdm_numeric_heat, dTdx_heat, and meta (m0, mw, e0, ew, P0_out, Pw, V0, Vw, u0, uw, xsi, Ptilda, utilda).
    """
    if times is None:
        times = np.array([0.1, 0.15, 0.25, 0.5, 0.75, 1.0], dtype=float) * 100.0
    times = np.asarray(times, dtype=float).ravel()

    (m0, mw, e0, ew, P0_out, Pw, V0, Vw, u0, uw, xsi, z, Ptilda, utilda, B, t, x) = manager_sub(
        mat, tau, iternum=iternum, xsi0=xsi0, P0=P0
    )

    t = np.flipud(t)
    x = np.flipud(x)
    # Now t[0] is rear (xi=0), t[-1] is front (xi=xsi)

    n_times = len(times)
    n_xi = len(t)

    m_heat = np.zeros((n_times, n_xi), dtype=float)
    P_heat = np.zeros((n_times, n_xi), dtype=float)
    dPdm_heat = np.zeros((n_times, n_xi), dtype=float)
    T_heat = np.zeros((n_times, n_xi), dtype=float)
    u_heat = np.zeros((n_times, n_xi), dtype=float)
    rho_heat = np.zeros((n_times, n_xi), dtype=float)
    drhodm_heat = np.zeros((n_times, n_xi), dtype=float)
    dPdx_heat = np.zeros((n_times, n_xi), dtype=float)
    drhodx_heat = np.zeros((n_times, n_xi), dtype=float)
    F_heat = np.zeros((n_times, n_xi - 1), dtype=float)
    dPdm_numeric_heat = np.zeros((n_times, n_xi - 1), dtype=float)
    drhodm_numeric_heat = np.zeros((n_times, n_xi - 1), dtype=float)
    dTdm_numeric_heat = np.zeros((n_times, n_xi - 1), dtype=float)
    dTdx_heat = np.zeros((n_times, n_xi - 1), dtype=float)

    t_xi = np.asarray(t, dtype=float)
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    for i in range(n_times):
        ti = times[i]
        m_heat[i, :] = m0 * (T0 ** mw[1]) * (ti ** mw[2]) * (t_xi / xsi)
        P_heat[i, :] = P0_out * (T0 ** Pw[1]) * (ti ** Pw[2]) * (x3 / Ptilda)
        dPdm_heat[i, :] = (
            P0_out * (T0 ** Pw[1]) * (ti ** Pw[2]) * (x4 / Ptilda)
            / (m0 * (T0 ** mw[1]) * (ti ** mw[2]) / xsi)
        )
        T_heat[i, :] = 100.0 * T0 * (ti ** tau) * (x3 * (x1 ** (1 - mat.mu))) ** (1.0 / mat.beta)
        u_heat[i, :] = u0 * (T0 ** uw[1]) * (ti ** uw[2]) * (x5 / utilda)
        rho_heat[i, :] = 1.0 / (V0 * (T0 ** Vw[1]) * (ti ** Vw[2]) * x1)
        drhodm_heat[i, :] = (
            -(V0 * (T0 ** Vw[1]) * (ti ** Vw[2]) * x2)
            / (m0 * (T0 ** mw[1]) * (ti ** mw[2]) / xsi)
            * (rho_heat[i, :] ** 2)
        )
        dPdx_heat[i, :] = dPdm_heat[i, :] * rho_heat[i, :]
        drhodx_heat[i, :] = drhodm_heat[i, :] * rho_heat[i, :]

        zeta = (x3 * (x1 ** (1 - mat.mu))) ** (1.0 / mat.beta)
        zeta = zeta ** (4 + mat.alpha)
        dzetady = np.diff(zeta) / np.diff(t_xi / xsi)
        F_heat[i, :] = -mid(x1 ** mat.lambda_) * dzetady * xsi

        dPdm_numeric_heat[i, :] = np.diff(P_heat[i, :]) / np.diff(m_heat[i, :])
        drhodm_numeric_heat[i, :] = np.diff(rho_heat[i, :]) / np.diff(m_heat[i, :])
        dTdm_numeric_heat[i, :] = np.diff(T_heat[i, :]) / np.diff(m_heat[i, :])
        dTdx_heat[i, :] = dTdm_numeric_heat[i, :] * mid(rho_heat[i, :])

    return {
        "times": times,
        "t": t_xi,
        "x": x,
        "m_heat": m_heat,
        "P_heat": P_heat,
        "dPdm_heat": dPdm_heat,
        "T_heat": T_heat,
        "u_heat": u_heat,
        "rho_heat": rho_heat,
        "drhodm_heat": drhodm_heat,
        "dPdx_heat": dPdx_heat,
        "drhodx_heat": drhodx_heat,
        "F_heat": F_heat,
        "dPdm_numeric_heat": dPdm_numeric_heat,
        "drhodm_numeric_heat": drhodm_numeric_heat,
        "dTdm_numeric_heat": dTdm_numeric_heat,
        "dTdx_heat": dTdx_heat,
        "m0": m0,
        "mw": mw,
        "e0": e0,
        "ew": ew,
        "P0": P0_out,
        "Pw": Pw,
        "V0": V0,
        "Vw": Vw,
        "u0": u0,
        "uw": uw,
        "xsi": xsi,
        "z": z,
        "Ptilda": Ptilda,
        "utilda": utilda,
        "B": B,
    }

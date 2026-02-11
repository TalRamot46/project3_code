# profiles_for_report_super.py
"""
Report profiles: builds m_heat, x_heat, T_heat over time (MATLAB profiles_for_report_super.m).

Role in solver structure:
    Post-processing layer. Calls manager_super once to get the self-similar solution and
    scaling constants, then evaluates the power-law formulas at a sequence of times to
    produce 2D arrays of areal mass, position, and temperature for reporting/plots.
    Depends on materials_super and manager_super only.

Structure:
    - compute_profiles_for_report(mat, tau, times, T0, iternum, xsi0):
      Returns a dict with times, t, x, m_heat, x_heat, T_heat, and meta (m0, mw, e0, ew, xsi, z, A).
"""
from __future__ import annotations
import numpy as np
from .materials_super import MaterialSuper
from .manager_super import manager_super


def compute_profiles_for_report(
    mat: MaterialSuper,
    tau: float,
    *,
    times: np.ndarray | None = None,
    T0: float = 2.0,
    iternum: int = 100,
    xsi0: float = 1.0,
):
    """
    Compute mass, position, and temperature profiles at a sequence of times.

    Parameters
    ----------
    mat : material
    tau : temporal power-law exponent
    times : 1D array of dimensionless times (default 0.01 to 1 step 0.01)
    T0 : reference temperature scale (MATLAB T0=2)
    iternum, xsi0 : passed to manager_super

    Returns
    -------
    dict with: times, t, x, m_heat, x_heat, T_heat, m0, mw, e0, ew, xsi, z, A
    - m_heat[i,:]: areal mass (g/cm^2) at each xi for time times[i]
    - x_heat[i,:]: position (cm) = m_heat / rho0
    - T_heat[i,:]: temperature; MATLAB uses 100*T0*times^tau*T_tilde (unit scaling by 100)
    """
    if times is None:
        times = np.arange(0.01, 1.0 + 0.005, 0.01)
    times = np.asarray(times, dtype=float).ravel()

    m0, mw, e0, ew, xsi, z, A, t, x = manager_super(
        mat, tau, iternum=iternum, xsi0=xsi0
    )

    # Self-similar T (dimensionless)
    T_tilde = np.asarray(x[:, 0], dtype=float)
    t_xi = np.asarray(t, dtype=float)

    n_times = len(times)
    n_xi = len(t_xi)

    m_heat = np.zeros((n_times, n_xi), dtype=float)
    x_heat = np.zeros((n_times, n_xi), dtype=float)
    # MATLAB: T_heat(i,:) = 100*T0*times(i)^tau*(x(:,1))
    # The factor 100 is a unit/scaling choice by the MATLAB author (e.g. for display or K vs HeV).
    T_heat = np.zeros((n_times, n_xi), dtype=float)

    for i in range(n_times):
        ti = times[i]
        # m_heat(i,:) = m0*T0^mw(2)*times(i)^mw(3) .* t'/xsi  (t and t' are column in MATLAB)
        m_heat[i, :] = m0 * (T0 ** mw[1]) * (ti ** mw[2]) * (t_xi / xsi)
        x_heat[i, :] = m_heat[i, :] / mat.rho0
        T_heat[i, :] = 100.0 * T0 * (ti ** tau) * T_tilde

    return {
        "times": times,
        "t": t_xi,
        "x": x,
        "m_heat": m_heat,
        "x_heat": x_heat,
        "T_heat": T_heat,
        "m0": m0,
        "mw": mw,
        "e0": e0,
        "ew": ew,
        "xsi": xsi,
        "z": z,
        "A": A,
    }


if __name__ == "__main__":
    from .materials_super import material_al

    # Example: Al with tau = 1/(4+alpha-2*beta) (constant-temperature-like scaling)
    mat = material_al()
    tau = 1.0 / (4.0 + mat.alpha - 2.0 * mat.beta)

    data = compute_profiles_for_report(mat, tau, T0=2.0)
    print("m_heat shape:", data["m_heat"].shape)
    print("xsi =", data["xsi"])
    print("m0 =", data["m0"], "e0 =", data["e0"])

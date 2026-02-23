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

import sys
from pathlib import Path

import numpy as np

try:
    from .materials_super import MaterialSuper
    from .manager_super import manager_super
except ImportError:
    # Run as script: ensure project_3 is on path (repo root = parent of project_3)
    _repo_root = Path(__file__).resolve().parents[3]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from project_3.shussman_solvers.supersonic_solver.materials_super import MaterialSuper
    from project_3.shussman_solvers.supersonic_solver.manager_super import manager_super


def compute_profiles_for_report(
    mat: MaterialSuper,
    tau: float,
    *,
    times: np.ndarray,
    T0: float,
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
    m0, mw, e0, ew, xsi, z, A, t, x = manager_super(mat, tau)
    # Self-similar T (dimensionless)
    T_tilde = np.asarray(x[:, 0], dtype=float)
    t_xi = np.asarray(t, dtype=float)

    n_times = len(times)
    n_xi = len(t_xi)

    m_heat = np.zeros((n_times, n_xi), dtype=float)
    x_heat = np.zeros((n_times, n_xi), dtype=float)
    # MATLAB: T_heat(i,:) = T0*times(i)^tau*(x(:,1))
    T_heat = np.zeros((n_times, n_xi), dtype=float)
    E_heat = np.zeros((n_times, n_xi), dtype=float)

    for i in range(n_times):
        ti = times[i]
        # m_heat(i,:) = m0*T0^mw(2)*times(i)^mw(3) .* t'/xsi  (t and t' are column in MATLAB)
        m_heat[i, :] = m0 * (T0 ** mw[1]) * (ti ** mw[2]) * (t_xi / xsi)
        x_heat[i, :] = m_heat[i, :] / mat.rho0
        T_heat[i, :] = T0 * (ti ** tau) * T_tilde
        E_heat[i, :] = e0 * (T0 ** mw[1]) * (ti ** mw[2]) * (t_xi / xsi)

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
    try:
        from .materials_super import material_au
    except ImportError:
        from project_3.shussman_solvers.supersonic_solver.materials_super import material_au

    # Example: Al with tau = 1/(4+alpha-2*beta) (constant-temperature-like scaling)
    mat = material_au()
    tau = 0.0

    data = compute_profiles_for_report(mat, tau, times=np.array([1.0]), T0=1) # Corresponds to T0=10000K
    print("m_heat shape:", data["m_heat"].shape)
    print("xsi =", data["xsi"])
    print("m0 =", data["m0"], "e0 =", data["e0"])


    import matplotlib.pyplot as plt
    plt.plot(data["x_heat"][0,:], data["T_heat"][0,:])
    plt.show()

# profiles_for_report_sub.py
"""
Report profiles: builds m_heat, P_heat, T_heat, u_heat, rho_heat, etc. (MATLAB profiles_for_report_sub.m).
Post-processing: calls manager_sub once, then evaluates power-law formulas at a sequence of times.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from .materials_sub import MaterialSub
    from .manager_sub import manager_sub
except ImportError:
    # Run as script: ensure project_3 (repo root) is on path
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from shussman_solvers.subsonic_solver.materials_sub import MaterialSub
    from shussman_solvers.subsonic_solver.manager_sub import manager_sub


def compute_profiles_for_report(
    mat: MaterialSub,
    T0_phys_HeV: float,
    tau: float,
    times_ns: np.ndarray,
):
    """
    Compute mass, pressure, temperature, velocity, density and derivative profiles at a sequence of times.
    Returns dict with times, t, x, m_heat, P_heat, dPdm_heat, T_heat, u_heat, rho_heat,
    drhodm_heat, dPdx_heat, drhodx_heat, zeta, F_heat, dPdm_numeric_heat, drhodm_numeric_heat,
    dTdm_numeric_heat, dTdx_heat, and meta (m0, mw, e0, ew, P0_out, Pw, V0, Vw, u0, uw, xsi, Ptilda, utilda).
    """
    times_ns = np.asarray(times_ns, dtype=float).ravel()

    (m0, mw, e0, ew, P0_out, Pw, V0, Vw, u0, uw, xsi, z, Ptilda, utilda, B, t, x) = manager_sub(mat, tau)

    t = np.flipud(t)
    x = np.flipud(x)
    # Now t[0] is rear (xi=0), t[-1] is front (xi=xsi)

    n_times = len(times_ns)
    n_xi = len(t)

    m_heat = np.zeros((n_times, n_xi), dtype=float)
    x_heat = np.zeros((n_times, n_xi), dtype=float)
    P_heat = np.zeros((n_times, n_xi), dtype=float)
    T_heat = np.zeros((n_times, n_xi), dtype=float)
    u_heat = np.zeros((n_times, n_xi), dtype=float)
    rho_heat = np.zeros((n_times, n_xi), dtype=float)

    t_xi = np.asarray(t, dtype=float)
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    for i in range(n_times):
        ti = times_ns[i]
        m_heat[i, :] = m0 * (T0_phys_HeV ** mw[1]) * (ti ** mw[2]) * (t_xi / xsi)
        x_heat[i, :] = np.cumsum(m_heat[i, :] / rho_heat[i, :])
        P_heat[i, :] = P0_out * (T0_phys_HeV ** Pw[1]) * (ti ** Pw[2]) * (x3 / Ptilda)
        T_heat[i, :] = T0_phys_HeV * (ti ** tau) * (x3 * (x1 ** (1 - mat.mu))) ** (1.0 / mat.beta)
        u_heat[i, :] = u0 * (T0_phys_HeV ** uw[1]) * (ti ** uw[2]) * (x5 / utilda)
        rho_heat[i, :] = 1.0 / (V0 * (T0_phys_HeV ** Vw[1]) * (ti ** Vw[2]) * x1)

    return {
        "times": times_ns,
        "t": t_xi,
        "m_heat": m_heat,
        "x_heat": x_heat,
        "P_heat": P_heat,
        "T_heat": T_heat,
        "u_heat": u_heat,
        "rho_heat": rho_heat,
    }

if __name__ == "__main__":
    try:
        from .materials_sub import material_au
    except ImportError:
        from shussman_solvers.subsonic_solver.materials_sub import material_au

    mat = material_au()
    tau = 0.0
    data = compute_profiles_for_report(mat, tau=tau, times_ns=np.array([1.0]), T0_phys_HeV=1) # Corresponds to T0=10000HeV
    import matplotlib.pyplot as plt
    plt.plot(data["m_heat"][0,:], data["T_heat"][0,:])
    plt.show()
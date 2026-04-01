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
    # Run as script: add both:
    # - repo root (so `shussman_solvers.*` works)
    # - GitHub parent (so `project3_code.*` works via implicit namespace package)
    p = Path(__file__).resolve()
    _repo_root = p.parents[2]  # .../GitHub/project3_code
    _project3_code_parent = p.parents[3]  # .../GitHub
    for _p in (_repo_root, _project3_code_parent):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from project3_code.shussman_solvers.supersonic_solver.materials_super import MaterialSuper
    from project3_code.shussman_solvers.supersonic_solver.manager_super import manager_super


def compute_profiles_for_report(
    mat: MaterialSuper,
    T0_phys_HeV: float,
    tau: float,
    times_ns: np.ndarray,
):
    """
    Compute mass, position, and temperature profiles at a sequence of times.
    """
    times_ns = np.asarray(times_ns, dtype=float).ravel()
    m0, mw, e0, ew, xsi, z, A, t, x = manager_super(mat, tau)
    # Self-similar T (dimensionless)
    T_tilde = np.asarray(x[:, 0], dtype=float)
    t_xi = np.asarray(t, dtype=float)

    n_times = len(times_ns)
    n_xi = len(t_xi)

    m_heat = np.zeros((n_times, n_xi), dtype=float)
    x_heat = np.zeros((n_times, n_xi), dtype=float)
    T_heat = np.zeros((n_times, n_xi), dtype=float)
    E_heat = np.zeros((n_times, n_xi), dtype=float)

    for i in range(n_times):
        ti = times_ns[i]
        # m_heat(i,:) = m0*T0^mw(2)*times(i)^mw(3) .* t'/xsi  (t and t' are column in MATLAB)
        m_heat[i, :] = m0 * (T0_phys_HeV ** mw[1]) * (ti ** mw[2]) * (t_xi / xsi) # g/cm^2
        x_heat[i, :] = m_heat[i, :] / mat.rho0 # cm
        T_heat[i, :] = T0_phys_HeV * (ti ** tau) * T_tilde # HeV
        E_heat[i, :] = e0 * (T0_phys_HeV ** mw[1]) * (ti ** mw[2]) * (t_xi / xsi) # hJ/cm^2 (internal unit conversion)

    return {
        "times": times_ns,
        "m_heat": m_heat,
        "x_heat": x_heat,
        "T_heat": T_heat,
        "E_heat": E_heat,
    }

def extract_m_final_expression(mat: MaterialSuper, tau: float):
    """ extracting the final expression for m_final and e_final"""
    (m0, mw, e0, ew, xsi, z, A, t, x) = manager_super(mat, tau)
    print(f"m_final = {m0*mat.rho0**((-mat.mu+mat.lambda_)/2):.2e}* rho_0^{(mat.mu-mat.lambda_)/2:.2f} * T0^{mw[1]:.2f} * t^{mw[2]:.2f}")
    print(f"E_final = {e0*mat.rho0**((mat.mu+mat.lambda_)/2):.2e}* rho_0^{-(mat.mu+mat.lambda_)/2:.2f} * T0^{ew[1]:.2f} * t^{ew[2]:.2f}")

if __name__ == "__main__":
    try:
        from .materials_super import material_au, material_be
    except ImportError:
        print("ImportError: materials_super not found")
        # Ensure both import styles work when executed as a script.
        p = Path(__file__).resolve()
        _repo_root = p.parents[2]  # .../GitHub/project3_code
        _project3_code_parent = p.parents[3]  # .../GitHub
        for _p in (_repo_root, _project3_code_parent):
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
        from shussman_solvers.supersonic_solver.materials_super import material_au, material_be

    # Example: Al with tau = 1/(4+alpha-2*beta) (constant-temperature-like scaling)
    mat = material_au()
    tau = 0.0
    T0_phys_HeV = 1
    times_ns = np.array([1.0])
    data = compute_profiles_for_report(mat, T0_phys_HeV=T0_phys_HeV, tau=tau, times_ns=times_ns) # Corresponds to T0=10000K
    import matplotlib.pyplot as plt
    plt.plot(data["m_heat"][0,:], data["T_heat"][0,:])
    plt.show()

    # mat_be = material_be()
    # extract_m_final_expression(mat_be, tau=0.0)
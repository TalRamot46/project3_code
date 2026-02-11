# run_sub.py
"""
Entry point to run the subsonic self-similar solver (mirrors run_super usage).
run_sub(mat, tau=None, ...) runs manager_sub; tau defaults from constant-temperature scaling if None.
"""
from __future__ import annotations
from .materials_sub import (
    MaterialSub,
    material_al,
    material_au,
    material_be,
    material_cu,
    material_pb,
)
from .manager_sub import manager_sub
from .profiles_for_report_sub import compute_profiles_for_report


def run_sub(
    mat: MaterialSub,
    tau: float | None = None,
    *,
    iternum: int = 3000,
    xsi0: float = 1.0,
    P0: float = 4.0,
):
    """
    Run manager_sub. If tau is None, use constant-temperature-like scaling
    (e.g. from solve_for_tau with etta1=mu/beta, etta2=(2-3*mu)/beta, etta3=-2/beta).
    """
    if tau is None:
        from .solve_for_tau_sub import solve_for_tau
        etta1 = mat.mu / mat.beta
        etta2 = (2 - 3 * mat.mu) / mat.beta
        etta3 = -2 / mat.beta
        tau = solve_for_tau(etta1, etta2, etta3, mat)
    return manager_sub(mat, tau, iternum=iternum, xsi0=xsi0, P0=P0)


if __name__ == "__main__":
    mat = material_al()
    tau = 0.3
    out = run_sub(mat, tau)
    m0, mw, e0, ew, P0_out, Pw, V0, Vw, u0, uw, xsi, z, Ptilda, utilda, B, t, x = out
    print("Material:", mat.name, "tau =", tau)
    print("xsi =", xsi, "m0 =", m0, "e0 =", e0, "P0 =", P0_out)
    data = compute_profiles_for_report(mat, tau, T0=7.0)
    print("m_heat shape:", data["m_heat"].shape)

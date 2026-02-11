# run_super.py
"""
Entry point to run the supersonic self-similar solver (mirrors run_shock_solver usage).

Role in solver structure:
    Script/API entry. run_super(mat, tau=None, ...) runs manager_super (and optionally
    uses tau = 1/(4+alpha-2*beta) if tau is None). __main__ demonstrates a full run with
    material_al and compute_profiles_for_report. Depends on materials_super,
    manager_super, profiles_for_report_super.
"""
from __future__ import annotations
from .materials_super import (
    MaterialSuper,
    material_al,
    material_au,
    material_be,
    material_pb,
    material_sio2,
)
from .manager_super import manager_super
from .profiles_for_report_super import compute_profiles_for_report


def run_super(
    mat: MaterialSuper,
    tau: float | None = None,
    *,
    iternum: int = 100,
    xsi0: float = 1.0,
):
    """
    Run manager_super and optionally use the constant-temperature tau if tau is None.
    tau = 1/(4+alpha-2*beta) is the value commented in profiles_for_report_super.m.
    """
    if tau is None:
        tau = 1.0 / (4.0 + mat.alpha - 2.0 * mat.beta)
    return manager_super(mat, tau, iternum=iternum, xsi0=xsi0)


if __name__ == "__main__":
    mat = material_al()
    tau = 1.0 / (4.0 + mat.alpha - 2.0 * mat.beta)
    m0, mw, e0, ew, xsi, z, A, t, x = run_super(mat, tau)
    print("Material:", mat.name, "tau =", tau)
    print("xsi =", xsi, "m0 =", m0, "e0 =", e0)
    data = compute_profiles_for_report(mat, tau, T0=2.0)
    print("m_heat shape:", data["m_heat"].shape)

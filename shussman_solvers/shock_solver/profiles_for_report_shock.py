# run_shock_solver.py
# Exact translation of Matlab_OG/profiles_for_report_shock.m and manager.m.
# Units: P0 in Barye (cgs), times in seconds. Internal scaling matches MATLAB
# (P0 in 10^12 Barye, time in 10^-9 s) via conversion in the formulas below.
from __future__ import annotations
import os
import sys
from pathlib import Path
from tkinter import N
from typing import Optional
import numpy as np
try:
    from .materials_shock import Material
    from .manager_shock import manager_shock
except ImportError:
    # Run as script: ensure project_3 (repo root) is on path
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from shussman_solvers.shock_solver.materials_shock import Material
    from shussman_solvers.shock_solver.manager_shock import manager_shock

_DEFAULT_NPZ = str(Path(__file__).resolve().parent / "shock_profiles.npz")


def mid(a: np.ndarray) -> np.ndarray:
    """MATLAB mid.m: mid = (x(1:end-1)+x(2:end))/2"""
    a = np.asarray(a, float)
    return 0.5 * (a[:-1] + a[1:])


def compute_shock_profiles(
    mat: Material,
    P0_phys_Barye: float,
    tau: Optional[float],
    Pw: Optional[np.ndarray],
    times_ns : np.ndarray,
    patching_method: bool,
    save_npz: Optional[str],
) -> dict:
    """
    Compute shock profiles for a given material, drive pressure, and time.
    """

    if patching_method and Pw is not None:
        Pw = np.asarray(Pw, float)
        tau = Pw[2]
    elif not patching_method and tau is not None:
        tau = float(tau)
    else:
        raise ValueError("Either Pw or tau must be provided when patching_method is True")

    m0, mw, e0, ew, u0, uw, xsi, z, utilda, ufront, t, x = manager_shock(mat, tau)  # pyright: ignore[reportArgumentType]

    # MATLAB: t = flipud(t); x = flipud(x);
    t = np.asarray(t, float)[::-1]
    x = np.asarray(x, float)[::-1, :]

    V_tilde = x[:, 0]
    P_tilde = x[:, 1]
    u_tilde = x[:, 2]


    # MATLAB uses times in "ns" units: times(i) such that t_phys = times(i)*1e-9 s
    # So for ti in seconds: times_MATLAB = ti * 1e9
    mw0, mw2 = float(mw[0]), float(mw[2])
    uw0, uw2 = float(uw[0]), float(uw[2])

    times_sec = times_ns * 1e-9
    out = {
        "times_sec": times_sec,
        "m_shock": [],
        "x_shock": [],
        "P_shock": [],
        "u_shock": [],
        "rho_shock": [],
        "e_shock": [],
        "T_shock": [],
    }

    for ti in times_ns:
        ti = float(ti)
        m_prof = m0 * P0_phys_Barye ** mw0 * (ti ** mw2) * (t / xsi)
        u_prof = ufront * (P0_phys_Barye ** uw0) * (ti ** uw2) * (u_tilde / utilda) if ti > 0 else np.zeros_like(t)
        rho_prof = 1.0 / (float(mat.V0) * V_tilde)
        x_prof = np.cumsum(m_prof / rho_prof)

        # separating into cases if patching method is used or not
        if patching_method and Pw is not None:
            P_prof = P0_phys_Barye * (ti ** Pw[2]) * P_tilde if ti > 0 else np.zeros_like(t)
            T_prof = (
                (P0_phys_Barye * P_prof / float(mat.r) / float(mat.f)) * (rho_prof ** (float(mat.mu) - 1.0))
            ) ** (1.0 / float(mat.beta))
        elif not patching_method and tau is not None:
            P_prof = P0_phys_Barye * (ti ** tau) * P_tilde if ti > 0 else np.zeros_like(t)
            T_prof = (
                (P0_phys_Barye * P_prof / float(mat.r) / float(mat.f)) * (rho_prof ** (float(mat.mu) - 1.0))
            ) ** (1.0 / float(mat.beta))
        else:
            raise ValueError("Either Pw or tau must be provided when patching_method is True")


        e_prof = P_prof / (rho_prof * float(mat.r))
        
        out["m_shock"].append(m_prof)
        out["x_shock"].append(x_prof)
        out["P_shock"].append(P_prof)
        out["u_shock"].append(u_prof)
        out["rho_shock"].append(rho_prof)
        out["e_shock"].append(e_prof)
        out["T_shock"].append(T_prof)

    if save_npz and not patching_method:  
        save_npz = str(save_npz)
        os.makedirs(os.path.dirname(save_npz) or ".", exist_ok=True)
        np.savez(
            save_npz,
            times_sec=np.array(out["times_sec"], dtype=object),
            m_shock=np.array(out["m_shock"], dtype=object),
            x_shock=np.array(out["x_shock"], dtype=object),
            P_shock=np.array(out["P_shock"], dtype=object),
            u_shock=np.array(out["u_shock"], dtype=object),
            rho_shock=np.array(out["rho_shock"], dtype=object),
            e_shock=np.array(out["e_shock"], dtype=object),
            T_shock=np.array(out["T_shock"], dtype=object)
        )
    else:
        pass
    return out


if __name__ == "__main__":
    try:
        from .materials_shock import au_supersonic_variant_1
    except ImportError:
        from shussman_solvers.shock_solver.materials_shock import au_supersonic_variant_1
    Au = au_supersonic_variant_1()
    P0 = 2.71e12 # Barye
    tau = -0.45
    times = np.array([1.0], float)  
    data = compute_shock_profiles(Au, P0, tau, Pw = None, times_ns=times, patching_method=False, save_npz=None)
    import matplotlib.pyplot as plt
    plt.plot(data["m_shock"][0], data["P_shock"][0])
    plt.show()

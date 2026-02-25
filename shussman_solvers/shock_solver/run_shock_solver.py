# run_shock_solver.py
# Exact translation of Matlab_OG/profiles_for_report_shock.m and manager.m.
# Units: P0 in Barye (cgs), times in seconds. Internal scaling matches MATLAB
# (P0 in 10^12 Barye, time in 10^-9 s) via conversion in the formulas below.
from __future__ import annotations
import os
import sys
from pathlib import Path
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
    P0: float,
    Pw,
    times=None,
    *,
    save_npz: str | None = None,
    drive_scaling: bool = True,
) -> dict:
    """
    Shock profiles matching Matlab_OG/profiles_for_report_shock.m and manager.m.

    Inputs
    ------
    mat : Material with {r, f, mu, beta, V0}
    P0  : drive pressure in Barye (cgs)
    Pw  : array-like length >= 3; Pw[2] = tau (temporal power-law index)
    times : array-like of times in seconds; default [0.1, ..., 1.0]*1e-9 (0.1–1 ns)
    save_npz : if set, save results to this path
    drive_scaling : if True, apply P0_eff = P0 * (7^Pw[2]) as in MATLAB script

    Returns
    -------
    dict with times, t, x, P0_eff, Pw, meta, and per-time lists: m_shock, x_shock,
    P_shock, u_shock, rho_shock, e_shock, T_shock, dPdx, drhodx, dTdx.
    All physical quantities in cgs: P in Barye, u in cm/s, T in K, rho in g/cm^3.
    """
    Pw = np.asarray(Pw, float)

    if times is None:
        times = np.array([1.0], float)
    else:
        times = np.asarray(times, float)

    # MATLAB: [m0,mw,...] = manager(mat,Pw(3));
    tau = float(Pw[2])
    m0, mw, e0, ew, u0, uw, xsi, z, utilda, ufront, t, x = manager_shock(mat, tau)

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
    Pw2 = float(Pw[2])

    out = {
        "times": times,
        "t": t,
        "x": x,
        "P0": P0,
        "Pw": Pw,
        "meta": dict(m0=m0, mw=mw, e0=e0, ew=ew, u0=u0, uw=uw, xsi=xsi, z=z, utilda=utilda, ufront=ufront),
        "m_shock": [],
        "x_shock": [],
        "P_shock": [],
        "u_shock": [],
        "rho_shock": [],
        "e_shock": [],
        "T_shock": [],
        "dPdx": [],
        "drhodx": [],
        "dTdx": [],
    }

    for ti in times:
        ti = float(ti)
        
        # times in MATLAB script: [10, 15, 25, 50, 75, 100] = physical time in 10^-9 s

        # MATLAB: m_shock(i,:)=m0*P0^mw(1)*times(i)^mw(3).*t'/xsi;
        m_prof = m0 * P0 ** mw0 * (ti ** mw2) * (t / xsi)

        # MATLAB: P_shock(i,:)=P0*times(i)^Pw(3).*x(:,2);  (P0 in 10^12 Barye there; we output Bar)
        P_prof = P0 * (ti ** Pw2) * P_tilde if ti > 0 else np.zeros_like(t)

        # MATLAB: u_shock(i,:)=ufront*(P0*1e12)^uw(1)*times(i)^uw(3).*x(:,3)/utilda/1e5;
        # With P0_eff in Barye, (P0_eff) plays the role of (P0*1e12) in MATLAB -> u in cm/s
        u_prof = ufront * (P0 ** uw0) * (ti ** uw2) * (u_tilde / utilda) if ti > 0 else np.zeros_like(t)

        # MATLAB: rho_shock(i,:)=1./(mat.V0*x(:,1));
        rho_prof = 1.0 / (float(mat.V0) * V_tilde)

        # MATLAB: T_shock = (...)^(1/mat.beta)/11605  (T in eV). We output T in Kelvin.
        # T_K = (P/(r*f*rho^(mu-1)))^(1/beta)
        T_prof = (
            (P0 * P_tilde * ti**Pw2 / float(mat.r) / float(mat.f)) * (rho_prof ** (float(mat.mu) - 1.0))
        ) ** (1.0 / float(mat.beta))

        # e = P/(rho*r) [cgs]
        e_prof = P_prof / (rho_prof * float(mat.r))

        # Integrate 1/rho over m to get x (position)
        x_prof = np.zeros_like(m_prof)
        x_prof[0] = m_prof[0] / rho_prof[0]
        for i in range(1, len(m_prof)):
            dm = m_prof[i] - m_prof[i - 1]
            x_prof[i] = x_prof[i - 1] + dm / (0.5 * (rho_prof[i] + rho_prof[i - 1]))

        # MATLAB: dPdm_numeric_shock = diff(P_shock)./diff(m_shock); dPdx = dPdm.*mid(rho);
        dm = np.diff(m_prof)
        rho_mid = mid(rho_prof)
        dPdx = (np.diff(P_prof) / dm) * rho_mid
        drhodx = (np.diff(rho_prof) / dm) * rho_mid
        dTdx = (np.diff(T_prof) / dm) * rho_mid

        out["m_shock"].append(m_prof)
        out["x_shock"].append(x_prof)
        out["P_shock"].append(P_prof)
        out["u_shock"].append(u_prof)
        out["rho_shock"].append(rho_prof)
        out["e_shock"].append(e_prof)
        out["T_shock"].append(T_prof)
        out["dPdx"].append(dPdx)
        out["drhodx"].append(drhodx)
        out["dTdx"].append(dTdx)

    if save_npz:
        save_npz = str(save_npz)
        os.makedirs(os.path.dirname(save_npz) or ".", exist_ok=True)
        np.savez(
            save_npz,
            times=times,
            t=t,
            x=x,
            P0=P0,
            Pw=Pw,
            **{k: np.array(v, dtype=object) for k, v in out.items() if isinstance(v, list)},
            **out["meta"],
        )

    return out


if __name__ == "__main__":
    try:
        from .materials_shock import au_supersonic_variant_1
    except ImportError:
        from shussman_solvers.shock_solver.materials_shock import au_supersonic_variant_1
    Au = au_supersonic_variant_1()
    P0 = 2.71e12 # Barye
    Pw = [0.0, 0.0, -0.45]
    data = compute_shock_profiles(Au, P0, Pw)
    import matplotlib.pyplot as plt
    plt.plot(data["m_shock"][0], data["T_shock"][0])
    plt.show()

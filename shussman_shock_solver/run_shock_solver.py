# run_shock_profiles.py
from __future__ import annotations
import os
import numpy as np
from materials_shock import Material, au_supersonic_variant_1
from manager_shock import manager_shock  # adjust path if needed


def mid(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return 0.5 * (a[:-1] + a[1:])


def compute_shock_profiles(mat: Material, P0: float, Pw, times=None, *, save_npz: str | None = "project_3\\shussman_shock_solver\\shock_profiles.npz") -> dict:
    """
    Lightweight translation of the MATLAB block.

    Inputs
    ------
    mat : object with fields {r,f,mu,beta,V0}
    P0  : float
    Pw  : array-like, length>=3 (MATLAB Pw(2), Pw(3) -> Python Pw[1], Pw[2])
    times : array-like of times (same units as MATLAB 'times'); default matches MATLAB
    save_npz : filename to save, or None to skip saving

    Returns
    -------
    dict with keys: times,t,x,P0,Pw, (and per-time lists of profiles + derivatives)
    """
    Pw = np.asarray(Pw, float)

    if times is None:
        times = np.array([0.1, 0.15, 0.25, 0.5, 0.75, 1.0], float) * 100.0e-9
    else:
        times = np.asarray(times, float)

    # MATLAB: P0 = P0*(7^Pw(2));
    P0_eff = float(P0) * (7.0 ** float(Pw[1]))

    # MATLAB: manager(mat,Pw(3));
    tau = float(Pw[2])
    m0, mw, e0, ew, u0, uw, xsi, z, utilda, ufront, t, x = manager_shock(mat, tau)

    # MATLAB: t = flipud(t); x = flipud(x);
    t = np.asarray(t, float)[::-1]
    x = np.asarray(x, float)[::-1, :]

    # self-similar profiles
    V_tilde = x[:, 0]
    P_tilde = x[:, 1]
    u_tilde = x[:, 2]

    # Store results per-time (lighter than huge nt×N arrays)
    out = {
        "times": times,
        "t": t,
        "x": x,
        "P0_eff": P0_eff,
        "Pw": Pw,
        "meta": dict(m0=m0, mw=mw, e0=e0, ew=ew, u0=u0, uw=uw, xsi=xsi, z=z, utilda=utilda, ufront=ufront),
        "m_shock": [],
        "x_shock": [],
        "P_shock": [],
        "u_shock": [],
        "rho_shock": [],
        "T_shock": [],
        "dPdx": [],
        "drhodx": [],
        "dTdx": [],
    }

    mw0, mw2 = float(mw[0]), float(mw[2])
    uw0, uw2 = float(uw[0]), float(uw[2])
    Pw2 = float(Pw[2])

    # loop times
    for ti in times:
        ti = float(ti)

        m_prof = (m0 * (P0_eff ** mw0) * (ti ** mw2)) * (t / xsi)
        P_prof = (P0_eff * (ti ** Pw2)) * P_tilde


        # My version of m_prof and P_prof
        m_prof = t * P0**(1/2) * mat.V0**(-1/2) * (ti)**(1+Pw2)
        P_prof = P0 * (ti**Pw2) * P_tilde

        u_prof = (ufront * ((P0_eff * 1e12) ** uw0) * (ti ** uw2)) * (u_tilde / utilda / 1e5)
        rho_prof = 1.0 / (float(mat.V0) * V_tilde)

        T_prof = (
            (
                (P0_eff * 1e12) * P_tilde * (ti ** Pw2)
                / float(mat.r) / float(mat.f)
                * (rho_prof ** (float(mat.mu) - 1.0))
            )
            ** (1.0 / float(mat.beta))
            / 11605.0
        )

        # integrate 1/rho over m to get x
        x_prof = np.zeros_like(m_prof)
        x_prof[0] = 1/rho_prof[0] 
        for i in range(1, len(m_prof)):
            dm = m_prof[i] - m_prof[i - 1]
            x_prof[i] = x_prof[i - 1] + dm / ((rho_prof[i] + rho_prof[i - 1]) / 2.0)
        

        # numeric d/dm then convert to d/dx using mid(rho)
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
        out["T_shock"].append(T_prof)
        out["dPdx"].append(dPdx)
        out["drhodx"].append(drhodx)
        out["dTdx"].append(dTdx)

    # optionally save (np.savez wants arrays, so use object arrays for ragged lists)
    if save_npz:
        # create the directory if needed
        save_npz = str(save_npz)
        
        os.makedirs(os.path.dirname(save_npz), exist_ok=True)
        np.savez(
            save_npz,
            times=times, t=t, x=x,
            P0_eff=P0_eff, Pw=Pw,
            **{k: np.array(v, dtype=object) for k, v in out.items() if isinstance(v, list)},
            **out["meta"],
        )

    return out


if __name__ == "__main__":
    # You already have your real mat/P0/Pw elsewhere. Example:
    # from materials import Au, P0, Pw
    Au = au_supersonic_variant_1()
    P0 = 10.0 # units = dyne / cm^2
    Pw = [2.0, 0.0, 0.0]  # example parameters
    data = compute_shock_profiles(Au, P0, Pw)
    pass

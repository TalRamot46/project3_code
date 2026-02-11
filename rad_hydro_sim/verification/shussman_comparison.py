# verification/shussman_comparison.py
"""
Full rad_hydro vs Shussman piecewise reference (constant temperature drive).

Reference is built from:
  1. Subsonic solver: hydrodynamic profiles from the boundary up to the shock front.
  2. Shock front: diagnosed automatically from the rad_hydro solution (max gradient in rho).
  3. Pressure at the front from the subsonic solver is used as a power-law drive P(t) = P0*t^tau.
  4. Shock solver: semi-analytic solution for the shock region driven by that P0, tau.

Physics note: For constant temperature drive we use tau=0 in the subsonic solver. If your
boundary condition or scaling differs, adjust tau or ask for clarification.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from project_3.hydro_sim.verification.compare_shock_plots import SimulationData

if TYPE_CHECKING:
    from project_3.shussman_solvers.shock_solver.materials_shock import Material
    from project_3.shussman_solvers.subsonic_solver.materials_sub import MaterialSub


def _rad_hydro_case_to_material_sub(case) -> MaterialSub:
    """Build Shussman subsonic MaterialSub from RadHydroCase (Gold-like)."""
    from project_3.shussman_solvers.subsonic_solver.materials_sub import (
        MaterialSub,
        STEFAN_BOLTZMANN,
        HEV_IN_KELVIN,
    )
    alpha = float(case.alpha)
    beta = float(case.gamma)
    lambda_ = float(case.lambda_)
    mu = float(case.mu)
    r = float(case.r)
    # Match material_au() units: f in J/g/HeV^beta, g in g/cm^2/HeV^alpha
    f = float(case.f) / (HEV_IN_KELVIN ** beta)
    g = float(case.g) / (HEV_IN_KELVIN ** alpha)
    sigma = STEFAN_BOLTZMANN
    return MaterialSub(
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        mu=mu,
        f=f,
        g=g,
        sigma=sigma,
        r=r,
        name="Au_rad_hydro",
    )


def _rad_hydro_case_to_material_shock(case) -> Material:
    """Build Shussman shock Material from RadHydroCase (same physics as subsonic)."""
    from project_3.shussman_solvers.shock_solver.materials_shock import (
        Material,
        HEV_IN_KELVIN,
    )
    alpha = float(case.alpha)
    beta = float(case.gamma)
    rho0 = float(case.rho0) if case.rho0 is not None else 19.32
    V0 = 1.0 / rho0
    f = float(case.f) / (HEV_IN_KELVIN ** beta)
    g = float(case.g) / (HEV_IN_KELVIN ** alpha)
    return Material(
        alpha=alpha,
        beta=beta,
        lambda_=float(case.lambda_),
        mu=float(case.mu),
        f=f,
        g=g,
        sigma=5.670373e-5,
        r=float(case.r),
        V0=V0,
        name="Au_shock",
    )


def diagnose_shock_front_at_times(
    times: np.ndarray,
    x_list: List[np.ndarray],
    rho_list: List[np.ndarray],
    *,
    method: str = "grad_rho",
) -> np.ndarray:
    """
    Diagnose shock front position x_shock(t) from rad_hydro history.

    method "grad_rho": position of maximum |d(rho)/dx| at each time.
    Returns array of length len(times): x_shock[i] in cm.
    """
    x_shock = np.zeros(len(times), dtype=float)
    for i in range(len(times)):
        x = np.asarray(x_list[i], float)
        rho = np.asarray(rho_list[i], float)
        if x.size < 3 or rho.size < 3:
            x_shock[i] = float(x[-1]) if x.size else 0.0
            continue
        drho_dx = np.gradient(rho, x)
        idx = np.argmax(np.abs(drho_dx))
        x_shock[i] = float(x[idx])
    return x_shock


def compute_subsonic_profiles_at_times(
    case,
    times_sec: np.ndarray,
    T0_hev: float,
    *,
    iternum: int = 3000,
    xsi0: float = 1.0,
    P0_norm: float = 4.0,
) -> dict:
    """
    Run subsonic solver at the given times (seconds) and T0 (HeV).
    Returns dict with times, m_heat, P_heat, rho_heat, u_heat, x_heat (position in cm),
    and P_front (pressure at heat front, shape (n_times,)) for use as drive.
    """
    from project_3.shussman_solvers.subsonic_solver.profiles_for_report_sub import (
        compute_profiles_for_report,
    )
    mat = _rad_hydro_case_to_material_sub(case)
    tau = float(case.tau) if case.tau is not None else 0.0  # constant T drive
    data = compute_profiles_for_report(
        mat,
        tau,
        times=times_sec,
        T0=T0_hev,
        iternum=iternum,
        xsi0=xsi0,
        P0=P0_norm,
    )
    n_times, n_xi = data["m_heat"].shape
    # Compute position x from integral dm/rho
    x_heat = np.zeros((n_times, n_xi), dtype=float)
    for i in range(n_times):
        m = data["m_heat"][i, :]
        rho = data["rho_heat"][i, :]
        x_heat[i, 0] = 0.0
        for j in range(1, n_xi):
            dm = m[j] - m[j - 1]
            rho_mid = 0.5 * (rho[j] + rho[j - 1])
            x_heat[i, j] = x_heat[i, j - 1] + dm / (rho_mid + 1e-30)
    data["x_heat"] = x_heat
    # Pressure at the front (last grid point = heat front)
    data["P_front"] = data["P_heat"][:, -1].copy()
    return data


def fit_power_law_drive(times: np.ndarray, P_front: np.ndarray) -> Tuple[float, float]:
    """
    Fit P_front(t) = P0 * t^tau (power-law drive).
    Returns (P0, tau). Uses log-linear fit; times must be > 0.
    """
    ok = (times > 0) & (P_front > 0)
    if np.sum(ok) < 2:
        return float(P_front[np.argmax(times)]), 0.0
    t = np.asarray(times[ok], float)
    p = np.asarray(P_front[ok], float)
    logt = np.log(t)
    logp = np.log(p)
    # log P = log P0 + tau * log t  =>  tau = cov(log t, log P) / var(log t)
    tau = np.polyfit(logt, logp, 1)[0]
    log_P0 = np.mean(logp - tau * logt)
    P0 = np.exp(log_P0)
    return float(P0), float(tau)


def compute_shock_profiles_at_times(
    case,
    times_sec: np.ndarray,
    P0_barye: float,
    wP2: float,
    wP3: float,
) -> dict:
    """Run shock solver with power-law drive P = P0 * t^tau. P0 in Barye (cgs)."""
    from project_3.shussman_solvers.shock_solver.run_shock_solver import compute_shock_profiles
    mat = _rad_hydro_case_to_material_shock(case)
    # Pw[0], Pw[1], Pw[2]: P0 exponent, T0 exponent, time exponent (tau)
    Pw = np.array([1, wP2, wP3], dtype=float)
    return compute_shock_profiles(mat, P0_barye, Pw, times=times_sec)


def build_piecewise_reference(
    subsonic_data: dict,
    shock_data: dict,
    x_shock_per_time: np.ndarray,
    times: np.ndarray,
) -> SimulationData:
    """
    Build a single SimulationData that is piecewise: subsonic for x < x_shock(t),
    shock for x >= x_shock(t). Profiles are concatenated and sorted by x at each time.
    """
    n_times = len(times)
    x_list: List[np.ndarray] = []
    m_list: List[np.ndarray] = []
    rho_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    e_list: List[np.ndarray] = []

    x_heat = subsonic_data["x_heat"]
    m_heat = subsonic_data["m_heat"]
    P_heat = subsonic_data["P_heat"]
    rho_heat = subsonic_data["rho_heat"]
    u_heat = subsonic_data["u_heat"]
    # e from EOS: e = P / (rho * (gamma-1)); gamma - 1 = r
    r = 0.25  # from case; subsonic doesn't return r in data
    e_heat = P_heat / (rho_heat * r + 1e-30)

    for i in range(n_times):
        x_s = x_shock_per_time[i]
        # Subsonic: take points with x <= x_shock (pre-shock region)
        x_sub = x_heat[i, :]
        mask_sub = x_sub <= x_s
        if not np.any(mask_sub):
            mask_sub = np.zeros_like(x_sub, dtype=bool)
            mask_sub[0] = True
        x_sub_sel = x_sub[mask_sub]
        m_sub_sel = m_heat[i, :][mask_sub]
        rho_sub_sel = rho_heat[i, :][mask_sub]
        p_sub_sel = P_heat[i, :][mask_sub]
        u_sub_sel = u_heat[i, :][mask_sub]
        e_sub_sel = e_heat[i, :][mask_sub]

        # Shock: x_shock and x_shock_prof from solver; place so shock front is at x_s
        x_prof = np.asarray(shock_data["x_shock"][i], float)
        rho_prof = np.asarray(shock_data["rho_shock"][i], float)
        p_prof = np.asarray(shock_data["P_shock"][i], float)
        u_prof = np.asarray(shock_data["u_shock"][i], float)
        e_prof = np.asarray(shock_data["e_shock"][i], float)
        m_prof = np.asarray(shock_data["m_shock"][i], float)
        if x_prof.size == 0:
            # Only subsonic (no shock profile at this time)
            x_all = x_sub_sel
            m_all = m_sub_sel
            rho_all = rho_sub_sel
            p_all = p_sub_sel
            u_all = u_sub_sel
            e_all = e_sub_sel
        else:
            x_front_shock = float(np.max(x_prof))
            # Lab position: shock profile runs from (x_s - length) to x_s
            x_shock_lab = x_s - (x_front_shock - x_prof)
            # Only take shock points with x_shock_lab >= last subsonic point (avoid overlap gap)
            x_min_join = float(x_sub_sel[-1]) if x_sub_sel.size else 0.0
            mask_shock = x_shock_lab >= x_min_join
            if not np.any(mask_shock):
                mask_shock = np.ones_like(x_shock_lab, dtype=bool)
            x_shock_sel = x_shock_lab[mask_shock]
            m_shock_sel = m_prof[mask_shock]
            rho_shock_sel = rho_prof[mask_shock]
            p_shock_sel = p_prof[mask_shock]
            u_shock_sel = u_prof[mask_shock]
            e_shock_sel = e_prof[mask_shock]
            # Concatenate and sort by x
            x_all = np.concatenate([x_sub_sel, x_shock_sel])
            m_all = np.concatenate([m_sub_sel, m_shock_sel])
            rho_all = np.concatenate([rho_sub_sel, rho_shock_sel])
            p_all = np.concatenate([p_sub_sel, p_shock_sel])
            u_all = np.concatenate([u_sub_sel, u_shock_sel])
            e_all = np.concatenate([e_sub_sel, e_shock_sel])
            order = np.argsort(x_all)
            x_all = x_all[order]
            m_all = m_all[order]
            rho_all = rho_all[order]
            p_all = p_all[order]
            u_all = u_all[order]
            e_all = e_all[order]

        x_list.append(x_all)
        m_list.append(m_all)
        rho_list.append(rho_all)
        p_list.append(p_all)
        u_list.append(u_all)
        e_list.append(e_all)

    return SimulationData(
        times=times,
        m=m_list,
        x=x_list,
        rho=rho_list,
        p=p_list,
        u=u_list,
        e=e_list,
        label="Shussman (subsonic + shock)",
        color="red",
        linestyle="--",
    )


def run_shussman_piecewise_reference(
    case,
    times_sec: np.ndarray,
    x_list: List[np.ndarray],
    rho_list: List[np.ndarray],
    *,
    subsonic_iternum: int = 3000,
) -> Optional[SimulationData]:
    """
    Build the piecewise Shussman reference (subsonic + shock) for comparison with rad_hydro.

    Uses rad_hydro history to diagnose shock front position at each time, runs subsonic
    at those times, fits P_front(t) = P0*t^tau, runs shock solver with that drive,
    then builds piecewise reference.
    """
    try:
        from project_3.shussman_solvers.subsonic_solver.profiles_for_report_sub import (
            compute_profiles_for_report,
        )
        from project_3.shussman_solvers.shock_solver.run_shock_solver import compute_shock_profiles
    except ImportError as e:
        print(f"  Shussman solvers not available: {e}")
        return None

    T0 = float(case.T0) if case.T0 is not None else 0.86
    # 1) Diagnose shock front from rad_hydro
    x_shock = diagnose_shock_front_at_times(times_sec, x_list, rho_list)

    # 2) Subsonic profiles at same times
    subsonic_data = compute_subsonic_profiles_at_times(
        case,
        times_sec,
        T0,
        iternum=subsonic_iternum,
    )
    P_front = subsonic_data["P_front"]
    _, wP2, wP3 = subsonic_data["Pw"]

    # 3) Shock solver with appropriate drive
    shock_data = compute_shock_profiles_at_times(case, times_sec, P_front, wP2, wP3)

    # 4) Piecewise reference
    ref = build_piecewise_reference(
        subsonic_data,
        shock_data,
        x_shock,
        times_sec,
    )
    return ref

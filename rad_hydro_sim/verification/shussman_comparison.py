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

from project_3.rad_hydro_sim.verification.hydro_data import RadHydroData
from project_3.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

if TYPE_CHECKING:
    from project_3.shussman_solvers.shock_solver.materials_shock import Material
    from project_3.shussman_solvers.subsonic_solver.materials_sub import MaterialSub


def _rad_hydro_case_to_material_sub(case) -> MaterialSub:
    """Build Shussman subsonic MaterialSub from RadHydroCase (Gold-like)."""
    from project_3.shussman_solvers.subsonic_solver.materials_sub import (
        MaterialSub,
        STEFAN_BOLTZMANN_KELVIN,
        HEV_IN_KELVIN,
    )
    alpha = float(case.alpha)
    beta = float(case.gamma)
    lambda_ = float(case.lambda_)
    mu = float(case.mu)
    r = float(case.r)
    # Match material_au() units: f in J/g/HeV^beta, g in g/cm^2/HeV^alpha
    f_Kelvin = float(case.f_Kelvin) 
    g_Kelvin = float(case.g_Kelvin) 
    return MaterialSub(
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        mu=mu,
        f=f_Kelvin,
        g=g_Kelvin,
        sigma=STEFAN_BOLTZMANN_KELVIN,
        r=r,
        name="Au_rad_hydro",
    )


def _rad_hydro_case_to_material_shock(case) -> Material:
    """Build Shussman shock Material from RadHydroCase (same physics as subsonic)."""
    from project_3.shussman_solvers.shock_solver.materials_shock import (
            Material,
            HEV_IN_KELVIN,
            STEFAN_BOLTZMANN_KELVIN
    )
    alpha = float(case.alpha)
    beta = float(case.gamma)
    rho0 = float(case.rho0) if case.rho0 is not None else 19.32
    V0 = 1.0 / rho0
    f_Kelvin = float(case.f_Kelvin) 
    g_Kelvin = float(case.g_Kelvin) 
    return Material(
        alpha=alpha,
        beta=beta,
        lambda_=float(case.lambda_),
        mu=float(case.mu),
        f=f_Kelvin,
        g=g_Kelvin,
        sigma=STEFAN_BOLTZMANN_KELVIN,
        r=float(case.r),
        V0=V0,
        name="Au_shock",
    )


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


def build_piecewise_reference(
    case,
    subsonic_data: dict,
    shock_data: dict,
    times_sec: np.ndarray,
    *,
    m_max: Optional[float] = None,
    rho0: Optional[float] = None,
    T_initial_Kelvin: Optional[float] = None,
) -> RadHydroData:
    """Piecewise Shussman: subsonic (density crossover) + shock [+ unperturbed tail]."""
    from project_3.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV, a_Hev

    times = np.asarray(times_sec, dtype=float).ravel()
    n_sub = len(subsonic_data["m_heat"])
    n_shock = len(shock_data["m_shock"])
    n = min(len(times), n_sub, n_shock)
    times = times[:n]
    r_gas = float(case.r)
    m_list, x_list, rho_list, p_list, u_list, e_list, T_list, E_list = [], [], [], [], [], [], [], []

    for k in range(n):
        m_h, x_h = np.asarray(subsonic_data["m_heat"][k], float), np.asarray(subsonic_data["x_heat"][k], float)
        rho_h, P_h = np.asarray(subsonic_data["rho_heat"][k], float), np.asarray(subsonic_data["P_heat"][k], float)
        u_h, T_h = np.asarray(subsonic_data["u_heat"][k], float), np.asarray(subsonic_data["T_heat"][k], float)
        m_sr = np.asarray(shock_data["m_shock"][k], float)
        x_sr = np.asarray(shock_data["x_shock"][k], float)
        rho_s, P_s = np.asarray(shock_data["rho_shock"][k], float), np.asarray(shock_data["P_shock"][k], float)
        u_s = np.asarray(shock_data["u_shock"][k], float)
        e_s = np.asarray(shock_data["e_shock"][k], float)
        T_s = np.asarray(shock_data["T_shock"][k], float) / KELVIN_PER_HEV

        m_trans, x_front = float(np.max(m_h)), float(x_h[-1])
        m_s, x_s = m_sr + m_trans, x_sr + x_front
        rel = np.where(m_s >= m_trans)[0]
        rel = rel[:-1] if len(rel) > 1 else rel

        if len(rel) == 0:
            m_k, x_k = m_h, x_h
            rho_k, p_k, u_k = rho_h, P_h, u_h
            e_k, T_k = P_h / (rho_h * r_gas + 1e-30), T_h
        else:
            rh = np.where(rho_h <= rho_s[rel[0]])[0]
            m_k = np.concatenate([m_h[rh], m_s[rel]])
            x_k = np.concatenate([x_h[rh], x_s[rel]])
            rho_k = np.concatenate([rho_h[rh], rho_s[rel]])
            p_k = np.concatenate([P_h[rh], P_s[rel]])
            u_k = np.concatenate([u_h[rh], u_s[rel]])
            e_k = np.concatenate([P_h[rh] / (rho_h[rh] * r_gas + 1e-30), e_s[rel]])
            T_k = np.concatenate([T_h[rh], T_s[rel]])

        if m_max and rho0 and T_initial_Kelvin and m_max > m_k[-1]:
            e0 = float(case.f_Kelvin) * (float(T_initial_Kelvin) ** float(case.gamma)) * (float(rho0) ** (-float(case.mu)))
            p0, T0 = r_gas * rho0 * e0, float(T_initial_Kelvin) / KELVIN_PER_HEV
            m_end = float(m_k[-1])
            m_ε = m_end + 1e-12  # step point: sharp jump to unperturbed
            x_ε = float(x_k[-1]) + (m_ε - m_end) / float(rho_k[-1])
            x_max = x_ε + (m_max - m_ε) / rho0
            m_k = np.concatenate([m_k, [m_ε, m_max]])
            x_k = np.concatenate([x_k, [x_ε, x_max]])
            rho_k = np.concatenate([rho_k, [rho0, rho0]])
            p_k = np.concatenate([p_k, [p0, p0]])
            u_k = np.concatenate([u_k, [0.0, 0.0]])
            e_k = np.concatenate([e_k, [e0, e0]])
            T_k = np.concatenate([T_k, [T0, T0]])

        E_list.append(a_Hev * T_k**4)
        m_list.append(m_k)
        x_list.append(x_k)
        rho_list.append(rho_k)
        p_list.append(p_k)
        u_list.append(u_k)
        e_list.append(e_k)
        T_list.append(T_k)

    return RadHydroData(times=times, m=m_list, x=x_list, rho=rho_list, p=p_list, u=u_list, e=e_list, T=T_list, E_rad=E_list, label="Shussman (piecewise)", color="green", linestyle="-.")

def run_shussman_piecewise_reference(
    case,
    times_ns: np.ndarray,
    T0_HeV: float,
) -> Optional[RadHydroData]:
    """
    Build the piecewise Shussman reference (subsonic + shock [+ unperturbed tail]) for comparison with rad_hydro.

    Uses rad_hydro history to diagnose shock front position at each time, runs subsonic
    at those times, fits P_front(t) = P0*t^tau, runs shock solver with that drive,
    then builds piecewise reference.     If case has x_max, rho0, T_initial_Kelvin, appends
    the unperturbed region from the end of the shock to m_max = x_max * rho0.
    """
    # importing the subsonic & shock solvers
    from project_3.shussman_solvers.subsonic_solver.profiles_for_report_sub import compute_profiles_for_report
    from project_3.shussman_solvers.shock_solver.profiles_for_report_shock import (
        compute_shock_profiles,
    )


    T0_HeV = float(case.T0_Kelvin) / KELVIN_PER_HEV
    # 1) Subsonic profiles at same times
    print("Starting subsonic solving...")
    mat = _rad_hydro_case_to_material_sub(case)
    tau = float(case.tau) if case.tau is not None else 0.0  # constant T drive
    subsonic_data = compute_profiles_for_report(
        mat = mat,
        T0_phys_HeV=T0_HeV,
        tau=tau,
        times_ns=times_ns,
    )
    P_front = subsonic_data["P_heat"][-1,-1] # units = MBar
    _, _, wP3 = subsonic_data["Pw"]

    # 2) Shock solver with appropriate drive
    print("Starting shock solving...")
    mat = _rad_hydro_case_to_material_shock(case)
    # Pw[0], Pw[1], Pw[2]: P0 exponent, T0 exponent, time exponent (tau)
    Pw = np.array([_, _, wP3], dtype=float)
    shock_data = compute_shock_profiles(mat, P_front, tau=None, Pw=Pw, times_ns=times_ns, 
                                            patching_method=True, save_npz=None)

    # making sure that both shocK_data and subsonic_data include T in Kelvin (they both return T in HeV)
    import matplotlib.pyplot as plt
    plt.plot(subsonic_data["m_heat"][-1], subsonic_data["T_heat"][-1])
    plt.plot(shock_data["m_shock"][-1], shock_data["T_shock"][-1])
    plt.show()
    

    # 3) Piecewise reference (subsonic + shock [+ unperturbed to m_max])
    print("Starting building piecewise reference...")
    rho0 = float(case.rho0) if case.rho0 is not None else None
    T_init = float(case.T_initial_Kelvin) if case.T_initial_Kelvin is not None else None
    m_max = float(case.x_max) * float(case.rho0) if (case.x_max is not None and rho0 is not None) else None
    ref = build_piecewise_reference(
        case,
        subsonic_data,
        shock_data,
        times_ns,
        m_max=m_max,
        rho0=rho0,
        T_initial_Kelvin=T_init,
    )
    print("times_ns of ablation region: ", times_ns)
    return ref


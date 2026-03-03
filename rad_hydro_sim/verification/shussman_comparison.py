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
    f = float(case.f) 
    g = float(case.g) 
    sigma = STEFAN_BOLTZMANN_KELVIN * HEV_IN_KELVIN**4
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
) -> RadHydroData:
    """
    Piecewise Shussman reference: subsonic (rear to Marshak front) + shock region.

    Assumes ``subsonic_data`` and ``shock_data`` are valid and aligned in time.
    For each time step:
      - take subsonic profiles from x=0 up to the Marshak front,
      - append shock profiles, offset in x and m so that the front is continuous.
    """
    times = np.asarray(times_sec, dtype=float)

    m_list: List[np.ndarray] = []
    x_list: List[np.ndarray] = []
    rho_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    e_list: List[np.ndarray] = []
    T_list: List[np.ndarray] = []
    E_rad_list: List[np.ndarray] = []

    r_gas = float(case.r)
    from project_3.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV, a_Hev

    for k, _t in enumerate(times):
        # Subsonic region (from boundary to Marshak front)
        m_sub = np.asarray(subsonic_data["m_heat"][k, :], float)
        x_sub = np.asarray(subsonic_data["x_heat"][k, :], float)
        rho_sub = np.asarray(subsonic_data["rho_heat"][k, :], float)
        p_sub = np.asarray(subsonic_data["P_heat"][k, :], float)
        u_sub = np.asarray(subsonic_data["u_heat"][k, :], float)
        T_sub = np.asarray(subsonic_data["T_heat"][k, :], float)
        # Subsonic T_heat is 100*T0*... (T0 in Hev) -> T_Hev = T_sub/100
        T_sub_Hev = T_sub / 100.0
        # Ideal-gas-like specific internal energy, consistent with shock solver
        e_sub = p_sub / (rho_sub * r_gas + 1e-30)

        # Shock region (coordinates measured from front in shock solver)
        a = shock_data["m_shock"]
        s = shock_data["m_shock"][k]
        m_shock_rel = np.asarray(s, float)
        x_shock_rel = np.asarray(shock_data["x_shock"][k], float)
        rho_shock = np.asarray(shock_data["rho_shock"][k], float)
        p_shock = np.asarray(shock_data["P_shock"][k], float)
        u_shock = np.asarray(shock_data["u_shock"][k], float)
        e_shock = np.asarray(shock_data["e_shock"][k], float)
        T_shock = np.asarray(shock_data["T_shock"][k], float)
        # Shock T is in Kelvin -> T_Hev = T_K / KELVIN_PER_HEV
        T_shock_Hev = T_shock / KELVIN_PER_HEV

        x_front = x_sub[-1]
        m_front = m_sub[-1]

        x_shock = x_shock_rel + x_front
        m_shock = m_shock_rel + m_front

        m_k = np.concatenate([m_sub, m_shock])
        x_k = np.concatenate([x_sub, x_shock])
        rho_k = np.concatenate([rho_sub, rho_shock])
        p_k = np.concatenate([p_sub, p_shock])
        u_k = np.concatenate([u_sub, u_shock])
        e_k = np.concatenate([e_sub, e_shock])
        T_k_Hev = np.concatenate([T_sub_Hev, T_shock_Hev])
        E_rad_k = a_Hev * T_k_Hev**4

        m_list.append(m_k)
        x_list.append(x_k)
        rho_list.append(rho_k)
        p_list.append(p_k)
        u_list.append(u_k)
        e_list.append(e_k)
        T_list.append(T_k_Hev)
        E_rad_list.append(E_rad_k)
    return RadHydroData(
        times=times,
        m=m_list,
        x=x_list,
        rho=rho_list,
        p=p_list,
        u=u_list,
        e=e_list,
        T=T_list,
        E_rad=E_rad_list,
        label="Shussman (piecewise)",
        color="green",
        linestyle="-.",
    )

def run_shussman_piecewise_reference(
    case,
    times_ns: np.ndarray,
    T0_HeV: float,
) -> Optional[RadHydroData]:
    """
    Build the piecewise Shussman reference (subsonic + shock) for comparison with rad_hydro.

    Uses rad_hydro history to diagnose shock front position at each time, runs subsonic
    at those times, fits P_front(t) = P0*t^tau, runs shock solver with that drive,
    then builds piecewise reference.
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
    print("times_ns of ablation region: ", times_ns)

    # 2) Shock solver with appropriate drive
    print("Starting shock solving...")
    mat = _rad_hydro_case_to_material_shock(case)
    # Pw[0], Pw[1], Pw[2]: P0 exponent, T0 exponent, time exponent (tau)
    Pw = np.array([1, _, tau], dtype=float)
    shock_data = compute_shock_profiles(mat, P_front, tau=tau, Pw=Pw, times_ns=times_ns, 
                                            patching_method=True, save_npz=None)
    
    # 3) Piecewise reference (subsonic + shock)
    print("Starting building piecewise reference...")
    ref = build_piecewise_reference(
        case,
        subsonic_data,
        shock_data,
        times_ns,
    )
    print("times_ns of ablation region: ", times_ns)
    return ref


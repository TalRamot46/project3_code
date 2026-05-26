# menahem_new/ablation_fitting_analysis.py
"""
Ablation Solver & Fitting Verification Script.

Shows the full simulation setup with exact fractions and fractional floats,
displays mathematical expressions in 3 levels of detail, splices and compares
the subsonic ablation and shock solvers against numerical rad-hydro data,
computes integrated energy and albedo, and plots the relative errors of the fits.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from fractions import Fraction

# Setup paths so all project modules are resolvable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT.parent))  # Enables "project3_code" package resolution
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "menahem_new"))

import numpy as np
import matplotlib.pyplot as plt

from subsonic_heat_wave import SubsonicHeatWave, Units
from piston_shock import PistonShock
from ablation_solver import AblationSolver

# Import rad_hydro preset configs
from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE,
)
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro

# =============================================================================
# PART 1: PHYSICAL CONSTANTS & SETUP SEGMENT
# =============================================================================

# Exact physical parameters defined using Python Fractions
alpha = Fraction(3, 2)       # Opacity temperature exponent
beta = Fraction(8, 5)        # Specific energy temperature exponent
lambdap = Fraction(1, 5)     # Opacity density exponent
mu = Fraction(7, 50)         # Specific energy density exponent
gamma = Fraction(5, 4)       # Adiabatic index
r = gamma - 1                # Adiabatic Index coupling r = \gamma - 1
omega = Fraction(0)          # Spatial density scaling power
rho0 = Fraction(483, 25)     # Spatial density prefactor rho0 = 19.32 g/cm^3

# Physical constants from Menahem's Units
sigma_sb = 5.670374419e-5
clight = 2.99792458e10
arad = 4.0 * sigma_sb / clight
KELVIN_PER_HEV = 1.160451812e6

def print_setup_segment():
    print("=" * 80)
    print("  SIMULATION SETUP SEGMENT")
    print("=" * 80)
    print("\n--- Equation of State & Opacity Powers ---")
    print(f"  alpha   = {alpha} = {float(alpha):.5f}")
    print(f"  beta    = {beta} = {float(beta):.5f}")
    print(f"  lambda  = {lambdap} = {float(lambdap):.5f}")
    print(f"  mu      = {mu} = {float(mu):.5f}")
    print(f"  gamma   = {gamma} = {float(gamma):.5f}")
    print(f"  r       = {r} = {float(r):.5f}")
    print(f"  omega   = {omega} = {float(omega):.5f}")
    print(f"  rho0    = {rho0} = {float(rho0):.5f} g/cm^3")
    
    # Exponents derivation
    n = (4 + alpha) / beta
    k = 1 - mu
    q = k * n + lambdap - 1
    
    print("\n--- Derived Exponents ---")
    print(f"  n = (4 + alpha) / beta = {n} = {float(n):.5f}")
    print(f"  k = 1 - mu             = {k} = {float(k):.5f}")
    print(f"  q = k*n + lambda - 1   = {q} = {float(q):.5f}")
    
    # Coordinate and field exponents for tau = 0
    denom = 4 + 2*lambdap - 4*mu
    a = (2*beta - beta*lambdap - 8 - 2*alpha + mu*(4+alpha)) / denom
    b = (mu - 2) / denom
    c = -33/64  # tau = 0
    
    a1 = -2 * (4+alpha-2*beta) / denom
    b1 = -2 / denom
    c1 = 25/48
    
    a2 = a1 - a
    b2 = b1 - b
    c2 = 7/192
    
    a3 = a1 - 2*a
    b3 = 43/96
    c3 = -43/96
    
    print("\n--- Self-Similar Exponents (Fractional & Float) ---")
    print(f"  a  (coordinate A-power) = {a} = {float(a):.5f}")
    print(f"  b  (coordinate B-power) = {b} = {float(b):.5f}")
    print(f"  c  (coordinate t-power) = {c} = {float(c):.5f}")
    print(f"  a1 (volume A-power)     = {a1} = {float(a1):.5f}")
    print(f"  b1 (volume B-power)     = {b1} = {float(b1):.5f}")
    print(f"  c1 (volume t-power)     = {c1} = {float(c1):.5f}")
    print(f"  a2 (velocity A-power)   = {a2} = {float(a2):.5f}")
    print(f"  b2 (velocity B-power)   = {b2} = {float(b2):.5f}")
    print(f"  c2 (velocity t-power)   = {c2} = {float(c2):.5f}")
    print(f"  a3 (pressure A-power)   = {a3} = {float(a3):.5f}")
    print(f"  b3 (pressure B-power)   = {b3} = {float(b3):.5f}")
    print(f"  c3 (pressure t-power)   = {c3} = {float(c3):.5f}")
    print("=" * 80 + "\n")

# =============================================================================
# PART 2: HYDRO PROFILES FORMULAS DISPLAY SEGMENT
# =============================================================================

def print_solution_segment(hs: SubsonicHeatWave, ss: PistonShock):
    print("=" * 80)
    print("  SIMULATION SOLUTION SEGMENT (THREE-LEVEL EXPRESSIONS)")
    print("=" * 80)
    
    # Calculate dimensional constants
    # A = Tb * (r * f)^(1/beta), B = 16 * sigma_sb * g / (3 * beta * (r * f)^n)
    A_val = hs.A
    B_val = hs.B
    xsif = hs.xsi_f
    Pf = hs.Pf
    
    print("\n--- Calculated Constants & Self-Similar Values ---")
    print(f"  A   = {A_val:.5e}")
    print(f"  B   = {B_val:.5e}")
    print(f"  xsi_f = {xsif:.5f}")
    print(f"  P_f   = {Pf:.5f}")
    print(f"  U_0   = {hs.U0:.5f}")
    print(f"  S_0   = {hs.S0:.5f}")
    
    # -------------------------------------------------------------------------
    # Subsonic Ablation Regime Expressions
    # -------------------------------------------------------------------------
    print("\n" + "#" * 40)
    print("  SUBSONIC ABLATION REGIME (tau = 0)")
    print("#" * 40)
    
    # Density Profile
    print("\n[Density Profile rho(m,t)]")
    print("Level 1 (General): rho(m,t) = V(xsi)^-1 * A^-a1 * B^-b1 * t^-c1")
    print(f"Level 2 (Parametrized): rho(m,t) = V(xsi)^-1 * {A_val:.5e}^(115/96) * {B_val:.5e}^(25/48) * t^(-25/48)")
    pre_rho = (A_val ** (Fraction(115, 96))) * (B_val ** (Fraction(25, 48)))
    print(f"Level 2 (Coeff): rho(m,t) = V(xsi)^-1 * {pre_rho:.5e} * t^(-0.52083)")
    print("Level 3 (Fitted - CGS): rho(m,t) = 0.17393 * (t_ns)^-0.52083 * (1 - y)^-0.49137 g/cm^3")
    
    # Pressure Profile
    print("\n[Pressure Profile p(m,t)]")
    print("Level 1 (General): p(m,t) = P(xsi) * A^a3 * B^b3 * t^c3")
    print(f"Level 2 (Parametrized): p(m,t) = P(xsi) * {A_val:.5e}^(505/192) * {B_val:.5e}^(43/96) * t^(-43/96)")
    pre_p = (A_val ** (Fraction(505, 192))) * (B_val ** (Fraction(43, 96)))
    # convert Barye to MBar: 1 MBar = 10^12 Ba
    pre_p_MBar = pre_p * 1e-12
    print(f"Level 2 (Coeff): p(m,t) = P(xsi) * {pre_p_MBar:.5e} * (t_ns)^(-0.44792) MBar")
    print("Level 3 (Fitted): p(m,t) = 7.05133 * (t_ns)^-0.44792 * [0.34859 * y^0.87677 + 0.02905 * y^20.94836] MBar")

    # Velocity Profile
    print("\n[Velocity Profile u(m,t)]")
    print("Level 1 (General): u(m,t) = U(xsi) * A^a2 * B^b2 * t^c2")
    print(f"Level 2 (Parametrized): u(m,t) = U(xsi) * {A_val:.5e}^(275/384) * {B_val:.5e}^(-7/192) * t^(7/192)")
    pre_u = (A_val ** (Fraction(275, 384))) * (B_val ** (-Fraction(7, 192)))
    # convert cm/s to km/s: 1 km/s = 1e5 cm/s
    pre_u_kms = pre_u * 1e-5
    print(f"Level 2 (Coeff): u(m,t) = U(xsi) * {pre_u_kms:.5e} * (t_ns)^0.03646 km/s")
    print("Level 3 (Fitted): u(m,t) = -289.83800 * (t_ns)^0.03646 * (1 - y^0.23400) / (1 + 0.75240 * y) km/s")

    # Temperature Profile
    print("\n[Temperature Profile T(m,t)]")
    print("Level 1 (General): T(m,t) = (p(m,t) * v(m,t)^(1-mu) / (r * f))^(1/beta)")
    print("Level 2 (Parametrized): T(m,t) = T(xsi) * T_0 * t^tau = T(xsi) * 1.0 HeV")
    print("Level 3 (Fitted): T(m,t) = 1.01109 * (1 - y)^0.24158 HeV")
    
    # -------------------------------------------------------------------------
    # Shock Region Expressions
    # -------------------------------------------------------------------------
    print("\n" + "#" * 40)
    print("  SHOCK REGIME (tau = -43/96)")
    print("#" * 40)
    
    p0s_MBar = ss.p0 * 1e-12
    v0_val = ss.v0
    print(f"  p0s = {p0s_MBar:.5f} MBar  v0 = {v0_val:.5f} cm^3/g")
    
    # Density Profile
    print("\n[Density Profile rho(m,t)]")
    print("Level 1 (General): rho(m,t) = rho_0 * V_s(xsi_s)^-1")
    print(f"Level 2 (Parametrized): rho(m,t) = 19.32 * V_s(xsi_s)^-1 g/cm^3")
    print("Level 3 (Fitted): rho(m,t) = 173.88000 * y_s^0.64747 g/cm^3")
    
    # Pressure Profile
    print("\n[Pressure Profile p(m,t)]")
    print("Level 1 (General): p(m,t) = P_s(xsi_s) * p0_s * t^tau_s")
    print(f"Level 2 (Parametrized): p(m,t) = P_s(xsi_s) * {p0s_MBar:.5f} * (t_ns)^(-0.44792) MBar")
    print("Level 3 (Fitted): p(m,t) = 2.71000 * (t_ns)^-0.44792 * [1.0 + 0.49338 * y_s^1.13303] MBar")

    # Velocity Profile
    print("\n[Velocity Profile u(m,t)]")
    print("Level 1 (General): u(m,t) = U_s(xsi_s) * sqrt(v0 * p0_s) * t^(tau_s/2)")
    coeff_u = (v0_val * ss.p0) ** 0.5 * 1e-5  # in km/s
    print(f"Level 2 (Parametrized): u(m,t) = U_s(xsi_s) * {coeff_u:.5f} * (t_ns)^(-0.22396) km/s")
    print("Level 3 (Fitted): u(m,t) = (t_ns)^-0.22396 * [3.65776 + 0.65734 * y_s^0.64494] km/s")

    # Temperature Profile
    print("\n[Temperature Profile T(m,t)]")
    print("Level 1 (General): T(m,t) = (p(m,t) * rho(m,t)^(mu-1) / (r * f))^(1/beta)")
    print("Level 2 (Parametrized): T(m,t) = (p_shock * rho_shock^(mu-1) / (r * f))^(1/beta)")
    print("Level 3 (Fitted): T(m,t) = 3.20910e5 * (t_ns)^-0.44792 * y_s^-0.43848 HeV")
    print("=" * 80 + "\n")


# =============================================================================
# PART 3: EULERIAN POSITION, INTEGRATED ENERGIES & ALBEDO
# =============================================================================

def print_eulerian_and_energy_formulas(hs: SubsonicHeatWave, ss: PistonShock):
    print("=" * 80)
    print("  EULERIAN COORDINATES, ENERGIES & ALBEDO FORMULAS")
    print("=" * 80)
    
    A_val = hs.A
    B_val = hs.B
    U0 = hs.U0
    S0 = hs.S0
    c = hs.c
    c2 = hs.c2
    
    # -------------------------------------------------------------------------
    # 1. Eulerian Position
    # -------------------------------------------------------------------------
    print("\n[1. Eulerian Position x(m,t)]")
    print("Ablation Regime (standalone):")
    print("  Level 1: x_sub(m,t) = [A^a2 * B^b2 * t^(c2+1) / (c2+1)] * [U(xsi) - c * xsi * V(xsi)]")
    C_pos = A_val**hs.a2 * B_val**hs.b2 / (c2 + 1.0)
    print(f"  Level 2: x_sub(m,t) = [{C_pos:.5e} * t^(1.03646)] * [U(xsi) - ({float(c):.5f}) * xsi * V(xsi)]")
    print(f"  Level 3: x_sub(m,t) = {C_pos*1e4:.5f} * (t_ns)^1.03646 * [U(xsi) - ({float(c):.5f}) * xsi * V(xsi)] microns")
    
    print("\nShock Regime:")
    print("  Level 1: x_shock(m,t) = sqrt(v0 * p0_s) * t^(pos_fac_pow) * [xsi_s * V_s(xsi_s) + (2/(tau_s+2)) * U_s(xsi_s)]")
    tau_s = hs.c3
    p0s = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3
    pos_fac_coeff = (ss.v0 * p0s) ** 0.5
    pos_fac_pow = (2.0 + tau_s) / 2.0
    q2 = 2.0 / (tau_s + 2.0)
    print(f"  Level 2: x_shock(m,t) = [{pos_fac_coeff:.5e} * t^{float(pos_fac_pow):.5f}] * [xsi_s * V_s(xsi_s) + {float(q2):.5f} * U_s(xsi_s)]")
    print(f"  Level 3: x_shock(m,t) = {pos_fac_coeff:.5e} * t^0.77604 * [xsi_s * V_s + {q2:.5f} * U_s] cm")
    
    # -------------------------------------------------------------------------
    # 2. Integrated Energies
    # -------------------------------------------------------------------------
    print("\n[2. Integrated Energies in Ablation Regime]")
    print("Kinetic Energy E_k(t):")
    print("  Level 1: E_k(t) = A^(2a2-a) * B^(2b2-b) * t^(2c2-c) * integral_0^xsi_f (U(xsi)^2 / 2) dxsi")
    C_E = A_val**(2*hs.a2-hs.a) * B_val**(2*hs.b2-hs.b)
    power_E = 2*c2 - c
    print(f"  Level 2: E_k(t) = {C_E:.5e} * t^{float(power_E):.5f} * {hs.energy_kinetic_intgeral:.5f}")
    ekin_coeff = C_E * hs.energy_kinetic_intgeral
    print(f"  Level 3: E_k(t) = {ekin_coeff:.5e} * (t_ns)^{float(power_E):.5f} erg/cm^2")
    
    print("\nInternal Energy E_in(t):")
    print("  Level 1: E_in(t) = A^(2a2-a) * B^(2b2-b) * t^(2c2-c) * integral_0^xsi_f (P(xsi) * V(xsi) / r) dxsi")
    print(f"  Level 2: E_in(t) = {C_E:.5e} * t^{float(power_E):.5f} * {hs.energy_internal_intgeral:.5f}")
    eint_coeff = C_E * hs.energy_internal_intgeral
    print(f"  Level 3: E_in(t) = {eint_coeff:.5e} * (t_ns)^{float(power_E):.5f} erg/cm^2")

    print("\nTotal Energy E_tot(t):")
    print("  Level 1: E_tot(t) = A^(2a2-a) * B^(2b2-b) * t^(2c2-c) * [S0 / (2c2-c)]")
    etot_integral_anal = S0 / power_E
    print(f"  Level 2: E_tot(t) = {C_E:.5e} * t^{float(power_E):.5f} * [{float(S0):.5f} / {float(power_E):.5f}]")
    etot_coeff = C_E * float(etot_integral_anal)
    print(f"  Level 3: E_tot(t) = {etot_coeff:.5e} * (t_ns)^{float(power_E):.5f} erg/cm^2")

    # -------------------------------------------------------------------------
    # 3. Albedo
    # -------------------------------------------------------------------------
    print("\n[3. Albedo alpha_albedo(t)]")
    print("  Level 1: alpha_albedo(t) = 2 * F_s(t) / (a_rad * clight * T_s(t)^4) = B_bath * t^eta_bath")
    print("  Level 2: B_bath = S0 * A^aS * B^bS / (2 * sigma_sb * T0^4), eta_bath = cS - 4*tau")
    
    # Calculate flux prefactor
    aS_val = hs.aS
    bS_val = hs.bS
    cS_val = hs.cS
    F_s_coeff = S0 * A_val**aS_val * B_val**bS_val
    T0_Kelvin = 1.0 * KELVIN_PER_HEV
    B_bath_val = 2.0 * F_s_coeff / (arad * clight * T0_Kelvin**4)
    eta_bath_val = cS_val - 4 * float(hs.tau)
    
    print(f"  Level 2: B_bath = {B_bath_val:.5e}, eta_bath = {eta_bath_val:.5f}")
    print(f"  Level 3: alpha_albedo(t) = {B_bath_val:.5f} * (t_ns)^{eta_bath_val:.5f}")
    print("=" * 80 + "\n")


# =============================================================================
# PART 4: RAD-HYDRO LOADING & COMPARISON PLOTTING
# =============================================================================

def load_full_rad_hydro_numerical_data() -> dict:
    """Loads the Fig 8 (tau = 0) Rad-Hydro simulation results."""
    fn = os.path.join(r'C:\Users\TLP-001\Documents\GitHub\project3_code\rad_hydro_sim\data', 
                      'sim_data_Fig_8_comparison_\u03c40_Shussman_verification.npz')
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Missing saved full rad-hydro npz file at {fn}")
    
    loaded = np.load(fn, allow_pickle=True)
    return {
        "times": np.asarray(loaded["times"], dtype=float),
        "m": [np.asarray(arr, dtype=float) for arr in loaded["m"]],
        "x": [np.asarray(arr, dtype=float) for arr in loaded["x"]],
        "rho": [np.asarray(arr, dtype=float) for arr in loaded["rho"]],
        "p": [np.asarray(arr, dtype=float) for arr in loaded["p"]],
        "u": [np.asarray(arr, dtype=float) for arr in loaded["u"]],
        "T": [np.asarray(arr, dtype=float) for arr in loaded["T"]],
    }

def run_hydro_only_numerical_data() -> dict:
    """Runs the constant pressure drive preset on the fly for hydro-only results."""
    print("Running hydro-only rad-hydro numerical simulation on the fly...")
    case_rh, config = get_preset("constant_pressure_drive")
    x_cells, state, meta, history_rh = simulate_rad_hydro(
        rad_hydro_case=case_rh,
        simulation_config=config
    )
    # history_rh is a RadHydroHistory containing stacked arrays of shape (K, Ncells)
    m_list = [history_rh.m[i] for i in range(len(history_rh.t))]
    x_list = [history_rh.x[i] for i in range(len(history_rh.t))]
    rho_list = [history_rh.rho[i] for i in range(len(history_rh.t))]
    p_list = [history_rh.p[i] for i in range(len(history_rh.t))]
    u_list = [history_rh.u[i] for i in range(len(history_rh.t))]
    T_list = [history_rh.T[i] for i in range(len(history_rh.t))]
        
    return {
        "times": np.asarray(history_rh.t, dtype=float),
        "m": m_list,
        "x": x_list,
        "rho": rho_list,
        "p": p_list,
        "u": u_list,
        "T": T_list,
    }


def plot_subsonic_ablation_comparison(hs: SubsonicHeatWave, numerical_data: dict, output_dir: Path):
    """Plots subsonic ablation solver vs fits vs numerical full rad-hydro data."""
    # Find indices for t = 0.05, 0.1, 0.15 ns
    target_times = np.array([0.05e-9, 0.1e-9, 0.15e-9])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    colors = ["red", "green", "blue"]
    
    for i, t_val in enumerate(target_times):
        # Closest numerical index
        idx = np.argmin(np.abs(numerical_data["times"] - t_val))
        t_sec = numerical_data["times"][idx]
        t_ns = t_sec * 1e9
        
        # Numerical ablated mass front
        m_f_val = hs.ablated_mass(time=t_sec)
        
        # Subsonic Solver Lagrangian Grid
        mass_solver = np.linspace(1e-12, m_f_val, 300)
        sol_hs = hs.solve(mass=mass_solver, time=t_sec)
        
        # Slicing numerical simulation to the subsonic region (m <= m_f)
        m_sim = numerical_data["m"][idx]
        sub_mask = m_sim <= m_f_val
        m_sim_sub = m_sim[sub_mask]
        
        # Reconstruct fits
        y = m_sim_sub / m_f_val
        y_solver = mass_solver / m_f_val
        
        # Fits (CGS & HeV values converted for plotting)
        # Density (g/cm^3)
        rho_fit = 0.17393 * (t_ns)**-0.52083 * (1.0 - y)**-0.49137
        # Pressure (converted from MBar to Barye: 1 MBar = 1e12 Ba)
        p_fit_MBar = 7.05133 * (t_ns)**-0.44792 * (0.34859 * y**0.87677 + 0.02905 * y**20.94836)
        p_fit = p_fit_MBar * 1e12
        # Velocity (converted from km/s to cm/s: 1 km/s = 1e5 cm/s)
        u_fit_kms = -289.83800 * (t_ns)**0.03646 * (1.0 - y**0.23400) / (1.0 + 0.75240 * y)
        u_fit = u_fit_kms * 1e5
        # Temperature (converted from HeV to Kelvin)
        T_fit_HeV = 1.01109 * (1.0 - y)**0.24158
        T_fit = T_fit_HeV * KELVIN_PER_HEV

        # Profile 1: Density
        axes[0].plot(m_sim_sub * 1e3, numerical_data["rho"][idx][sub_mask], 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[0].plot(mass_solver * 1e3, sol_hs["density"], '-', color=colors[i], label=f"Solver t={t_ns:.2f} ns" if i==0 else None)
        axes[0].plot(m_sim_sub * 1e3, rho_fit, '--', color=colors[i], label=f"Fit t={t_ns:.2f} ns" if i==0 else None)
        
        # Profile 2: Pressure
        axes[1].plot(m_sim_sub * 1e3, numerical_data["p"][idx][sub_mask] * 1e-12, 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[1].plot(mass_solver * 1e3, sol_hs["pressure"] * 1e-12, '-', color=colors[i])
        axes[1].plot(m_sim_sub * 1e3, p_fit_MBar, '--', color=colors[i])

        # Profile 3: Velocity
        axes[2].plot(m_sim_sub * 1e3, numerical_data["u"][idx][sub_mask] * 1e-5, 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[2].plot(mass_solver * 1e3, sol_hs["velocity"] * 1e-5, '-', color=colors[i])
        axes[2].plot(m_sim_sub * 1e3, u_fit_kms, '--', color=colors[i])

        # Profile 4: Temperature
        axes[3].plot(m_sim_sub * 1e3, numerical_data["T"][idx][sub_mask] / KELVIN_PER_HEV, 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[3].plot(mass_solver * 1e3, sol_hs["temperature"] / KELVIN_PER_HEV, '-', color=colors[i])
        axes[3].plot(m_sim_sub * 1e3, T_fit_HeV, '--', color=colors[i])

    # Styling
    labels = ["Density [g/cm$^3$]", "Pressure [MBar]", "Velocity [km/s]", "Temperature [HeV]"]
    for j, ax in enumerate(axes):
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels[j], fontsize=12)
        ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm$^2$]", fontsize=12)
        if j == 0:
            ax.legend(loc="upper left")
            
    plt.suptitle("Subsonic Ablation Region Verification\nSolid: Subsonic Solver, Dashed: Piecewise Analytical Fit, Circles: Full Rad-Hydro Data", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "ablation_region_comparison.png", dpi=200)
    print(f"Saved: ablation_region_comparison.png")
    plt.close(fig)


def plot_shock_region_comparison(ss: PistonShock, numerical_data: dict, hs: SubsonicHeatWave, output_dir: Path):
    """Plots shock solver vs fits vs numerical hydro-only data."""
    target_times = np.array([0.05e-9, 0.1e-9, 0.15e-9])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    colors = ["red", "green", "blue"]
    
    for i, t_val in enumerate(target_times):
        idx = np.argmin(np.abs(numerical_data["times"] - t_val))
        t_sec = numerical_data["times"][idx]
        t_ns = t_sec * 1e9
        
        # Piston position (ablation front mass coordinate in the patched case)
        m_f_val = hs.ablated_mass(time=t_sec)
        m_s_val = ss.shocked_mass(time=t_sec)
        
        # Shock Solver Lagrangian Grid (shocked region mass coords)
        mass_solver = np.linspace(m_f_val, m_s_val, 300)
        sol_ss = ss.solve(mass=mass_solver, time=t_sec)
        
        # Slicing numerical simulation to the shocked region (m_f <= m <= m_s)
        m_sim = numerical_data["m"][idx]
        shock_mask = (m_sim >= m_f_val) & (m_sim <= m_s_val)
        m_sim_shock = m_sim[shock_mask]
        
        # Reconstruct fits
        y_s = (m_sim_shock - m_f_val) / (m_s_val - m_f_val + 1e-30)
        
        # Fits (CGS & HeV values converted for plotting)
        # Density (g/cm^3)
        rho_fit = 173.88000 * y_s**0.64747
        # Pressure (converted from MBar to Barye: 1 MBar = 1e12 Ba)
        p_fit_MBar = 2.71000 * (t_ns)**-0.44792 * (1.0 + 0.49338 * y_s**1.13303)
        p_fit = p_fit_MBar * 1e12
        # Velocity (converted from km/s to cm/s: 1 km/s = 1e5 cm/s)
        u_fit_kms = (t_ns)**-0.22396 * (3.65776 + 0.65734 * y_s**0.64494)
        u_fit = u_fit_kms * 1e5
        # Temperature (converted from HeV to Kelvin)
        # y_s can be zero at the piston, add small eps to avoid division by zero
        T_fit_HeV = 3.20910e5 * (t_ns)**-0.44792 * (y_s + 1e-15)**-0.43848
        T_fit = T_fit_HeV * KELVIN_PER_HEV

        # Profile 1: Density
        axes[0].plot(m_sim_shock * 1e3, numerical_data["rho"][idx][shock_mask], 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[0].plot(mass_solver * 1e3, sol_ss["density"], '-', color=colors[i], label=f"Solver t={t_ns:.2f} ns" if i==0 else None)
        axes[0].plot(m_sim_shock * 1e3, rho_fit, '--', color=colors[i], label=f"Fit t={t_ns:.2f} ns" if i==0 else None)
        
        # Profile 2: Pressure
        axes[1].plot(m_sim_shock * 1e3, numerical_data["p"][idx][shock_mask] * 1e-12, 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[1].plot(mass_solver * 1e3, sol_ss["pressure"] * 1e-12, '-', color=colors[i])
        axes[1].plot(m_sim_shock * 1e3, p_fit_MBar, '--', color=colors[i])

        # Profile 3: Velocity
        axes[2].plot(m_sim_shock * 1e3, numerical_data["u"][idx][shock_mask] * 1e-5, 'o', color=colors[i], markersize=3, alpha=0.5)
        axes[2].plot(mass_solver * 1e3, sol_ss["velocity"] * 1e-5, '-', color=colors[i])
        axes[2].plot(m_sim_shock * 1e3, u_fit_kms, '--', color=colors[i])

        # Profile 4: Temperature
        T_num_HeV = ((numerical_data["p"][idx] * numerical_data["rho"][idx]**(mu-1.))/(r * float(case_rh.f_Kelvin)))**(1./float(beta)) / KELVIN_PER_HEV
        axes[3].plot(m_sim_shock * 1e3, T_num_HeV[shock_mask], 'o', color=colors[i], markersize=3, alpha=0.5)
        
        # Reconstruct solver temperature via EOS: T = (p * v^(1-mu) / (r*f))^(1/beta)
        T_ss_HeV = ((sol_ss["pressure"] * (1./sol_ss["density"])**(1.-float(mu))) / (r * float(case_rh.f_Kelvin)))**(1./float(beta)) / KELVIN_PER_HEV
        axes[3].plot(mass_solver * 1e3, T_ss_HeV, '-', color=colors[i])
        axes[3].plot(m_sim_shock * 1e3, T_fit_HeV, '--', color=colors[i])

    # Styling
    labels = ["Density [g/cm$^3$]", "Pressure [MBar]", "Velocity [km/s]", "Temperature [HeV]"]
    for j, ax in enumerate(axes):
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels[j], fontsize=12)
        ax.set_xlabel("Lagrangian Mass Coordinate $m$ [mg/cm$^2$]", fontsize=12)
        if j == 0:
            ax.legend(loc="upper left")
        if j == 3:
            # Temperature plot in shock region has a singularity near y_s=0, clip y limit
            ax.set_ylim(0, 1.2 * np.max(T_num_HeV[shock_mask]))
            
    plt.suptitle("Shock Region Verification\nSolid: Piston Shock Solver, Dashed: Piecewise Analytical Fit, Circles: Hydro-Only Data", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "shock_region_comparison.png", dpi=200)
    print(f"Saved: shock_region_comparison.png")
    plt.close(fig)


# =============================================================================
# PART 5: SELF-SIMILAR PROFILE RELATIVE ERROR GRAPHS
# =============================================================================

def plot_relative_errors(hs: SubsonicHeatWave, output_dir: Path):
    """Plots the relative error of self-similar profile fits as a function of y."""
    # Obtain numerical profiles on a fine grid
    y_grid = np.linspace(1e-6, 1.0 - 1e-6, 500)
    xsi_grid = y_grid * hs.xsi_f
    
    profiles = hs.get_self_similar_profiles(xsi_vec=xsi_grid)
    V_num, P_num, U_num, T_num, S_num = profiles["V"], profiles["P"], profiles["U"], profiles["T"], profiles["S"]
    
    # Piecewise fits from report
    # Temperature fit: T_fit(y) = 1.01109 * (1 - y)^0.24158
    T_fit = 1.01109 * (1.0 - y_grid)**0.24158
    # Pressure fit: P_fit(y) = 0.35486 * y^0.87677 + 0.02905 * y^20.94836
    P_fit = 0.35486 * y_grid**0.87677 + 0.02905 * y_grid**20.94836
    # Velocity fit (Fit 4): U_fit(y) = -3.33700 * (1 - y^0.51600) * y^-0.19000
    U_fit = -3.33700 * (1.0 - y_grid**0.51600) * y_grid**-0.19000
    # Flux fit: S_fit(y) = 1.60300 * (1 - y)^0.54500
    S_fit = 1.60300 * (1.0 - y_grid)**0.54500
    
    # Calculate relative errors
    # Temperature needs to be evaluated up to y=0.99 as T(y->1)->0
    valid_T = y_grid <= 0.99
    err_T = np.abs((T_fit[valid_T] - T_num[valid_T]) / T_num[valid_T])
    
    err_P = np.abs((P_fit - P_num) / (P_num + 1e-15))
    err_U = np.abs((U_fit - U_num) / (U_num + 1e-15))
    err_S = np.abs((S_fit - S_num) / (S_num + 1e-15))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(y_grid[valid_T], err_T * 100, label="Temperature $\\tilde{T}(y)$ (up to $y=0.99$)", lw=2)
    ax.plot(y_grid, err_P * 100, label="Pressure $\\tilde{P}(y)$", lw=2)
    ax.plot(y_grid, err_U * 100, label="Velocity $\\tilde{U}(y)$", lw=2)
    ax.plot(y_grid, err_S * 100, label="Radiation Flux $\\tilde{S}(y)$", lw=2)
    
    ax.set_xlabel("Normalized coordinate $y = \\xi / \\xi_f$", fontsize=13)
    ax.set_ylabel("Relative Error [\\%]", fontsize=13)
    ax.set_title("Relative Error of Self-Similar Profile Fits ($\\tau = 0$)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    
    fig.tight_layout()
    fig.savefig(output_dir / "self_similar_relative_errors.png", dpi=200)
    print(f"Saved: self_similar_relative_errors.png")
    plt.close(fig)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def build_menahem_kwargs(tau: float) -> dict:
    """Kwargs matching exactly the RadHydroCase benchmark parameters."""
    time_rise = Units.nsec
    Tb = KELVIN_PER_HEV * (1.0 / time_rise) ** tau
    alpha = 1.5
    beta = 1.6
    f = 3.4e13 / (KELVIN_PER_HEV ** beta)
    g = 1.0 / (7200.0 * KELVIN_PER_HEV ** alpha)
    return dict(
        Tb=Tb, tau=tau, g=g, alpha=alpha, lambdap=0.2,
        f_heat=f, beta_heat=beta, mu_heat=0.14, gamma_heat=1.25,
        rho0=19.32, omega=0.0,
        f_shock=f, beta_shock=beta, mu_shock=0.14, gamma_shock=1.25,
    )

def main():
    # 1. Print Setup Exponents
    print_setup_segment()
    
    # Build solvers
    kwargs_tau0 = build_menahem_kwargs(0.0)
    solver_tau0 = AblationSolver(**kwargs_tau0)
    
    hs = solver_tau0.heat_solver
    ss = solver_tau0.shock_solver
    
    # 2. Print Level Expressions for both regimes
    print_solution_segment(hs, ss)
    
    # 3. Print Eulerian positions and Energy formulas
    print_eulerian_and_energy_formulas(hs, ss)
    
    # Define outputs directory
    results_dir = Path(__file__).resolve().parent / "position_verification_results"
    results_dir.mkdir(exist_ok=True)
    
    # 4. Load full rad-hydro numerical simulation data (base case)
    print("Loading full rad-hydro simulation data...")
    full_rad_hydro_data = load_full_rad_hydro_numerical_data()
    
    # Run the subsonic ablation comparison
    plot_subsonic_ablation_comparison(hs, full_rad_hydro_data, results_dir)
    
    # 5. Run the hydro-only numerical simulation on the fly
    hydro_only_data = run_hydro_only_numerical_data()
    
    # Run the shock region comparison
    plot_shock_region_comparison(ss, hydro_only_data, hs, results_dir)
    
    # 6. Plot the relative errors of self-similar profile fits
    plot_relative_errors(hs, results_dir)
    
    print("\n*** ALL TASKS SUCCESSFULLY COMPLETED! ***")

if __name__ == "__main__":
    # Ensure get_preset can resolve properly
    from project3_code.rad_hydro_sim.problems.presets_config import PRESET_TEST_CASES
    global case_rh
    case_rh = PRESET_TEST_CASES["constant_pressure_drive"]
    main()

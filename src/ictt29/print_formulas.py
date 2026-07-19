# ictt29/print_formulas.py
"""
Formula printer for ablation solver expressions.

Prints setup parameters, three-level solution expressions for subsonic
and shock regimes, Eulerian position formulas, integrated energies, and albedo.
No plots, no simulations.
"""
from __future__ import annotations

import sys
from pathlib import Path
from fractions import Fraction

# Setup paths so all project modules are resolvable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT.parent))  # Enables "project3_code" package resolution
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "menahem_new"))

from subsonic_heat_wave_og import SubsonicHeatWave, Units
from piston_shock_og import PistonShock
from ablation_solver_og import AblationSolver

# =============================================================================
# PHYSICAL CONSTANTS & SETUP
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

# =============================================================================
# PART 1: PHYSICAL CONSTANTS & SETUP SEGMENT
# =============================================================================

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
    
    # Subsonic Ablation Regime Expressions
    print("\n" + "#" * 40)
    print("  SUBSONIC ABLATION REGIME (tau = 0)")
    print("#" * 40)
    
    # Density Profile
    print("\n[Density Profile rho(m,t)]")
    print("Level 1 (General): rho(m,t) = V(xsi)^-1 * A^-a1 * B^-b1 * t^-c1")
    print(f"Level 2 (Parametrized): rho(m,t) = V(xsi)^-1 * {A_val:.5e}^(115/96) * {B_val:.5e}^(25/48) * t^(-25/48)")
    pre_rho = (A_val ** (Fraction(115, 96))) * (B_val ** (Fraction(25, 48)))
    print(f"Level 2 (Coeff): rho(m,t) = V(xsi)^-1 * {pre_rho:.5e} * t^(-0.52083)")
    print("Level 3 (Fitted - CGS): rho(m,t) = Derived via EOS from T_fit and P_fit profiles")
    
    # Pressure Profile
    print("\n[Pressure Profile p(m,t)]")
    print("Level 1 (General): p(m,t) = P(xsi) * A^a3 * B^b3 * t^c3")
    print(f"Level 2 (Parametrized): p(m,t) = P(xsi) * {A_val:.5e}^(505/192) * {B_val:.5e}^(43/96) * t^(-43/96)")
    pre_p = (A_val ** (Fraction(505, 192))) * (B_val ** (Fraction(43, 96)))
    pre_p_MBar = pre_p * 1e-12
    print(f"Level 2 (Coeff): p(m,t) = P(xsi) * {pre_p_MBar:.5e} * (t_ns)^(-0.44792) MBar")
    print("Level 3 (Fitted): p(m,t) = 7.05133 * (t_ns)^-0.44792 * [0.34866 * y^0.87714 + 0.02903 * y^21.08862] MBar")

    # Velocity Profile
    print("\n[Velocity Profile u(m,t)]")
    print("Level 1 (General): u(m,t) = U(xsi) * A^a2 * B^b2 * t^c2")
    print(f"Level 2 (Parametrized): u(m,t) = U(xsi) * {A_val:.5e}^(275/384) * {B_val:.5e}^(-7/192) * t^(7/192)")
    pre_u = (A_val ** (Fraction(275, 384))) * (B_val ** (-Fraction(7, 192)))
    pre_u_kms = pre_u * 1e-5
    print(f"Level 2 (Coeff): u(m,t) = U(xsi) * {pre_u_kms:.5e} * (t_ns)^0.03646 km/s")
    print("Level 3 (Fitted): u(m,t) = -191.29403 * (t_ns)^0.03646 * (1 - y) / (1 + 4.78201 * y) km/s")

    # Temperature Profile
    print("\n[Temperature Profile T(m,t)]")
    print("Level 1 (General): T(m,t) = (p(m,t) * v(m,t)^(1-mu) / (r * f))^(1/beta)")
    print("Level 2 (Parametrized): T(m,t) = T(xsi) * T_0 * t^tau = T(xsi) * 1.0 HeV")
    print("Level 3 (Fitted): T(m,t) = ((1 - y) * (1 + 0.20224 * y))^(10/39) HeV")
    
    # Shock Region Expressions
    print("\n" + "#" * 40)
    print("  SHOCK REGIME (tau = -43/96)")
    print("#" * 40)
    
    p0s_MBar = ss.p0 * 1e-12
    v0_val = ss.v0
    print(f"  p0s = {p0s_MBar:.5f} MBar  v0 = {v0_val:.5f} cm^3/g")
    
    print("\n[Density Profile rho(m,t)]")
    print("Level 1 (General): rho(m,t) = rho_0 * V_s(xsi_s)^-1")
    print(f"Level 2 (Parametrized): rho(m,t) = 19.32 * V_s(xsi_s)^-1 g/cm^3")
    print("Level 3 (Fitted): rho(m,t) = 173.88000 * y_s^0.64747 g/cm^3")
    
    print("\n[Pressure Profile p(m,t)]")
    print("Level 1 (General): p(m,t) = P_s(xsi_s) * p0_s * t^tau_s")
    print(f"Level 2 (Parametrized): p(m,t) = P_s(xsi_s) * {p0s_MBar:.5f} * (t_ns)^(-0.44792) MBar")
    print("Level 3 (Fitted): p(m,t) = 2.71000 * (t_ns)^-0.44792 * [1.0 + 0.49338 * y_s^1.13303] MBar")

    print("\n[Velocity Profile u(m,t)]")
    print("Level 1 (General): u(m,t) = U_s(xsi_s) * sqrt(v0 * p0_s) * t^(tau_s/2)")
    coeff_u = (v0_val * ss.p0) ** 0.5 * 1e-5
    print(f"Level 2 (Parametrized): u(m,t) = U_s(xsi_s) * {coeff_u:.5f} * (t_ns)^(-0.22396) km/s")
    print("Level 3 (Fitted): u(m,t) = (t_ns)^-0.22396 * [3.65776 + 0.65734 * y_s^0.64494] km/s")

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
    
    # 1. Eulerian Position
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
    
    # 2. Integrated Energies
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


    

    # 3. Albedo
    print("\n[3. Albedo alpha_albedo(t)]")
    print("  Level 1: alpha_albedo(t) = 2 * F_s(t) / (a_rad * clight * T_s(t)^4) = B_bath * t^eta_bath")
    print("  Level 2: B_bath = S0 * A^aS * B^bS / (2 * sigma_sb * T0^4), eta_bath = cS - 4*tau")
    
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
# MAIN
# =============================================================================

def build_menahem_kwargs(tau: float) -> dict:
    """Kwargs matching exactly the RadHydroCase benchmark parameters."""
    time_rise = Units.nsec
    Tb = KELVIN_PER_HEV * (1.0 / time_rise) ** tau
    alpha_f = 1.5
    beta_f = 1.6
    f = 3.4e13 / (KELVIN_PER_HEV ** beta_f)
    g = 1.0 / (7200.0 * KELVIN_PER_HEV ** alpha_f)
    return dict(
        Tb=Tb, tau=tau, g=g, alpha=alpha_f, lambdap=0.2,
        f_heat=f, beta_heat=beta_f, mu_heat=0.14, gamma_heat=1.25,
        rho0=19.32, omega=0.0,
        f_shock=f, beta_shock=beta_f, mu_shock=0.14, gamma_shock=1.25,
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
    
    print("\n*** ALL FORMULA PRINTING COMPLETED! ***")

if __name__ == "__main__":
    main()

# ictt29/print_fitted_formulas.py
"""
Dynamic Formula Generator and LaTeX Compiler.
Solves subsonic heat wave and piston shock similarity ODEs,
optimizes curve fits dynamically, substitutes physical constants (A, B, v0, p0)
at t = 1 ns in CGS/simplified units, reads piecewise_analytic_formulas_template.tex,
writes piecewise_analytic_formulas.tex, and compiles it to piecewise_analytic_formulas.pdf.
"""
from __future__ import annotations

import sys
import os
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import scipy.integrate
import subprocess

# Dynamic monkeypatch of numpy.trapezoid to numpy.trapz for compatibility with NumPy 1.x
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# Ensure proper project imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_PARENT))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "menahem_new"))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
    PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE
)
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _heat_kwargs_from_case,
    _shock_kwargs_from_case
)
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

from subsonic_heat_wave_og import SubsonicHeatWave
from piston_shock_og import PistonShock

def main():
    print("====================================================")
    # 1. SOLVE similarity ODEs
    print("1) Solving subsonic similarity ODEs (finding xsi_f)...")
    case_heat, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    heat_kwargs = _heat_kwargs_from_case(case_heat)
    solver_heat = SubsonicHeatWave(**heat_kwargs).find_xsi_f()

    print("2) Solving shock similarity ODEs (finding xsi_s)...")
    case_shock, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    shock_kwargs = _shock_kwargs_from_case(case_shock)
    solver_shock = PistonShock(
        rho0=float(case_shock.rho0),
        omega=0.0,
        p0=shock_kwargs["p0"],
        tau=shock_kwargs["tau"],
        gamma=1.25
    )

    # 2. RUN GRIDS & EXTRACT PROFILES
    # Subsonic Ablation Regime
    y_grid_h = np.linspace(0.0, 1.0 - 1e-10, 2000)
    xsi_vec_h = y_grid_h * solver_heat.xsi_f
    profiles_heat = solver_heat.get_self_similar_profiles(xsi_vec=xsi_vec_h)
    imin = profiles_heat.get("imin", 0)

    y_valid_h = y_grid_h[imin:]
    V_valid_h = profiles_heat["V"][imin:]
    U_valid_h = profiles_heat["U"][imin:]
    P_valid_h = profiles_heat["P"][imin:]
    T_valid_h = profiles_heat["T"][imin:]
    rho_valid_h = 1.0 / V_valid_h

    # Shock compressed Regime
    y_grid_s = np.linspace(0.0, 1.0, 2000)
    xsi_vec_s = y_grid_s * solver_shock.xsi_s
    xsi_vec_s[0] = 1e-10
    V_valid_s, U_valid_s, P_valid_s = solver_shock.get_self_similar_profiles(xsi_vec=xsi_vec_s)
    rho_valid_s = 1.0 / V_valid_s
    T_valid_s = P_valid_s * V_valid_s / solver_shock.r

    mask_s = (y_grid_s > 0.005) & np.isfinite(V_valid_s) & np.isfinite(U_valid_s) & np.isfinite(P_valid_s)
    y_valid_s = y_grid_s[mask_s]
    U_valid_s = U_valid_s[mask_s]
    P_valid_s = P_valid_s[mask_s]
    rho_valid_s = rho_valid_s[mask_s]
    T_valid_s = T_valid_s[mask_s]

    # 3. DYNAMICAL CURVE FITTING
    print("3) Running curve fits on self-similar profiles...")
    
    # Subsonic Ablation Fits
    # Temperature fit: Smith approximation
    def smith_approximation(y, R):
        return ((1.0 - y) * (1.0 + R * y))**(10.0 / 39.0)
    popt_T_h, _ = curve_fit(smith_approximation, y_valid_h, T_valid_h, p0=[0.5])
    R_val = popt_T_h[0]

    # Pressure fit: Power law
    def power_law_origin(y, a, b, c, d):
        return a * y**c + b * y**(c+d)
    popt_P_h, _ = curve_fit(power_law_origin, y_valid_h, P_valid_h, p0=[0.355, 0.5, 0.04, 2.3])
    a_P_h, b_P_h, c_P_h, d_P_h = popt_P_h

    # Velocity fit: Rational fit
    def fit_u_rational(y, U0, b):
        return U0 * (1.0 - y) / (1.0 + b * y)
    popt_u_h, _ = curve_fit(fit_u_rational, y_valid_h, U_valid_h, p0=[U_valid_h[0], 0.5])
    U0_h, b_u_h = popt_u_h

    # Specific volume fit: Power law
    def fit_v_sub(y, V0, d):
        return V0 * (1.0 - y)**d
    popt_v_h, _ = curve_fit(fit_v_sub, y_valid_h, V_valid_h, p0=[V_valid_h[0], 0.5])
    V0_h, d_v_h = popt_v_h

    # Shock Region Fits
    # Pressure fit
    P_s = P_valid_s[-1]
    def power_law_P(y, d):
        return 1.0 - (1.0 - P_s) * y**d
    popt_P_s, _ = curve_fit(power_law_P, y_valid_s, P_valid_s, p0=[1.0])
    d_P_s = popt_P_s[0]

    # Velocity fit
    U_0_s = U_valid_s[0]
    U_s_s = U_valid_s[-1]
    def power_law_U(y, d):
        return U_0_s - (U_0_s - U_s_s) * y**d
    popt_u_s, _ = curve_fit(power_law_U, y_valid_s, U_valid_s, p0=[1.0])
    d_u_s = popt_u_s[0]

    # Density fit
    rho_s_s = rho_valid_s[-1]
    def power_law_R(y, d):
        return rho_s_s * y**d
    popt_r_s, _ = curve_fit(power_law_R, y_valid_s, rho_valid_s, p0=[0.5])
    d_rho_s = popt_r_s[0]

    # Temperature fit (for the power laws of shock profiles based on ideal gas relation)
    T_s_s = T_valid_s[-1]
    def power_law_T(y, d):
        return T_s_s * y**d
    popt_t_s, _ = curve_fit(power_law_T, y_valid_s, T_valid_s, p0=[-0.5])
    d_T_s = popt_t_s[0]

    # 4. PHYSICAL CONSTANTS AND CGS/SIMPLIFIED SCALE CONVERSIONS (t = 1 ns)
    A_h = solver_heat.A
    B_h = solver_heat.B
    t_ns_h = 1e-9

    # Exponents fractions
    a_v_frac, b_v_frac, c_v_frac = r"-\frac{115}{96}", r"-\frac{25}{48}", r"\frac{25}{48}"
    a_u_frac, b_u_frac, c_u_frac = r"\frac{275}{384}", r"-\frac{7}{192}", r"\frac{7}{192}"
    a_p_frac, b_p_frac, c_p_frac = r"\frac{505}{192}", r"\frac{43}{96}", r"-\frac{43}{96}"
    
    # Exponents decimals
    a_v, b_v, c_v = -115/96, -25/48, 25/48
    a_u, b_u, c_u = 275/384, -7/192, 7/192
    a_p, b_p, c_p = 505/192, 43/96, -43/96
    # Note: EXACT subsonic front scaling derived algebraically:
    a_mf, b_mf, c_mf = -245/128, -31/64, -33/64

    # Subsonic Heat Front mass coordinate scaling
    m_f_scale = solver_heat.xsi_f * A_h**(-a_mf) * B_h**(-b_mf) * t_ns_h**(-c_mf)
    inv_mf = 1.0 / m_f_scale
    c_mf_fraction = r"-\frac{33}{64}"

    # Specific volume scale and lead coeff
    v_scale_h = A_h**a_v * B_h**b_v * t_ns_h**c_v
    lead_coeff_v_h = V0_h * v_scale_h

    # Velocity scale in km/s (10^-5 factor) and lead coeff
    u_scale_h_kms = A_h**a_u * B_h**b_u * t_ns_h**c_u * 1e-5
    lead_coeff_u_h = U0_h * u_scale_h_kms
    denom_coeff_u_h = b_u_h * inv_mf

    # Pressure scale in MBar (10^-12 factor) and lead coeff
    p_scale_h_mbar = A_h**a_p * B_h**b_p * t_ns_h**c_p * 1e-12
    lead_coeff_p_a_h = a_P_h * p_scale_h_mbar
    lead_coeff_p_b_h = b_P_h * p_scale_h_mbar

    # Shock Region Conversions (t = 1 ns)
    v0_s = solver_shock.v0
    p0_s = solver_shock.p0
    tau_s = solver_shock.tau
    xsi_s_val = solver_shock.xsi_s

    # Shock mass coordinate scaling
    m_s_scale = solver_shock.shocked_mass(time=1e-9)
    inv_ms = 1.0 / m_s_scale
    c_ms_fraction = r"\frac{149}{192}"

    # Shock velocity scale in km/s (10^-5 factor) and lead coeffs
    u_scale_s_kms = np.sqrt(v0_s * p0_s) * (1e-9)**(tau_s / 2.0) * 1e-5
    lead_coeff_u_0_s = U_0_s * u_scale_s_kms
    lead_coeff_u_diff_s = (U_s_s - U_0_s) * u_scale_s_kms
    
    # Velocity Level 3 Multiplied Coefficients
    lead_u_mult_s = lead_coeff_u_diff_s * inv_ms**d_u_s
    power_u_t_s = -149/192 * d_u_s

    # Shock pressure scale in MBar (10^-12 factor) and lead coeffs
    p_scale_s_mbar = p0_s * (1e-9)**tau_s * 1e-12
    lead_coeff_p_0_s = p_scale_s_mbar
    lead_p_diff = - (1.0 - P_s)
    lead_coeff_p_diff_s = p_scale_s_mbar * lead_p_diff
    
    # Pressure Level 3 Multiplied Coefficients
    lead_p_mult_s = lead_coeff_p_diff_s * inv_ms**d_P_s
    power_p_t_s = -149/192 * d_P_s

    # Density Conversions
    rho_scale_s = 1.0 / v0_s
    lead_coeff_rho_s = rho_scale_s * rho_s_s
    
    # Density Level 3 Multiplied Coefficients
    lead_rho_mult_s = lead_coeff_rho_s * inv_ms**d_rho_s
    power_rho_t_s = -149/192 * d_rho_s

    # Shock Temperature Algebraic EOS Derivation
    p_scale_barye = p_scale_s_mbar * 1e12
    v_scale_eos = 1.0 / (solver_shock.rho0 * rho_s_s)
    
    # T_scale_s CGS prefactor from EOS in Kelvin
    T_coeff_cgs = ( (p_scale_barye * (v_scale_eos ** 1.14)) / (solver_shock.r * solver_heat.f) ) ** 0.625
    T_coeff_hev = T_coeff_cgs / KELVIN_PER_HEV
    
    # Temperature Level 3 Multiplied Coefficients
    lead_T_mult_s = T_coeff_hev * inv_ms**(-0.7125 * d_rho_s)
    lead_T_diff_mult_s = lead_p_diff * inv_ms**d_P_s
    power_T_t_s = 1.14 * d_rho_s * 149/192
    power_T_t_s2 = -(d_P_s - 1.14 * d_rho_s) * 149/192

    # 5. DYNAMIC LATEX GENERATION
    doc_dir = Path("c:/Users/TLP-001/Documents/GitHub/project3_docs/fitting")
    template_path = doc_dir / "piecewise_analytic_formulas_template.tex"
    
    if not template_path.exists():
        print(f"Error: Template LaTeX file not found at {template_path}")
        sys.exit(1)
        
    print(f"4) Reading LaTeX template from: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        latex_template = f.read()

    # 6. REPLACE ALL PLACEHOLDERS WITH DYNAMIC fit / computed values
    print("5) Injecting fit coefficients into the LaTeX template...")
    latex_content = latex_template
    latex_content = latex_content.replace("__A_VAL__", f"{A_h:.5f}")
    latex_content = latex_content.replace("__B_VAL__", f"{B_h:.5e}")
    
    # Fractions
    latex_content = latex_content.replace("__A_V_FRAC__", a_v_frac)
    latex_content = latex_content.replace("__B_V_FRAC__", b_v_frac)
    latex_content = latex_content.replace("__C_V_FRAC__", c_v_frac)
    latex_content = latex_content.replace("__A_U_FRAC__", a_u_frac)
    latex_content = latex_content.replace("__B_U_FRAC__", b_u_frac)
    latex_content = latex_content.replace("__C_U_FRAC__", c_u_frac)
    latex_content = latex_content.replace("__A_P_FRAC__", a_p_frac)
    latex_content = latex_content.replace("__B_P_FRAC__", b_p_frac)
    latex_content = latex_content.replace("__C_P_FRAC__", c_p_frac)
    
    # Subsonic Ablation values
    latex_content = latex_content.replace("__V0_H__", f"{V0_h:.5f}")
    latex_content = latex_content.replace("__D_V_H__", f"{d_v_h:.5f}")
    latex_content = latex_content.replace("__LEAD_V_H__", f"{lead_coeff_v_h:.5f}")
    latex_content = latex_content.replace("__V_SCALE_H__", f"{v_scale_h:.5f}")
    latex_content = latex_content.replace("__INV_MF__", f"{inv_mf:.5f}")
    latex_content = latex_content.replace("__C_MF_FRAC__", c_mf_fraction)
    latex_content = latex_content.replace("__U0_H__", f"{U0_h:.5f}")
    latex_content = latex_content.replace("__B_U_H__", f"{b_u_h:.5f}")
    
    # Support both __LEAD_U_H__ and __LEAD_COEFF_U_H__ placeholders
    latex_content = latex_content.replace("__LEAD_U_H__", f"{lead_coeff_u_h:.5f}")
    latex_content = latex_content.replace("__LEAD_COEFF_U_H__", f"{lead_coeff_u_h:.5f}")
    latex_content = latex_content.replace("__U_SCALE_H_KMS__", f"{u_scale_h_kms:.5f}")
    
    latex_content = latex_content.replace("__DENOM_COEFF_U_H__", f"{denom_coeff_u_h:.5f}")
    latex_content = latex_content.replace("__P_SCALE_H__", f"{p_scale_h_mbar:.5f}")
    latex_content = latex_content.replace("__A_P_H__", f"{a_P_h:.5f}")
    latex_content = latex_content.replace("__B_P_H__", f"{b_P_h:.5f}")
    latex_content = latex_content.replace("__C_P_H__", f"{c_P_h:.5f}")
    latex_content = latex_content.replace("__C_P_H_D__", f"{c_P_h + d_P_h:.5f}")
    latex_content = latex_content.replace("__LEAD_COEFF_P_A_H__", f"{lead_coeff_p_a_h:.5f}")
    latex_content = latex_content.replace("__LEAD_COEFF_P_B_H__", f"{lead_coeff_p_b_h:.5f}")
    latex_content = latex_content.replace("__XSI_F_H__", f"{solver_heat.xsi_f:.5f}")
    latex_content = latex_content.replace("__M_F_SCALE__", f"{m_f_scale:.5f}")
    latex_content = latex_content.replace("__R_VAL__", f"{R_val:.5f}")
    latex_content = latex_content.replace("__DENOM_T__", f"{R_val * inv_mf:.5f}")
    
    # Shock Values
    latex_content = latex_content.replace("__XSI_S_S__", f"{xsi_s_val:.5f}")
    latex_content = latex_content.replace("__M_S_SCALE__", f"{m_s_scale:.5f}")
    latex_content = latex_content.replace("__INV_MS__", f"{inv_ms:.5f}")
    latex_content = latex_content.replace("__RHO_S_S__", f"{rho_s_s:.5f}")
    latex_content = latex_content.replace("__D_RHO_S__", f"{d_rho_s:.5f}")
    latex_content = latex_content.replace("__RHO_SCALE_S__", f"{rho_scale_s:.5f}")
    latex_content = latex_content.replace("__LEAD_RHO_S__", f"{lead_coeff_rho_s:.5f}")
    latex_content = latex_content.replace("__LEAD_RHO_MULT_S__", f"{lead_rho_mult_s:.5f}")
    latex_content = latex_content.replace("__POWER_RHO_T_S__", f"{power_rho_t_s:.5f}")
    
    latex_content = latex_content.replace("__U_0_S__", f"{U_0_s:.5f}")
    latex_content = latex_content.replace("__U_DIFF_COEFF_S__", f"{U_s_s - U_0_s:.5f}")
    latex_content = latex_content.replace("__U_SCALE_S_KMS__", f"{u_scale_s_kms:.5f}")
    latex_content = latex_content.replace("__LEAD_COEFF_U_0_S__", f"{lead_coeff_u_0_s:.5f}")
    latex_content = latex_content.replace("__LEAD_COEFF_U_DIFF_S__", f"{lead_coeff_u_diff_s:.5f}")
    latex_content = latex_content.replace("__D_U_S__", f"{d_u_s:.5f}")
    latex_content = latex_content.replace("__LEAD_U_MULT_S__", f"{lead_u_mult_s:.5f}")
    latex_content = latex_content.replace("__POWER_U_T_S__", f"{power_u_t_s:.5f}")
    
    latex_content = latex_content.replace("__P_DIFF_COEFF_S__", f"{lead_p_diff:.5f}")
    latex_content = latex_content.replace("__LEAD_COEFF_P_0_S__", f"{lead_coeff_p_0_s:.5f}")
    latex_content = latex_content.replace("__LEAD_COEFF_P_DIFF_S__", f"{lead_coeff_p_diff_s:.5f}")
    latex_content = latex_content.replace("__D_P_S__", f"{d_P_s:.5f}")
    latex_content = latex_content.replace("__LEAD_P_MULT_S__", f"{lead_p_mult_s:.5f}")
    latex_content = latex_content.replace("__POWER_P_T_S__", f"{power_p_t_s:.5f}")
    
    latex_content = latex_content.replace("__T_COEFF_HEV__", f"{T_coeff_hev:.5f}")
    latex_content = latex_content.replace("__T_COEFF_KELVIN__", f"{T_coeff_cgs:.5f}")
    latex_content = latex_content.replace("__POWER_T_Y1__", f"{-0.7125 * d_rho_s:.5f}")
    latex_content = latex_content.replace("__POWER_T_Y2__", f"{d_P_s:.5f}")
    latex_content = latex_content.replace("__DIFF_P_S__", f"{lead_p_diff:.5f}")
    latex_content = latex_content.replace("__LEAD_T_MULT_S__", f"{lead_T_mult_s:.5f}")
    latex_content = latex_content.replace("__POWER_T_M_S__", f"{-1.14 * d_rho_s:.5f}")
    latex_content = latex_content.replace("__POWER_T_T_S__", f"{power_T_t_s:.5f}")
    latex_content = latex_content.replace("__LEAD_T_DIFF_MULT_S__", f"{lead_T_diff_mult_s:.5f}")
    latex_content = latex_content.replace("__POWER_T_M_S2__", f"{d_P_s - 1.14 * d_rho_s:.5f}")
    latex_content = latex_content.replace("__POWER_T_T_S2__", f"{power_T_t_s2:.5f}")

    # Target directory and file paths
    doc_dir.mkdir(parents=True, exist_ok=True)
    tex_path = doc_dir / "piecewise_analytic_formulas.tex"
    
    print(f"6) Overwriting LaTeX source: {tex_path}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
        
    print("7) Compiling LaTeX document using pdflatex...")
    try:
        # Run pdflatex in non-interactive mode
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=str(doc_dir),
            capture_output=True,
            text=True,
            check=True
        )
        print("LaTeX compilation SUCCESSFUL!")
    except subprocess.CalledProcessError as e:
        print("LaTeX compilation FAILED!")
        print("Stdout:\n", e.stdout)
        sys.exit(1)

    # 7. Print compiled equations beautifully to stdout
    print("\n" + "=" * 80)
    print("  DYNAMICALLY GENERATED FORMULAS (substituted A, B at t = 1 ns)")
    print("=" * 80)
    
    print("\n--- Subsonic Fluid Velocity u(m,t) ---")
    print(f"u(m,t) = U(\\xi) A^{a_u_frac} B^{b_u_frac} t^{c_u_frac}")
    print(f"       \\approx {U0_h:.5f} * ((1 - y)/(1 + {b_u_h:.5f} * y)) * A^{a_u_frac} B^{b_u_frac} t^{c_u_frac}")
    print(f"       \\approx {lead_coeff_u_h:.5f} * ((1 - {inv_mf:.5f}*m*(t/ns)^-33/64) / (1 + {denom_coeff_u_h:.5f}*m*(t/ns)^-33/64)) * (t/ns)^{c_u_frac} km/s")
    
    print("\n--- Subsonic Specific Volume v(m,t) ---")
    print(f"v(m,t) = V(\\xi) A^{a_v_frac} B^{b_v_frac} t^{c_v_frac}")
    print(f"       \\approx {V0_h:.5f} * (1 - y)^{d_v_h:.5f} * A^{a_v_frac} B^{b_v_frac} t^{c_v_frac}")
    print(f"       \\approx {lead_coeff_v_h:.5f} * (1 - {inv_mf:.5f}*m*(t/ns)^-33/64)^{d_v_h:.5f} * (t/ns)^{c_v_frac} cm^3/g")
    
    print("\n====================================================")
    print("*** ALL TASKS COMPLETED SUCCESSFULLY! ***")
    print("====================================================")

if __name__ == "__main__":
    main()

# ictt29/print_fitted_formulas.py
"""
Dynamic Formula Generator and LaTeX Compiler.
Solves subsonic heat wave and piston shock similarity ODEs,
extracts the optimal curve fits dynamically from sub_fitting.py and shock_fitting.py,
substitutes physical constants at t = 1 ns in CGS/simplified units,
reads piecewise_analytic_formulas_template.tex, writes piecewise_analytic_formulas.tex,
and compiles it to piecewise_analytic_formulas.pdf.
"""
from __future__ import annotations

import sys
import os
import numpy as np
from pathlib import Path
import subprocess

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

import sub_fitting
import shock_fitting

def main():
    print("====================================================")
    print("1) Solving similarity ODEs and extracting optimal fits...")
    
    # 1. SOLVE similarity ODEs & RUN CURVE FITTING via sub_fitting / shock_fitting
    import data_loader
    case_heat, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    solver_heat = data_loader.get_sub_similarity_solver(case_heat, "heat_const_T")
    params_heat = sub_fitting.perform_subsonic_fitting(solver_heat)
    
    case_shock, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    solver_shock = data_loader.get_shock_similarity_solver(case_shock, "shock_const_T")
    params_shock = shock_fitting.perform_shock_fitting(solver_shock)
    
    # 2. SUBSONIC SCALING CONSTANTS AND conversions (t = 1 ns)
    A_h = solver_heat.A
    B_h = solver_heat.B
    t_ns_h = 1e-9
    
    # Exponents fractions
    a_v_frac, b_v_frac, c_v_frac = r"-\frac{115}{96}", r"-\frac{25}{48}", r"\frac{25}{48}"
    a_u_frac, b_u_frac, c_u_frac = r"\frac{275}{384}", r"-\frac{7}{192}", r"\frac{7}{192}"
    a_p_frac, b_p_frac, c_p_frac = r"\frac{505}{192}", r"\frac{43}{96}", r"-\frac{43}{96}"
    
    a_v, b_v, c_v = -115/96, -25/48, 25/48
    a_u, b_u, c_u = 275/384, -7/192, 7/192
    a_p, b_p, c_p = 505/192, 43/96, -43/96
    a_mf, b_mf, c_mf = -245/128, -31/64, -33/64
    
    # Subsonic Heat Front mass coordinate scaling
    m_f_scale = solver_heat.xsi_f * A_h**(-a_mf) * B_h**(-b_mf) * t_ns_h**(-c_mf)
    inv_mf = 1.0 / m_f_scale
    c_mf_fraction = r"-\frac{33}{64}"
    
    # Specific volume scale and lead coeff
    v_scale_h = A_h**a_v * B_h**b_v * t_ns_h**c_v
    
    # Simple Specific Volume fit parameters (as used in template)
    y_valid_h = params_heat["y_valid"]
    V_valid_h = solver_heat.get_self_similar_profiles(xsi_vec=y_valid_h*solver_heat.xsi_f)["V"]
    # Bypass cells with inf/nan
    valid_v_idx = np.isfinite(V_valid_h)
    y_valid_v = y_valid_h[valid_v_idx]
    V_valid_v = V_valid_h[valid_v_idx]
    
    # Fit V simple power law: V0 * (1-y)**d
    from scipy.optimize import curve_fit
    def fit_v_sub(y, V0, d):
        return V0 * (1.0 - y)**d
    popt_v_h, _ = curve_fit(fit_v_sub, y_valid_v, V_valid_v, p0=[V_valid_v[0], 0.5])
    V0_h, d_v_h = popt_v_h
    lead_coeff_v_h = V0_h * v_scale_h
    
    # Subsonic Density CGS conversions from EOS
    rho_scale_h = 1.0 / v_scale_h
    r_val = 0.25
    mu_val = 0.14
    T0 = 1.0
    beta = 1.6
    lead_rho_scale_h = rho_scale_h * (r_val * T0**beta)**(-1.0 / (1.0 - mu_val))
    power_rho_t_numer = r"\frac{16}{39}"
    power_rho_out = f"{-1.0 / (1.0 - mu_val):.5f}"
    
    # Velocity scale in km/s (10^-5 factor)
    u_scale_h_kms = A_h**a_u * B_h**b_u * t_ns_h**c_u * 1e-5
    
    # Pressure scale in MBar (10^-12 factor)
    p_scale_h_mbar = A_h**a_p * B_h**b_p * t_ns_h**c_p * 1e-12
    
    # Extract Subsonic Pressure Fit (Fit 4)
    best_p_sub = params_heat["best_p"]
    popt_P_sub = best_p_sub["popt"]  # [a, b, e, c, d, f]
    a_P_sub, b_P_sub, e_P_sub = popt_P_sub[0], popt_P_sub[1], popt_P_sub[2]
    c_P_sub = popt_P_sub[3]
    cpd_P_sub = popt_P_sub[3] + popt_P_sub[4]
    cpdf_P_sub = popt_P_sub[3] + popt_P_sub[4] + popt_P_sub[5]
    Pf_sub = params_heat["P_valid"][-1]
    
    # CGS lead coefficients for pressure
    lead_pf_sub = Pf_sub * p_scale_h_mbar
    lead_a_p_sub = a_P_sub * p_scale_h_mbar
    lead_b_p_sub = b_P_sub * p_scale_h_mbar
    lead_e_p_sub = e_P_sub * p_scale_h_mbar
    
    # Extract Subsonic Velocity Fit (Fit 12)
    best_u_sub = params_heat["best_u"]
    popt_U_sub = best_u_sub["popt"]  # [a1, a2, a3, a4, alpha, b1, b2, b3, b4, y0]
    ua1, ua2, ua3, ua4 = popt_U_sub[0], popt_U_sub[1], popt_U_sub[2], popt_U_sub[3]
    ualpha = popt_U_sub[4]
    ub1, ub2, ub3, ub4 = popt_U_sub[5], popt_U_sub[6], popt_U_sub[7], popt_U_sub[8]
    uy0 = popt_U_sub[9]
    U0_sub = params_heat["U_valid"][0]
    UF_sub = params_heat["U_valid"][-1]
    
    # CGS lead coefficients for velocity
    lead_u0_sub = U0_sub * u_scale_h_kms
    lead_uf_sub = UF_sub * u_scale_h_kms
    lead_ua1_sub = ua1 * u_scale_h_kms
    lead_ua2_sub = ua2 * u_scale_h_kms
    lead_ua3_sub = ua3 * u_scale_h_kms
    lead_ua4_sub = ua4 * u_scale_h_kms
    lead_ub1_sub = ub1 * u_scale_h_kms
    lead_ub2_sub = ub2 * u_scale_h_kms
    lead_ub3_sub = ub3 * u_scale_h_kms
    lead_ub4_sub = ub4 * u_scale_h_kms
    
    # Temperature Fit (Smith approximation)
    popt_T_h = params_heat["popt_T"]
    R_val = popt_T_h[0]
    denom_T = R_val * inv_mf
    
    # 3. SHOCK REGION CONSTANTS AND conversions (t = 1 ns)
    v0_s = solver_shock.v0
    p0_s = solver_shock.p0
    tau_s = solver_shock.tau
    xsi_s_val = solver_shock.xsi_s
    
    # Shock mass coordinate scaling
    m_s_scale = solver_shock.shocked_mass(time=1e-9)
    inv_ms = 1.0 / m_s_scale
    
    # Shock velocity scale in km/s
    u_scale_s_kms = np.sqrt(v0_s * p0_s) * (1e-9)**(tau_s / 2.0) * 1e-5
    
    # Shock pressure scale in MBar
    p_scale_s_mbar = p0_s * (1e-9)**tau_s * 1e-12
    
    # Extract Shock Pressure Fit (1-parameter power law: P = 1 - (1-Ps)*y^d)
    popt_P_shock = params_shock["popt_P"]
    dP_shock = popt_P_shock[0]
    Ps_shock = solver_shock.Ps
    lead_p_diff = - (1.0 - Ps_shock)
    lead_coeff_p_diff_s = p_scale_s_mbar * lead_p_diff
    
    # Extract Shock Velocity Fit (Fit 3: c - a * y^b)
    best_u_shock = params_shock["best_u"]
    popt_U_shock = best_u_shock["popt"]  # [c, a, b]
    cu_shock, au_shock, bu_shock = popt_U_shock[0], popt_U_shock[1], popt_U_shock[2]
    lead_cu_shock = cu_shock * u_scale_s_kms
    lead_au_shock = au_shock * u_scale_s_kms
    
    # Extract Shock Density Piecewise Fits
    rho_0_shock = params_shock["rho_0"]
    rho_s_shock = 1.0 / solver_shock.Rs_or_Vs
    popt_rho_low = params_shock["popt_rho_low"]  # [a, d] for low domain
    arho_low, drho_low = popt_rho_low[0], popt_rho_low[1]
    
    # Extract Shock Temperature Fit in high domain (Fit 2 Double Power Law: Ts + c1 * (1-y)^a + c2 * (1-y)^b)
    best_T_shock = params_shock["best_T"]
    popt_T_shock = best_T_shock["popt"]  # [c, a, b]
    c1_t, a_t, b_t = popt_T_shock[0], popt_T_shock[1], popt_T_shock[2]
    Ts_shock = params_shock["T_valid"][-1]
    T_0_shock = params_shock["T_valid"][0]
    c2_t = T_0_shock - Ts_shock - c1_t
    
    # Shock Temperature Scaling to Kelvin (EOS parameters)
    p_scale_barye = p_scale_s_mbar * 1e12
    v_scale_eos = 1.0 / solver_shock.rho0
    
    # T_scale_s CGS prefactor from EOS in Kelvin: T = (P * V**(1-mu) / (r * f))**(1/beta)
    # where r = 0.25, f = 6730.91 (f_Kelvin)
    # The scaling factor is therefore:
    T_scale_kelvin = ( (p_scale_barye * (v_scale_eos ** 0.86)) / (solver_shock.r * 6730.91) ) ** 0.625
    T_pow = (tau_s + 0.86 * (tau_s + 2.0) / 2.0) / 1.6
    
    # 3b. COMPUTE ERROR MARGINS FOR SECTION 6
    # Subsonic Errors
    T_valid_h = params_heat["T_valid"]
    P_valid_h = params_heat["P_valid"]
    U_valid_h = params_heat["U_valid"]
    y_valid_h = params_heat["y_valid"]
    
    T_fit_h, P_fit_h, U_fit_h, _ = sub_fitting.fit_by_params(y_valid_h, params_heat)
    _, _, _, rho_fit_h = sub_fitting.fit_by_params(params_heat["y_rho"], params_heat)
    
    err_T_h = np.abs((T_fit_h - T_valid_h) / T_valid_h)
    err_P_h = np.abs((P_fit_h - P_valid_h) / P_valid_h)
    err_U_h = np.abs((U_fit_h - U_valid_h) / (U_valid_h + 1e-15))
    err_rho_h = np.abs((rho_fit_h - params_heat["rho_valid"]) / params_heat["rho_valid"])
    
    avg_T_h, max_T_h = np.mean(err_T_h), np.max(err_T_h)
    avg_P_h, max_P_h = np.mean(err_P_h), np.max(err_P_h)
    avg_U_h, max_U_h = best_u_sub["avg_err"], best_u_sub["max_err"]
    avg_rho_h, max_rho_h = np.mean(err_rho_h), np.max(err_rho_h)
    
    # Shock Errors
    y_valid_s = params_shock["y_valid"]
    T_valid_s = params_shock["T_valid"]
    P_valid_s = params_shock["P_valid"]
    U_valid_s = params_shock["U_valid"]
    rho_valid_s = params_shock["rho_valid"]
    
    T_fit_s, P_fit_s, U_fit_s, rho_fit_s = shock_fitting.fit_by_params(y_valid_s, params_shock)
    
    err_T_s = np.abs((T_fit_s - T_valid_s) / T_valid_s)
    err_P_s = np.abs((P_fit_s - P_valid_s) / P_valid_s)
    err_U_s = np.abs((U_fit_s - U_valid_s) / (U_valid_s + 1e-15))
    err_rho_s = np.abs((rho_fit_s - rho_valid_s) / rho_valid_s)
    
    avg_T_s, max_T_s = np.mean(err_T_s), np.max(err_T_s)
    avg_P_s, max_P_s = np.mean(err_P_s), np.max(err_P_s)
    avg_U_s, max_U_s = best_u_shock["avg_err"], best_u_shock["max_err"]
    avg_rho_s, max_rho_s = np.mean(err_rho_s), np.max(err_rho_s)
    
    # 3c. EULERIAN COORDINATE TRANSFORMATION CALCULATIONS
    m_f_val = solver_heat.ablated_mass(time=t_ns_h)
    pos_scale_s = solver_shock._position_temporal_factor(time=t_ns_h)
    q1_s = 1.0 - solver_shock.omega
    q2_s = (2.0 - solver_shock.omega) / (solver_shock.tau + 2.0)
    
    x_p_fit_val = pos_scale_s * q2_s * solver_shock.U0
    
    xsi_mf_val = m_f_val * solver_shock.xsi_over_m(time=t_ns_h)
    y_mf_val = xsi_mf_val / solver_shock.xsi_s
    
    _, _, U_fit_sh_mf_val, rho_fit_sh_mf_val = shock_fitting.fit_by_params(np.array([y_mf_val]), params_shock)
    V_fit_sh_mf_val = 1.0 / rho_fit_sh_mf_val[0]
    
    x_af_fit_val = pos_scale_s * (q1_s * xsi_mf_val * V_fit_sh_mf_val + q2_s * U_fit_sh_mf_val[0])
    C_pos_val = solver_heat._position_temporal_factor(time=t_ns_h)
    x_b_fit_val = solver_heat.boundary_position(time=t_ns_h) + x_af_fit_val
    x_s_fit_val = solver_shock.shock_position(time=t_ns_h)
    x_b_sub_coeff = C_pos_val * solver_heat.U0

    # 4. DYNAMIC LATEX GENERATION
    doc_dir = Path("c:/Users/TLP-001/Documents/GitHub/project3_docs/fitting")
    template_path = doc_dir / "piecewise_analytic_formulas_template.tex"
    
    if not template_path.exists():
        print(f"Error: Template LaTeX file not found at {template_path}")
        sys.exit(1)
        
    print(f"2) Reading LaTeX template from: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        latex_template = f.read()
        
    # 5. REPLACE ALL PLACEHOLDERS WITH DYNAMIC fit / computed values
    print("3) Injecting fit coefficients into the LaTeX template...")
    latex_content = latex_template
    
    # Constants
    latex_content = latex_content.replace("__A_VAL__", f"{A_h:.5f}")
    latex_content = latex_content.replace("__B_VAL__", f"{B_h:.5e}")
    
    # Exponent Fractions
    latex_content = latex_content.replace("__A_V_FRAC__", a_v_frac)
    latex_content = latex_content.replace("__B_V_FRAC__", b_v_frac)
    latex_content = latex_content.replace("__C_V_FRAC__", c_v_frac)
    latex_content = latex_content.replace("__A_U_FRAC__", a_u_frac)
    latex_content = latex_content.replace("__B_U_FRAC__", b_u_frac)
    latex_content = latex_content.replace("__C_U_FRAC__", c_u_frac)
    latex_content = latex_content.replace("__A_P_FRAC__", a_p_frac)
    latex_content = latex_content.replace("__B_P_FRAC__", b_p_frac)
    latex_content = latex_content.replace("__C_P_FRAC__", c_p_frac)
    
    # Normalization mass scales
    latex_content = latex_content.replace("__M_F_SCALE__", f"{m_f_scale:.5f}")
    latex_content = latex_content.replace("__M_S_SCALE__", f"{m_s_scale:.5f}")
    latex_content = latex_content.replace("__XSI_S_S__", f"{xsi_s_val:.5f}")
    latex_content = latex_content.replace("__INV_MF__", f"{inv_mf:.5f}")
    latex_content = latex_content.replace("__INV_MS__", f"{inv_ms:.5f}")
    
    # Subsonic simple volume fit
    latex_content = latex_content.replace("__V0_H__", f"{V0_h:.5f}")
    latex_content = latex_content.replace("__D_V_H__", f"{d_v_h:.5f}")
    latex_content = latex_content.replace("__V_SCALE_H__", f"{v_scale_h:.5f}")
    latex_content = latex_content.replace("__LEAD_V_H__", f"{lead_coeff_v_h:.5f}")
    
    # Custom formatter for signed coefficients to prevent + - double signs in LaTeX
    def fmt_signed(val, format_spec=".5f"):
        formatted = f"{val:{format_spec}}"
        if val < 0 or formatted.startswith("-"):
            if formatted.startswith("-"):
                formatted = formatted[1:]
            return f"- {formatted}"
        else:
            return f"+ {formatted}"

    def fmt_signed_scientific(val, format_spec=".5e"):
        formatted = f"{val:{format_spec}}"
        is_neg = False
        if val < 0 or formatted.startswith("-"):
            is_neg = True
            if formatted.startswith("-"):
                formatted = formatted[1:]
        if "e" in formatted:
            base, exp = formatted.split("e")
            exp = int(exp)
            formatted = f"{base} \\times 10^{{{exp}}}"
        if is_neg:
            return f"- {formatted}"
        else:
            return f"+ {formatted}"

    # Subsonic Pressure Fit (Fit 4)
    latex_content = latex_content.replace("__PF_SUB__", f"{Pf_sub:.5f}")
    latex_content = latex_content.replace("__A_P_SUB_SIGNED__", fmt_signed(a_P_sub))
    latex_content = latex_content.replace("__B_P_SUB_SIGNED__", fmt_signed(b_P_sub))
    latex_content = latex_content.replace("__E_P_SUB_SIGNED__", fmt_signed(e_P_sub))
    latex_content = latex_content.replace("__C_P_SUB__", f"{c_P_sub:.5f}")
    latex_content = latex_content.replace("__CPD_P_SUB__", f"{cpd_P_sub:.5f}")
    latex_content = latex_content.replace("__CPDF_P_SUB__", f"{cpdf_P_sub:.5f}")
    latex_content = latex_content.replace("__P_SCALE_SUB__", f"{p_scale_h_mbar:.5f}")
    latex_content = latex_content.replace("__LEAD_PF_SUB__", f"{lead_pf_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_A_P_SUB_SIGNED__", fmt_signed(lead_a_p_sub))
    latex_content = latex_content.replace("__LEAD_B_P_SUB_SIGNED__", fmt_signed(lead_b_p_sub))
    latex_content = latex_content.replace("__LEAD_E_P_SUB_SIGNED__", fmt_signed(lead_e_p_sub))
    
    # Subsonic Velocity Fit (Fit 12)
    latex_content = latex_content.replace("__U0_SUB__", f"{U0_sub:.5f}")
    latex_content = latex_content.replace("__UF_SUB__", f"{UF_sub:.5f}")
    latex_content = latex_content.replace("__UA1_SUB_SIGNED__", fmt_signed(ua1))
    latex_content = latex_content.replace("__UA2_SUB_SIGNED__", fmt_signed(ua2))
    latex_content = latex_content.replace("__UA3_SUB_SIGNED__", fmt_signed(ua3))
    latex_content = latex_content.replace("__UA4_SUB_SIGNED__", fmt_signed(ua4))
    latex_content = latex_content.replace("__UB1_SUB_SIGNED__", fmt_signed(ub1))
    latex_content = latex_content.replace("__UB2_SUB_SIGNED__", fmt_signed(ub2))
    latex_content = latex_content.replace("__UB3_SUB_SIGNED__", fmt_signed(ub3))
    latex_content = latex_content.replace("__UB4_SUB_SIGNED__", fmt_signed(ub4))
    latex_content = latex_content.replace("__UALPHA_SUB__", f"{ualpha:.5f}")
    latex_content = latex_content.replace("__UY0_SUB__", f"{uy0:.5f}")
    latex_content = latex_content.replace("__U_SCALE_SUB__", f"{u_scale_h_kms:.5f}")
    latex_content = latex_content.replace("__LEAD_U0_SUB__", f"{lead_u0_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UF_SUB__", f"{lead_uf_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UA1_SUB_SIGNED__", fmt_signed(lead_ua1_sub))
    latex_content = latex_content.replace("__LEAD_UA2_SUB_SIGNED__", fmt_signed(lead_ua2_sub))
    latex_content = latex_content.replace("__LEAD_UA3_SUB_SIGNED__", fmt_signed(lead_ua3_sub))
    latex_content = latex_content.replace("__LEAD_UA4_SUB_SIGNED__", fmt_signed(lead_ua4_sub))
    latex_content = latex_content.replace("__LEAD_UB1_SUB_SIGNED__", fmt_signed(lead_ub1_sub))
    latex_content = latex_content.replace("__LEAD_UB2_SUB_SIGNED__", fmt_signed(lead_ub2_sub))
    latex_content = latex_content.replace("__LEAD_UB3_SUB_SIGNED__", fmt_signed(lead_ub3_sub))
    latex_content = latex_content.replace("__LEAD_UB4_SUB_SIGNED__", fmt_signed(lead_ub4_sub))
    
    # Subsonic Temperature & Density EOS formulas
    latex_content = latex_content.replace("__R_T_SUB__", f"{R_val:.5f}")
    latex_content = latex_content.replace("__DENOM_T__", f"{denom_T:.5f}")
    latex_content = latex_content.replace("__LEAD_RHO_SCALE_H__", f"{lead_rho_scale_h:.5f}")
    latex_content = latex_content.replace("__RHO_SCALE_SUB__", f"{rho_scale_h:.5f}")
    latex_content = latex_content.replace("__POWER_RHO_T_NUMER__", power_rho_t_numer)
    latex_content = latex_content.replace("__POWER_RHO_OUT__", power_rho_out)
    
    # Shock Pressure Fit
    latex_content = latex_content.replace("__PS_SHOCK__", f"{Ps_shock:.5f}")
    latex_content = latex_content.replace("__D_P_SHOCK__", f"{dP_shock:.5f}")
    latex_content = latex_content.replace("__P_SCALE_SHOCK__", f"{p_scale_s_mbar:.5f}")
    latex_content = latex_content.replace("__P_DIFF_COEFF_S_SIGNED__", fmt_signed(lead_p_diff))
    latex_content = latex_content.replace("__LEAD_P_DIFF_SHOCK_SIGNED__", fmt_signed(lead_coeff_p_diff_s))
    latex_content = latex_content.replace("__RHO_0_CGS__", f"{solver_shock.rho0:.5f}")
    
    # Shock Velocity Fit (Fit 3)
    latex_content = latex_content.replace("__CU_SHOCK__", f"{cu_shock:.5f}")
    latex_content = latex_content.replace("__BU_SHOCK__", f"{bu_shock:.5f}")
    latex_content = latex_content.replace("__U_SCALE_SHOCK__", f"{u_scale_s_kms:.5f}")
    latex_content = latex_content.replace("__LEAD_CU_SHOCK__", f"{lead_cu_shock:.5f}")
    latex_content = latex_content.replace("__AU_SHOCK_NEG_SIGNED__", fmt_signed(-au_shock))
    latex_content = latex_content.replace("__LEAD_AU_SHOCK_NEG_SIGNED__", fmt_signed(-lead_au_shock))
    
    # Shock Density (piecewise)
    latex_content = latex_content.replace("__RHO0_SHOCK__", f"{rho_0_shock:.5f}")
    latex_content = latex_content.replace("__RHOS_SHOCK__", f"{rho_s_shock:.5f}")
    latex_content = latex_content.replace("__A_RHO_LOW__", f"{arho_low:.5f}")
    latex_content = latex_content.replace("__D_RHO_LOW__", f"{drho_low:.5f}")
    
    # Shock Temperature Fit (piecewise high domain Fit 2)
    latex_content = latex_content.replace("__TS_SHOCK__", f"{Ts_shock:.5f}")
    latex_content = latex_content.replace("__C1_T_HIGH_SIGNED__", fmt_signed(c1_t))
    latex_content = latex_content.replace("__C2_T_HIGH_SIGNED__", fmt_signed(c2_t))
    latex_content = latex_content.replace("__A_T_HIGH__", f"{a_t:.5f}")
    latex_content = latex_content.replace("__B_T_HIGH__", f"{b_t:.5f}")
    latex_content = latex_content.replace("__T_SCALE_KELVIN_SHOCK__", f"{T_scale_kelvin:.5f}")
    latex_content = latex_content.replace("__T_POW_SHOCK__", f"{T_pow:.5f}")
    
    # Helper to format values in scientific notation for LaTeX
    def fmt_latex_scientific(val, format_spec=".5e"):
        formatted = f"{val:{format_spec}}"
        if "e" in formatted:
            base, exp = formatted.split("e")
            exp = int(exp)
            return f"{base} \\times 10^{{{exp}}}"
        return formatted

    # Helper to format error margins as percentages
    def fmt_err(val):
        pct = val * 100.0
        if pct < 1e-2 and pct > 0:
            return f"{pct:.3e}"
        else:
            return f"{pct:.3f}"

    # Error values replacements for Section 6
    latex_content = latex_content.replace("__ERR_UH_AVG__", fmt_err(avg_U_h))
    latex_content = latex_content.replace("__ERR_UH_MAX__", fmt_err(max_U_h))
    latex_content = latex_content.replace("__ERR_US_AVG__", fmt_err(avg_U_s))
    latex_content = latex_content.replace("__ERR_US_MAX__", fmt_err(max_U_s))
    
    latex_content = latex_content.replace("__ERR_PH_AVG__", fmt_err(avg_P_h))
    latex_content = latex_content.replace("__ERR_PH_MAX__", fmt_err(max_P_h))
    latex_content = latex_content.replace("__ERR_PS_AVG__", fmt_err(avg_P_s))
    latex_content = latex_content.replace("__ERR_PS_MAX__", fmt_err(max_P_s))
    
    latex_content = latex_content.replace("__ERR_RHOH_AVG__", fmt_err(avg_rho_h))
    latex_content = latex_content.replace("__ERR_RHOH_MAX__", fmt_err(max_rho_h))
    latex_content = latex_content.replace("__ERR_RHOS_AVG__", fmt_err(avg_rho_s))
    latex_content = latex_content.replace("__ERR_RHOS_MAX__", fmt_err(max_rho_s))
    
    latex_content = latex_content.replace("__ERR_TH_AVG__", fmt_err(avg_T_h))
    latex_content = latex_content.replace("__ERR_TH_MAX__", fmt_err(max_T_h))
    latex_content = latex_content.replace("__ERR_TS_AVG__", fmt_err(avg_T_s))
    latex_content = latex_content.replace("__ERR_TS_MAX__", fmt_err(max_T_s))
    
    # Eulerian transformations replacements
    latex_content = latex_content.replace("__X_AF_1NS__", fmt_latex_scientific(x_af_fit_val))
    latex_content = latex_content.replace("__C_POS_SCALE__", fmt_latex_scientific(C_pos_val))
    latex_content = latex_content.replace("__POS_SCALE_S__", fmt_latex_scientific(pos_scale_s))
    latex_content = latex_content.replace("__X_P_1NS__", fmt_latex_scientific(x_p_fit_val))
    latex_content = latex_content.replace("__X_S_1NS__", fmt_latex_scientific(x_s_fit_val))
    latex_content = latex_content.replace("__U0_SUB__", fmt_signed(solver_heat.U0))
    latex_content = latex_content.replace("__U0_SHOCK__", fmt_signed(solver_shock.U0))
    latex_content = latex_content.replace("__LEAD_XB_COEFF_SIGNED__", fmt_signed_scientific(x_b_sub_coeff))
    
    # Subsonic Temperature converted to Kelvin
    T0_kelvin = 1.0 * KELVIN_PER_HEV
    latex_content = latex_content.replace("__T0_KELVIN__", fmt_latex_scientific(T0_kelvin))

    # 3d. INTEGRATED ENERGIES CALCULATIONS & PRINTS & REPLACEMENTS
    # Subsonic Ablation energies
    C_E_ablation = A_h**(2*a_u - a_mf) * B_h**(2*b_u - b_mf)
    power_E_ablation = 2*c_u - c_mf
    
    ekin_integral_ablation = solver_heat.energy_kinetic_intgeral
    ein_integral_ablation = solver_heat.energy_internal_intgeral
    etot_integral_anal_ablation = ekin_integral_ablation + ein_integral_ablation
    
    ekin_coeff_ablation = C_E_ablation * (1e-9)**power_E_ablation * ekin_integral_ablation
    eint_coeff_ablation = C_E_ablation * (1e-9)**power_E_ablation * ein_integral_ablation
    etot_coeff_ablation = C_E_ablation * (1e-9)**power_E_ablation * etot_integral_anal_ablation
    
    # Shock Compressed energies
    import scipy.integrate
    omega_s = solver_shock.omega
    r_s = solver_shock.r
    C_E_shock = (v0_s)**(1.0 / (2.0 - omega_s)) * (p0_s)**((3.0 - omega_s) / (2.0 - omega_s))
    power_E_shock = (3.0 * tau_s - tau_s * omega_s + 2.0) / (2.0 - omega_s)
    
    y_grid_shock = np.linspace(0.0, 1.0, 500)
    xsi_vec_shock = y_grid_shock * xsi_s_val
    xsi_vec_shock[0] = 1e-10
    V_shock, U_shock, P_shock = solver_shock.get_self_similar_profiles(xsi_vec=xsi_vec_shock)
    
    integrand_kin_shock = 0.5 * U_shock**2
    integrand_in_shock = P_shock * V_shock / r_s
    
    ekin_integral_shock = scipy.integrate.simps(y=integrand_kin_shock, x=xsi_vec_shock)
    ein_integral_shock = scipy.integrate.simps(y=integrand_in_shock, x=xsi_vec_shock)
    etot_integral_shock = ekin_integral_shock + ein_integral_shock
    
    ekin_coeff_shock = C_E_shock * (1e-9)**power_E_shock * ekin_integral_shock
    eint_coeff_shock = C_E_shock * (1e-9)**power_E_shock * ein_integral_shock
    etot_coeff_shock = C_E_shock * (1e-9)**power_E_shock * etot_integral_shock

    # print("\n" + "=" * 80)
    # print("  INTEGRATED ENERGIES IN ABLATION REGIME")
    # print("=" * 80)
    # print("Kinetic Energy E_k(t):")
    # print("  Level 1: E_k(t) = A^(2a2-a) * B^(2b2-b) * t^(2c2-c) * integral_0^xsi_f (U(xsi)^2 / 2) dxsi")
    # print(f"  Level 2: E_k(t) = {C_E_ablation:.5e} * t^{float(power_E_ablation):.5f} * {ekin_integral_ablation:.5f}")
    # print(f"  Level 3: E_k(t) = {ekin_coeff_ablation:.5e} * (t_ns)^{float(power_E_ablation):.5f} erg/cm^2")
    
    # print("\nInternal Energy E_in(t):")
    # print("  Level 1: E_in(t) = A^(2a2-a) * B^(2b2-b) * t^(2c2-c) * integral_0^xsi_f (P(xsi) * V(xsi) / r) dxsi")
    # print(f"  Level 2: E_in(t) = {C_E_ablation:.5e} * t^{float(power_E_ablation):.5f} * {ein_integral_ablation:.5f}")
    # print(f"  Level 3: E_in(t) = {eint_coeff_ablation:.5e} * (t_ns)^{float(power_E_ablation):.5f} erg/cm^2")

    # print("\nTotal Energy E_tot(t):")
    # print("  Level 1: E_tot(t) = A^(2a2-a) * B^(2b2-b) * t^(2c2-c) * integral_0^xsi_f (U(xsi)^2 / 2 + P(xsi) * V(xsi) / r) dxsi")
    # print(f"  Level 2: E_tot(t) = {C_E_ablation:.5e} * t^{float(power_E_ablation):.5f} * {etot_integral_anal_ablation:.5f}")
    # print(f"  Level 3: E_tot(t) = {etot_coeff_ablation:.5e} * (t_ns)^{float(power_E_ablation):.5f} erg/cm^2")
    # print("=" * 80 + "\n")

    # print("=" * 80)
    # print("  INTEGRATED ENERGIES IN SHOCK REGIME")
    # print("=" * 80)
    # print("Kinetic Energy E_k(t):")
    # print("  Level 1: E_k(t) = v0**(1/(2-omega)) * p0**((3-omega)/(2-omega)) * t**((3*tau - tau*omega + 2)/(2-omega)) * integral_0^xsi_s (U(xsi)^2 / 2) dxsi")
    # print(f"  Level 2: E_k(t) = {C_E_shock:.5e} * t^{float(power_E_shock):.5f} * {ekin_integral_shock:.5f}")
    # print(f"  Level 3: E_k(t) = {ekin_coeff_shock:.5e} * (t_ns)^{float(power_E_shock):.5f} erg/cm^2")
    
    # print("\nInternal Energy E_in(t):")
    # print("  Level 1: E_in(t) = v0**(1/(2-omega)) * p0**((3-omega)/(2-omega)) * t**((3*tau - tau*omega + 2)/(2-omega)) * integral_0^xsi_s (P(xsi) * V(xsi) / r) dxsi")
    # print(f"  Level 2: E_in(t) = {C_E_shock:.5e} * t^{float(power_E_shock):.5f} * {ein_integral_shock:.5f}")
    # print(f"  Level 3: E_in(t) = {eint_coeff_shock:.5e} * (t_ns)^{float(power_E_shock):.5f} erg/cm^2")

    # print("\nTotal Energy E_tot(t):")
    # print("  Level 1: E_tot(t) = E_in(t) + E_k(t) = v0**(1/(2-omega)) * p0**((3-omega)/(2-omega)) * t**((3*tau - tau*omega + 2)/(2-omega)) * integral_0^xsi_s (P(xsi)*V(xsi)/r + U(xsi)^2/2) dxsi")
    # print(f"  Level 2: E_tot(t) = {C_E_shock:.5e} * t^{float(power_E_shock):.5f} * {etot_integral_shock:.5f}")
    # print(f"  Level 3: E_tot(t) = {etot_coeff_shock:.5e} * (t_ns)^{float(power_E_shock):.5f} erg/cm^2")
    # print("=" * 80 + "\n")

    # Integrated Energies replacements
    latex_content = latex_content.replace("__C_E_KIN_SUB__", fmt_latex_scientific(ekin_coeff_ablation))
    latex_content = latex_content.replace("__C_E_INT_SUB__", fmt_latex_scientific(eint_coeff_ablation))
    latex_content = latex_content.replace("__C_E_TOT_SUB__", fmt_latex_scientific(etot_coeff_ablation))
    latex_content = latex_content.replace("__C_E_KIN_SHOCK__", fmt_latex_scientific(ekin_coeff_shock))
    latex_content = latex_content.replace("__C_E_INT_SHOCK__", fmt_latex_scientific(eint_coeff_shock))
    latex_content = latex_content.replace("__C_E_TOT_SHOCK__", fmt_latex_scientific(etot_coeff_shock))
    latex_content = latex_content.replace("__RATIO_E_SUB__", f"{ekin_integral_ablation / ein_integral_ablation:.5f}")
    latex_content = latex_content.replace("__RATIO_E_SHOCK__", f"{ekin_integral_shock / ein_integral_shock:.5f}")
    
    # Target file paths
    tex_path = doc_dir / "piecewise_analytic_formulas.tex"
    print(f"4) Overwriting LaTeX source: {tex_path}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
        
    print("5) Compiling LaTeX document using pdflatex...")
    try:
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

    print("\n*** ALL TASKS COMPLETED SUCCESSFULLY! ***")

if __name__ == "__main__":
    main()

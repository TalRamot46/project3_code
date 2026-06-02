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
    case_heat, _ = get_preset(PRESET_FIG_8_CONSTANT_TEMPERATURE)
    solver_heat = sub_fitting.get_cached_sub_solver(case_heat, "heat_const_T")
    params_heat = sub_fitting.perform_subsonic_fitting(solver_heat)
    
    case_shock, _ = get_preset(PRESET_FIG_7_SHOCK_ONLY_ABLATION_FROM_CONSTANT_TEMPERATURE)
    solver_shock = shock_fitting.get_cached_shock_solver(case_shock, "shock_const_T")
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
    
    # Subsonic Pressure Fit (Fit 4)
    latex_content = latex_content.replace("__PF_SUB__", f"{Pf_sub:.5f}")
    latex_content = latex_content.replace("__A_P_SUB__", f"{a_P_sub:.5f}")
    latex_content = latex_content.replace("__B_P_SUB__", f"{b_P_sub:.5f}")
    latex_content = latex_content.replace("__E_P_SUB__", f"{e_P_sub:.5f}")
    latex_content = latex_content.replace("__C_P_SUB__", f"{c_P_sub:.5f}")
    latex_content = latex_content.replace("__CPD_P_SUB__", f"{cpd_P_sub:.5f}")
    latex_content = latex_content.replace("__CPDF_P_SUB__", f"{cpdf_P_sub:.5f}")
    latex_content = latex_content.replace("__P_SCALE_SUB__", f"{p_scale_h_mbar:.5f}")
    latex_content = latex_content.replace("__LEAD_PF_SUB__", f"{lead_pf_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_A_P_SUB__", f"{lead_a_p_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_B_P_SUB__", f"{lead_b_p_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_E_P_SUB__", f"{lead_e_p_sub:.5f}")
    
    # Subsonic Velocity Fit (Fit 12)
    latex_content = latex_content.replace("__U0_SUB__", f"{U0_sub:.5f}")
    latex_content = latex_content.replace("__UF_SUB__", f"{UF_sub:.5f}")
    latex_content = latex_content.replace("__UA1_SUB__", f"{ua1:.5f}")
    latex_content = latex_content.replace("__UA2_SUB__", f"{ua2:.5f}")
    latex_content = latex_content.replace("__UA3_SUB__", f"{ua3:.5f}")
    latex_content = latex_content.replace("__UA4_SUB__", f"{ua4:.5f}")
    latex_content = latex_content.replace("__UB1_SUB__", f"{ub1:.5f}")
    latex_content = latex_content.replace("__UB2_SUB__", f"{ub2:.5f}")
    latex_content = latex_content.replace("__UB3_SUB__", f"{ub3:.5f}")
    latex_content = latex_content.replace("__UB4_SUB__", f"{ub4:.5f}")
    latex_content = latex_content.replace("__UALPHA_SUB__", f"{ualpha:.5f}")
    latex_content = latex_content.replace("__UY0_SUB__", f"{uy0:.5f}")
    latex_content = latex_content.replace("__U_SCALE_SUB__", f"{u_scale_h_kms:.5f}")
    latex_content = latex_content.replace("__LEAD_U0_SUB__", f"{lead_u0_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UF_SUB__", f"{lead_uf_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UA1_SUB__", f"{lead_ua1_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UA2_SUB__", f"{lead_ua2_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UA3_SUB__", f"{lead_ua3_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UA4_SUB__", f"{lead_ua4_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UB1_SUB__", f"{lead_ub1_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UB2_SUB__", f"{lead_ub2_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UB3_SUB__", f"{lead_ub3_sub:.5f}")
    latex_content = latex_content.replace("__LEAD_UB4_SUB__", f"{lead_ub4_sub:.5f}")
    
    # Subsonic Temperature & Density EOS formulas
    latex_content = latex_content.replace("__R_T_SUB__", f"{R_val:.5f}")
    latex_content = latex_content.replace("__DENOM_T__", f"{denom_T:.5f}")
    latex_content = latex_content.replace("__LEAD_RHO_SCALE_H__", f"{lead_rho_scale_h:.5f}")
    latex_content = latex_content.replace("__POWER_RHO_T_NUMER__", power_rho_t_numer)
    latex_content = latex_content.replace("__POWER_RHO_OUT__", power_rho_out)
    
    # Shock Pressure Fit
    latex_content = latex_content.replace("__PS_SHOCK__", f"{Ps_shock:.5f}")
    latex_content = latex_content.replace("__D_P_SHOCK__", f"{dP_shock:.5f}")
    latex_content = latex_content.replace("__P_SCALE_SHOCK__", f"{p_scale_s_mbar:.5f}")
    latex_content = latex_content.replace("__LEAD_P_DIFF_SHOCK__", f"{lead_coeff_p_diff_s:.5f}")
    
    # Shock Velocity Fit (Fit 3)
    latex_content = latex_content.replace("__CU_SHOCK__", f"{cu_shock:.5f}")
    latex_content = latex_content.replace("__AU_SHOCK__", f"{au_shock:.5f}")
    latex_content = latex_content.replace("__BU_SHOCK__", f"{bu_shock:.5f}")
    latex_content = latex_content.replace("__U_SCALE_SHOCK__", f"{u_scale_s_kms:.5f}")
    latex_content = latex_content.replace("__LEAD_CU_SHOCK__", f"{lead_cu_shock:.5f}")
    latex_content = latex_content.replace("__LEAD_AU_SHOCK__", f"{lead_au_shock:.5f}")
    
    # Shock Density (piecewise)
    latex_content = latex_content.replace("__RHO0_SHOCK__", f"{rho_0_shock:.5f}")
    latex_content = latex_content.replace("__RHOS_SHOCK__", f"{rho_s_shock:.5f}")
    latex_content = latex_content.replace("__A_RHO_LOW__", f"{arho_low:.5f}")
    latex_content = latex_content.replace("__D_RHO_LOW__", f"{drho_low:.5f}")
    
    # Shock Temperature Fit (piecewise high domain Fit 2)
    latex_content = latex_content.replace("__TS_SHOCK__", f"{Ts_shock:.5f}")
    latex_content = latex_content.replace("__C1_T_HIGH__", f"{c1_t:.5f}")
    latex_content = latex_content.replace("__C2_T_HIGH__", f"{c2_t:.5f}")
    latex_content = latex_content.replace("__A_T_HIGH__", f"{a_t:.5f}")
    latex_content = latex_content.replace("__B_T_HIGH__", f"{b_t:.5f}")
    latex_content = latex_content.replace("__T_SCALE_KELVIN_SHOCK__", f"{T_scale_kelvin:.5f}")
    latex_content = latex_content.replace("__T_POW_SHOCK__", f"{T_pow:.5f}")
    
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

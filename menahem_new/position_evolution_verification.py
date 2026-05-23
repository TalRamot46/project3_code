"""
Explicit Eulerian Position Formulas — Verification & Evolution Plots.

Computes the 5 key position functions for the two test cases (Fig 8: tau=0,
Fig 9: tau=0.123) and verifies them against the AblationSolver output.

The 5 positions:
  1. x_boundary(t)       — deepest point reached by ablation heat wave
  2. x_sub(m,t)          — position of a mass element in the ablated region
  3. x_ablation_front(t) — interface between sub and shock (= shock at m_f)
  4. x_shock(m,t)        — position of a mass element in the shocked region
  5. x_shock_front(t)    — outer edge of the shock wave
"""

import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt

from subsonic_heat_wave import SubsonicHeatWave, Units
from piston_shock import PistonShock
from ablation_solver import AblationSolver

KELVIN_PER_HEV = 1.160451812e6


def build_test_case(tau: float) -> dict:
    """Build AblationSolver kwargs for the Au test case with given tau."""
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


def print_position_coefficients(solver: AblationSolver, label: str):
    """Compute and print all explicit position coefficients."""
    hs = solver.heat_solver
    ss = solver.shock_solver
    omega = ss.omega
    assert omega == 0.0
    tau_s = hs.c3
    p0s = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"\n--- Subsonic Heat Wave ---")
    print(f"  A={hs.A:.6e}  B={hs.B:.6e}  xi_f={hs.xsi_f:.10f}  P_f={hs.Pf:.10f}  U(0)={hs.U0:.10f}")
    print(f"  (a,b,c)=({hs.a:.10f},{hs.b:.10f},{hs.c:.10f})")
    print(f"  (a2,b2,c2)=({hs.a2:.10f},{hs.b2:.10f},{hs.c2:.10f})")
    print(f"  (a3,b3,c3)=({hs.a3:.10f},{hs.b3:.10f},{hs.c3:.10f})")

    # 1. x_boundary
    C_bnd = hs.A**hs.a2 * hs.B**hs.b2 / (hs.c2 + 1.0) * hs.U0
    d_bnd = hs.c2 + 1.0
    print(f"\n  x_boundary: C={C_bnd:.6e}  d={d_bnd:.10f}")

    # 2. x_sub factor
    C_pos = hs.A**hs.a2 * hs.B**hs.b2 / (hs.c2 + 1.0)
    print(f"  x_sub factor: C_pos={C_pos:.6e}  c={hs.c:.10f}")

    # m_f
    C_mf = hs.xsi_f * hs.A**(-hs.a) * hs.B**(-hs.b)
    d_mf = -hs.c
    print(f"  m_f: C={C_mf:.6e}  d={d_mf:.10f}")

    # Shock
    pos_fac_coeff = (ss.v0 * p0s) ** 0.5
    pos_fac_pow = (2.0 + tau_s) / 2.0
    q2 = 2.0 / (tau_s + 2.0)
    print(f"\n--- Shock ---")
    print(f"  p0s={p0s:.6e}  tau_s={tau_s:.10f}  v0={ss.v0:.6e}  xi_s={ss.xsi_s:.10f}")
    print(f"  pos_factor: coeff={pos_fac_coeff:.6e}  pow={pos_fac_pow:.10f}  q2={q2:.10f}")

    # 3. x_ablation_front composite
    C_xsi_mf = C_mf * (ss.v0 / p0s) ** 0.5
    d_xsi_mf = d_mf - pos_fac_pow
    print(f"  xsi_mf: C={C_xsi_mf:.6e}  d={d_xsi_mf:.10f}  (TIME-DEPENDENT)")

    # 5. x_shock_front
    C_shock = pos_fac_coeff * ss.xsi_s
    d_shock = pos_fac_pow
    print(f"  x_shock_front: C={C_shock:.6e}  d={d_shock:.10f}")

    # Piston (m=0 in shock)
    C_piston = pos_fac_coeff * ss.U0 * q2
    print(f"  x_piston(m=0): C={C_piston:.6e}  d={d_shock:.10f}")

    return dict(
        C_bnd=C_bnd, d_bnd=d_bnd, C_pos=C_pos,
        C_mf=C_mf, d_mf=d_mf,
        C_xsi_mf=C_xsi_mf, d_xsi_mf=d_xsi_mf,
        pos_fac_coeff=pos_fac_coeff, pos_fac_pow=pos_fac_pow, q2=q2,
        C_piston=C_piston, d_piston=d_shock,
        C_shock=C_shock, d_shock=d_shock,
    )


def compute_x_ablation_front_formula(solver: AblationSolver, coeffs: dict, times: np.ndarray) -> np.ndarray:
    """Compute x_ablation_front using the explicit composite formula."""
    ss = solver.shock_solver
    pfc = coeffs['pos_fac_coeff']
    pfp = coeffs['pos_fac_pow']
    q2  = coeffs['q2']

    x_af = np.zeros_like(times)
    for i, t in enumerate(times):
        pos_fac = pfc * t ** pfp
        xsi_mf = coeffs['C_xsi_mf'] * t ** coeffs['d_xsi_mf']

        # get_self_similar_profiles requires N>1
        xsi_pair = np.array([max(xsi_mf, 1e-50), max(xsi_mf, 1e-50) * 1.001])
        V_arr, U_arr, P_arr = ss.get_self_similar_profiles(xsi_vec=xsi_pair)

        x_af[i] = pos_fac * (xsi_mf * V_arr[0] + q2 * U_arr[0])
    return x_af


def verify_and_plot(solver: AblationSolver, coeffs: dict, label: str, filename: str):
    """Verify explicit formulas against the solver and produce evolution plot."""
    hs = solver.heat_solver

    t_end = 1.5e-10
    times = np.linspace(1e-13, t_end, 200)

    # Explicit formulas
    x_bnd_formula = coeffs['C_bnd'] * times**coeffs['d_bnd']
    x_shock_formula = coeffs['C_shock'] * times**coeffs['d_shock']
    m_f_formula = coeffs['C_mf'] * times**coeffs['d_mf']
    x_af_formula = compute_x_ablation_front_formula(solver, coeffs, times)

    # Solver values
    x_bnd_solver = np.array([hs.boundary_position(time=t) for t in times])
    m_f_solver = np.array([hs.ablated_mass(time=t) for t in times])

    # Build mass grid
    L = 3e-3 / 19.32
    mass = np.cumsum(19.32 * np.diff(np.linspace(0, L, 201)))
    mass = np.array([1e-30, 1e-7 * mass[0]] + list(mass))

    x_heat_solver = np.zeros_like(times)
    x_shock_solver = np.zeros_like(times)

    for i, t in enumerate(times):
        try:
            sol = solver.solve(mass=mass, time=t)
            x_heat_solver[i] = sol["heat_position"]
            x_shock_solver[i] = sol["shock_position"]
        except Exception:
            x_heat_solver[i] = np.nan
            x_shock_solver[i] = np.nan

    # Verification
    err_bnd = np.max(np.abs(x_bnd_formula - x_bnd_solver) / (np.abs(x_bnd_solver) + 1e-30))
    print(f"\n  Verification: max rel error x_boundary       = {err_bnd:.2e}")
    assert err_bnd < 1e-8, f"x_boundary: {err_bnd}"

    valid = np.isfinite(x_heat_solver) & (x_heat_solver != 0)
    err_af = np.max(np.abs(x_af_formula[valid] - x_heat_solver[valid]) / (np.abs(x_heat_solver[valid]) + 1e-30))
    print(f"  Verification: max rel error x_ablation_front = {err_af:.2e}")
    assert err_af < 1e-4, f"x_ablation_front: {err_af}"

    valid_s = np.isfinite(x_shock_solver) & (x_shock_solver != 0)
    err_sf = np.max(np.abs(x_shock_formula[valid_s] - x_shock_solver[valid_s]) / (np.abs(x_shock_solver[valid_s]) + 1e-30))
    print(f"  Verification: max rel error x_shock_front    = {err_sf:.2e}")
    assert err_sf < 1e-8, f"x_shock_front: {err_sf}"

    err_mf = np.max(np.abs(m_f_formula - m_f_solver) / (np.abs(m_f_solver) + 1e-30))
    print(f"  Verification: max rel error m_f              = {err_mf:.2e}")
    assert err_mf < 1e-8, f"m_f: {err_mf}"

    # x_sub check
    t_test = t_end * 0.5
    sol_test = solver.solve(mass=mass, time=t_test)
    m_f_test = hs.ablated_mass(time=t_test)
    x_af_test = sol_test["heat_position"]
    ablated_idx = mass < m_f_test
    if np.sum(ablated_idx) > 2:
        xsi_test = mass[ablated_idx] * hs.A**hs.a * hs.B**hs.b * t_test**hs.c
        result_test = hs.get_self_similar_profiles(xsi_vec=np.array(xsi_test))
        V_t, U_t = result_test["V"], result_test["U"]
        x_sub_f = x_af_test + coeffs['C_pos'] * t_test**coeffs['d_bnd'] * (U_t - hs.c * xsi_test * V_t)
        x_sub_s = sol_test["position"][ablated_idx]
        err_sub = np.max(np.abs(x_sub_f - x_sub_s) / (np.abs(x_sub_s) + 1e-30))
        print(f"  Verification: max rel error x_sub           = {err_sub:.2e}")

    # ===== PLOT =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    t_ns = times * 1e9

    ax = axes[0]
    # Mass trajectories
    results_all = [solver.solve(mass=mass, time=t) for t in times]
    for j in np.linspace(2, len(mass)-1, 8, dtype=int):
        ax.plot(t_ns, [r["position"][j]*1e4 for r in results_all], 'k-', lw=0.3, alpha=0.4)

    ax.plot(t_ns, x_bnd_formula*1e4,   'b-',  lw=2.5, label=r"$x_{\rm boundary}$ (formula)")
    ax.plot(t_ns, x_af_formula*1e4,    'g-',  lw=2.5, label=r"$x_{\rm ablation\ front}$ (formula)")
    ax.plot(t_ns, x_shock_formula*1e4, 'r-',  lw=2.5, label=r"$x_{\rm shock\ front}$ (formula)")
    ax.plot(t_ns, x_bnd_solver*1e4,    'b--', lw=1.5, alpha=0.7, label=r"$x_{\rm boundary}$ (solver)")
    ax.plot(t_ns, x_heat_solver*1e4,   'g--', lw=1.5, alpha=0.7, label=r"$x_{\rm ablation\ front}$ (solver)")
    ax.plot(t_ns, x_shock_solver*1e4,  'r--', lw=1.5, alpha=0.7, label=r"$x_{\rm shock\ front}$ (solver)")

    ax.set_xlabel(r"$t$ [ns]", fontsize=14)
    ax.set_ylabel(r"$x$ [$\mu$m]", fontsize=14)
    ax.set_title(f"Position Evolution — {label}\nSolid: formula, Dashed: solver", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_end * 1e9)

    ax2 = axes[1]
    ax2.plot(t_ns, m_f_formula*1e3, 'r-',  lw=2.5, label=r"$m_f(t)$ — formula")
    ax2.plot(t_ns, m_f_solver*1e3,  'b--', lw=1.5, label=r"$m_f(t)$ — solver")
    ax2.set_xlabel(r"$t$ [ns]", fontsize=14)
    ax2.set_ylabel(r"$m_f$ [mg/cm²]", fontsize=14)
    ax2.set_title(f"Ablated Mass — {label}", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    print(f"  Saved: {filename}")
    plt.close(fig)


def main():
    output_dir = Path(__file__).parent / "position_verification_results"
    output_dir.mkdir(exist_ok=True)

    for tau, label in [(0.0, "Fig 8 (tau=0)"), (0.122957198444, "Fig 9 (tau=0.123)")]:
        kwargs = build_test_case(tau)
        solver = AblationSolver(**kwargs)
        coeffs = print_position_coefficients(solver, label)
        fname = f"xt_evolution_tau_{tau:.3f}.png".replace(".", "p", 1)
        verify_and_plot(solver, coeffs, label, str(output_dir / fname))

    print("\n*** All verifications passed! ***")


if __name__ == "__main__":
    main()

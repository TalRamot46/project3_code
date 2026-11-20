"""Tables and figures of the Be / Al runs (light_analytic_start.py), md/misalignment/01_surface_equilibrium.md section 3.

    python -m test_misalignment.light_analytic_start_figures
"""
from __future__ import annotations

import logging

import numpy as np

logging.disable(logging.INFO)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from test_misalignment import light_analytic_start as L
from test_misalignment.test3_transparency import FIGURES

K = ivd.K_PER_HEV
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#1a1a1a"
TIMES = (0.2e-9, 0.5e-9, 1e-9, 2e-9)
RUN_TEXT = {"cold_paper": "cold start, paper bath", "analytic_paper": "analytic start, paper bath",
            "analytic_calibrated": "analytic start, T-calibrated bath",
            "analytic_calibrated_flux": "analytic start, flux-calibrated bath",
            "cold_paper_n2500": "cold start, paper bath, 2500 cells",
            "analytic_paper_n2500": "analytic start, paper bath, 2500 cells"}


def solver(mat):
    from rad_hydro_sim.problems import presets_config as P
    L._register()
    case, _ = L.case_of(f"{mat}_cold", "paper")
    return L.solver_of(case, mat if mat == "be" else "test6_full"), case


def fronts(r, case, k):
    m = r["m"]
    mc = m - 0.5 * np.diff(m, prepend=0.0)
    T = r["T_material"][k].astype(float)
    rho = r["rho"][k].astype(float)
    rho_init = ivd._initial_density(m, case)
    mh, ms, _ = ivd.fronts_of_profile(m, T, rho, r["p"][k].astype(float), rho_init)
    j = int(np.searchsorted(mc, mh)) if np.isfinite(mh) else mc.size
    p_dense = float(r["p"][k][j:j + 10].max()) if j < mc.size else np.nan
    dm = np.diff(m, prepend=0.0)
    kap = rho**case.lambda_ / (case.g_Kelvin * np.maximum(T, 1.0)**case.alpha)
    hot = T > 0.5 * T[0]
    hot[j:] = False
    i_mid = min(int(np.searchsorted(mc, 0.5 * mh)), mc.size - 1) if np.isfinite(mh) else 0
    return dict(m_heat=mh, m_shock=ms, p_dense=p_dense, Tm0=T[0] / K, Tr0=float(r["T_rad"][k][0]) / K,
                eq_mid=T[i_mid] / float(r["T_rad"][k][i_mid]), tau=float(np.sum((kap * dm)[hot])))


def solver_values(s, case, t):
    hs = s.heat_solver
    mf = float(hs.ablated_mass(time=t))
    ms = float(np.ravel(s.shock_solver.shocked_mass(time=t))[0])
    p0 = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3
    m = np.geomspace(2e-3 * mf, 0.999 * mf, 500)
    sol = s.solve(mass=np.concatenate((m, [1.1 * mf, 1.2 * mf, 1.3 * mf])), time=t)
    rho, T = np.asarray(sol["density"])[:500], np.asarray(sol["temperature"])[:500]
    hot = T > 0.5 * K
    tau = float(np.trapz((rho**case.lambda_ / (case.g_Kelvin * T**case.alpha))[hot], m[hot]))
    return dict(m_heat=mf, m_shock=ms, p_dense=float(p0 * t**hs.c3), tau=tau)


def table():
    lines = []
    for mat in ("al", "be"):
        s, case = solver(mat)
        lines.append(f"\n**{ {'al': 'Al (Test 6)', 'be': 'Be'}[mat] }**, over the solver's (𝒯: optical depth of the "
                     "matter hotter than half the surface temperature; solver's value in parentheses)\n")
        lines.append("| run | t [ns] | T_matter(0) [HeV] | T_rad(0) [HeV] | T_m/T_r mid-layer | heat front | shock | "
                     "front pressure | 𝒯 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for tag in RUN_TEXT:
            path = L.OUT / f"{mat}_{tag}.npz"
            if not path.exists():
                continue
            r = np.load(path)
            for t in TIMES:
                k = int(np.argmin(np.abs(r["t"] - t)))
                f, sv = fronts(r, case, k), solver_values(s, case, float(r["t"][k]))
                lines.append(f"| {RUN_TEXT[tag]} | {r['t'][k] * 1e9:.2f} | {f['Tm0']:.3f} | {f['Tr0']:.3f} | "
                             f"{f['eq_mid']:.3f} | {f['m_heat'] / sv['m_heat']:.3f} | {f['m_shock'] / sv['m_shock']:.3f} | "
                             f"{f['p_dense'] / sv['p_dense']:.3f} | {f['tau']:.2f} ({sv['tau']:.2f}) |")
    text = "\n".join(lines)
    (FIGURES / "light_table.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def figure():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, mat in enumerate(("al", "be")):
        s, case = solver(mat)
        for i, tag in enumerate(RUN_TEXT):
            path = L.OUT / f"{mat}_{tag}.npz"
            if not path.exists() or (mat, tag) == ("be", "analytic_calibrated") or tag.endswith("n2500"):
                continue
            r = np.load(path)
            ks = np.flatnonzero(r["t"] > 2e-11)[::2]
            F = [fronts(r, case, k) for k in ks]
            Sv = [solver_values(s, case, float(r["t"][k])) for k in ks]
            t = r["t"][ks] * 1e9
            ax = axes[row, 0]
            ax.plot(t, [f["Tm0"] for f in F], color=C[i], lw=1.6, label=RUN_TEXT[tag] + ", matter")
            ax.plot(t, [f["Tr0"] for f in F], color=C[i], lw=1.0, ls="--", label=RUN_TEXT[tag] + ", radiation")
            axes[row, 1].plot(t, [f["m_heat"] / v["m_heat"] for f, v in zip(F, Sv)], color=C[i], lw=1.6)
            axes[row, 1].plot(t, [f["m_shock"] / v["m_shock"] for f, v in zip(F, Sv)], color=C[i], lw=1.0, ls="--")
            axes[row, 2].plot(t, [f["p_dense"] / v["p_dense"] for f, v in zip(F, Sv)], color=C[i], lw=1.6)
        name = {"al": "Al (Test 6)", "be": "Be"}[mat]
        axes[row, 0].axhline(1.0, color=INK, lw=1.4)
        axes[row, 0].set_ylabel(f"{name}: surface T [HeV]")
        axes[row, 0].set_ylim(0.5, 1.8)
        axes[row, 1].set_ylim(0.6, 1.6)
        axes[row, 2].set_ylim(0.0, 2.0)
        axes[row, 1].set_ylabel("front mass / solver's (solid: heat, dashed: shock)")
        axes[row, 2].set_ylabel("front pressure (dense side) / solver's")
        for ax in axes[row]:
            ax.set_xscale("log")
            ax.set_xlabel("t [ns]")
            ax.axhline(1.0, color="#888", lw=0.8)
            ivd.style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "light_materials.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    table()
    figure()

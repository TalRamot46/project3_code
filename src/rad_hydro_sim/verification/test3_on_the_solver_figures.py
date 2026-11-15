"""Figures and table of test3_on_the_solver.py (md/phase_c/test3_on_the_solver.md).

- startup.png: front pressure, heat front and shock over the solver's against time (cold starts).
- profiles_<run>.png: density, pressure, velocity, temperature in the shocked region at 1, 1.5, 2 ns.
- full_<run>.png: the same over the whole domain, against the full AblationSolver.
- ratio_cold.png, ratio_seeded.png: density, pressure, velocity over the solver's and the entropy
  against the solver's, at 1, 1.5 and 2 ns.
- entropy.png: the entropy p / rho^gamma against the solver's, and its change between 1 and 2 ns.
- chi_convergence.png, chi_table.md: the 2T cold start against the coupling factor chi.
- table.md: the ratios and the fronts at 2 ns.
"""
from __future__ import annotations

import numpy as np

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from rad_hydro_sim.verification import test3_hydro_only as h
from rad_hydro_sim.verification import test3_hydro_only_figures as hf
from rad_hydro_sim.verification import test3_on_the_solver as t3

K_PER_HEV = ivd.K_PER_HEV
BLUES, INK, ORANGE = hf.BLUES, hf.INK, hf.ORANGE
GREY = "#9aa0a6"
COLORS = {"cold_1T_30K": "#c0392b", "cold_1T_300K": "#12305e", "cold_2T_30K": "#2f9e8f",
          "seeded_2T_30K": "#2f9e8f", "seeded_2T_300K": "#12305e", "seeded_2T_30K_smeared": "#c0392b",
          t3.CHI100_SEEDED: "#b8860b"}
# the runs before this work, on the old mesh at 300 K: (file stem, tags, text, colour)
OLD = {"cold": (("test3_smooth", ["diffusion"], "2T diffusion, old mesh, 300 K", GREY),
                ("test3_smooth", ["imc1", "imc2", "imc3"], "IMC (3 seeds), old mesh, 300 K", ORANGE)),
       "seeded": (("test3_smooth_analytic", ["diffusion"], "2T diffusion, old mesh, 300 K (fix A)", GREY),
                  ("test3_smooth_analytic", ["imc1", "imc2", "imc3"], "IMC (3 seeds), old mesh, 300 K (fix A)", ORANGE))}
TABLE_MASSES = np.array([2, 3, 5, 8, 12, 20, 30, 50, 100, 150, 200]) * 1e-3


def _new(start):
    return [n for n, (st, _, _) in t3.RUNS.items()
            if st == start and n not in t3.CHI and n not in t3.IMC and n not in t3.IMC_COLD and n != t3.CHI100_FINE and n != t3.IMC_X4
            and (t3.OUT / f"{n}.npz").exists()]


def _load(name):
    return np.load(t3.OUT / f"{name}.npz")


def _old(stem, tag):
    return np.load(ivd.DEFAULT_OUT / f"{stem}_{tag}.npz")


def _mean_ratios(s, runs, t, m_lo):
    """Cell-centre masses, the mean ratios to the solver (and "S", the entropy function
    p / rho^gamma itself, where the ratio is defined), and the shock mass over the solver's."""
    parts = [hf._ratios(s, r, t, m_lo=m_lo) for r in runs]
    ratios = {k: np.mean([p[1][k] for p in parts], axis=0) for k in ("rho", "p", "u", "A")}
    k = hf._frame(runs[0], t)
    S = np.mean([r["p"][k] / r["rho"][k]**h.GAMMA for r in runs], axis=0)
    ratios["S"] = np.where(np.isfinite(ratios["A"]), S, np.nan)
    return parts[0][0], ratios, float(np.mean([p[2] for p in parts]))


IMC_COLOR = "#8e44ad"


def _imc_runs(start="seeded"):
    names = t3.IMC if start == "seeded" else t3.IMC_COLD
    return [_load(n) for n in names if (t3.OUT / f"{n}.npz").exists()]


def _imc_text(start="seeded"):
    return (f"IMC ({len(_imc_runs(start))} seeds), new mesh, " + ("smeared shock, " if start == "seeded" else "") + "30 K")


def _curves(s, start, t):
    """[(mc, ratios, shock ratio, colour, text, lw)] of the old and the new runs of a start."""
    m_lo = 1.05 * float(s.heat_solver.ablated_mass(time=t))
    out = []
    for stem, tags, text, color in OLD[start]:
        mc, r, shock = _mean_ratios(s, [_old(stem, tag) for tag in tags], t, m_lo)
        out.append((mc, r, shock, color, text, 1.0))
    if _imc_runs(start):
        mc, r, shock = _mean_ratios(s, _imc_runs(start), t, m_lo)
        out.append((mc, r, shock, IMC_COLOR, _imc_text(start), 1.2))
    if start == "seeded" and (t3.OUT / f"{t3.IMC_X4}.npz").exists():
        mc, r, shock = _mean_ratios(s, [_load(t3.IMC_X4)], t, m_lo)
        out.append((mc, r, shock, "#e83e8c", t3.RUN_TEXT[t3.IMC_X4], 1.2))
    for n in _new(start):
        mc, r, shock = _mean_ratios(s, [_load(n)], t, m_lo)
        out.append((mc, r, shock, COLORS[n], t3.RUN_TEXT[n], 1.7))
    return out


# --------------------------------------------------------------------------- the start
def plot_startup(path):
    import matplotlib.pyplot as plt

    s = h.solver()
    p0, c3 = h.solver_pressure_law(s)
    hs, ps = s.heat_solver, s.shock_solver
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    runs = []
    old = _old("test3_smooth", "diffusion")
    ok = np.isfinite(old["p_abl"])
    kappa = h.front_pressure_factor("test3_smooth", ["diffusion"])
    runs.append((old["series_t"][ok], kappa * old["p_abl"][ok], old["m_heat"][ok], old["m_shock"][ok], None,
                 GREY, "2T diffusion, old mesh, 300 K"))
    first_imc = [n for n in t3.IMC_COLD if (t3.OUT / f"{n}.npz").exists()][:1]
    for n in _new("cold") + first_imc:
        r = _load(n)
        ok = np.isfinite(r["p_dense"]) & (r["series_t"] > 0)
        runs.append((r["series_t"][ok], r["p_dense"][ok], r["m_heat"][ok], r["m_shock"][ok], r["T_surface"][ok],
                     COLORS.get(n, IMC_COLOR), t3.RUN_TEXT[n]))
    for t, p, mh, ms, Ts, color, text in runs:
        m_f = np.array([float(hs.ablated_mass(time=x)) for x in t])
        m_s = np.array([float(np.ravel(ps.shocked_mass(time=x))[0]) for x in t])
        axes[0].semilogx(t * 1e9, p / (p0 * t**c3), color=color, lw=1.4, label=text)
        axes[1].semilogx(t * 1e9, mh / m_f, color=color, lw=1.4)
        axes[2].semilogx(t * 1e9, ms / m_s, color=color, lw=1.4)
        if Ts is not None:
            axes[3].semilogx(t * 1e9, Ts / K_PER_HEV, color=color, lw=1.4)
    for ax, title, ylim in zip(axes, ("Pressure on the dense side of the front / solver's",
                                      "Heat front mass / solver's", "Shock mass / solver's",
                                      "Matter temperature at the surface [HeV]"),
                               ((0.75, 1.3), (0.9, 2.0), (0.9, 1.3), (0.0, 1.1))):
        ax.axhline(1.0, color=INK, lw=1.0)
        ax.set_xlim(1e-6, 3.0)
        ax.set_ylim(*ylim)
        ax.set_xlabel("t [ns]")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        hf._style(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(r"Test 3: Au ($\omega=+0.5$) · the first picoseconds of the cold-start runs", x=0.04, ha="left",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- profiles of one run
def plot_profiles(name, path, region="shock"):
    import matplotlib.pyplot as plt

    run = _load(name)
    s = h.solver()
    case = hf._case()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4))
    panels = (("rho", r"$\rho$ [g/cm$^3$]", "Density", 1.0), ("p", r"$P$ [Mbar]", "Pressure", 1e-12),
              ("u", r"$u$ [km/s]", "Velocity", 1e-5), ("T_material", r"$T$ [HeV]", "Temperature", 1 / K_PER_HEV))
    lims = {key: [np.inf, -np.inf] for key, *_ in panels}
    x_lo = 1.0 if region == "shock" else 1e-2
    for j, t_ns in enumerate(h.FIGURE_NS):
        t = t_ns * 1e-9
        k = hf._frame(run, t)
        tk = float(run["t"][k])
        mc = hf._mc(run["m"][k])
        m_f = float(s.heat_solver.ablated_mass(time=tk))
        m_s = float(np.ravel(s.shock_solver.shocked_mass(time=tk))[0])
        if region == "shock":
            sel = mc > 1.02 * m_f
            ref, _ = hf._reference(s, mc, tk)
            ref["T"] = (ref["pressure"] * ref["density"]**(case.mu - 1.0) / (case.r * case.f_Kelvin))**(1.0 / case.beta_Rosen)
            y_ref = {"rho": ref["density"], "p": ref["pressure"], "u": ref["velocity"], "T_material": ref["T"]}
            m_ref = mc
        else:
            sel = mc > 0
            m_ref = np.unique(np.concatenate((mc[mc < 0.999 * m_s], np.geomspace(0.9 * m_f, 1.1 * m_f, 400))))
            m_ref = m_ref[m_ref < 0.999 * m_s]
            sol = s.solve(mass=m_ref, time=tk)
            y_ref = {"rho": sol["density"], "p": sol["pressure"], "u": sol["velocity"], "T_material": sol["temperature"]}
        for ax, (key, _, _, unit) in zip(axes.flat, panels):
            ax.plot(mc[sel] * 1e3, run[key][k][sel] * unit, color=BLUES[j], lw=1.7, label=f"Simulation, {t_ns:g} ns", zorder=3)
            inside = m_ref < m_s
            edge = np.concatenate((m_ref[inside], [m_s, m_s]))
            y_edge = np.concatenate((y_ref[key][inside], [y_ref[key][inside][-1], 0.0]))
            ax.plot(edge * 1e3, y_edge * unit, color=INK, lw=1.2, label="Analytic" if j == 0 else None, zorder=4)
            keep = (mc > max(x_lo * 1e-3, 1.02 * m_f if region == "shock" else 0)) & (mc < 0.99 * m_s)
            vals = np.concatenate((run[key][k][keep], y_ref[key][(m_ref > x_lo * 1e-3) & (m_ref < 0.99 * m_s)])) * unit
            vals = vals[np.isfinite(vals)]
            lims[key] = [min(lims[key][0], vals.min()), max(lims[key][1], vals.max())]
    for ax, (key, ylabel, title, _) in zip(axes.flat, panels):
        lo, hi = lims[key]
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xscale("log")
        ax.set_xlim(x_lo, 330.0)
        ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        hf._style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [i for i, lab in enumerate(labels) if lab != "Analytic"] + [labels.index("Analytic")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.06, 0.945), ncol=5, frameon=False, fontsize=10.5)
    start = "cold start (t = 0)" if t3.RUNS[name][0] == "cold" else "analytic start at 0.1 ns"
    where = "shocked region" if region == "shock" else "full domain"
    fig.suptitle(rf"Test 3: Au ($\omega=+0.5$) · {t3.RUN_TEXT[name]} · {start} · {where}",
                 x=0.06, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- ratios to the solver
def plot_ratios(start, path):
    import matplotlib.pyplot as plt

    s = h.solver()
    rows = (("rho", r"$\rho\,/\,\rho_{\rm solver}$"), ("p", r"$P\,/\,P_{\rm solver}$"),
            ("u", r"$u\,/\,u_{\rm solver}$"), ("S", r"entropy $P/\rho^\gamma$ [cgs]"))
    fig, axes = plt.subplots(len(rows), len(h.FIGURE_NS), figsize=(16, 13), sharex=True, sharey="row")
    for j, t_ns in enumerate(h.FIGURE_NS):
        curves = _curves(s, start, t_ns * 1e-9)
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            if key == "S":                                  # the entropy itself, against the solver's
                mc = curves[-1][0]
                ax.loglog(mc * 1e3, hf._reference(s, mc, t_ns * 1e-9)[0]["entropy"], color=INK, lw=1.3,
                          label="Analytic (shock solver)", zorder=5)
            else:
                ax.axhline(1.0, color=INK, lw=1.0, zorder=1)
            for mc, r, _, color, text, lw in curves:
                ax.semilogx(mc * 1e3, r[key], color=color, lw=lw, label=text, zorder=3 if lw > 1.5 else 2)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"{t_ns:g} ns", loc="left", fontweight="bold")
            if i == len(rows) - 1:
                ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
            ax.set_xlim(1.0, 330.0)
            hf._style(ax)
            if key != "S":
                ax.ticklabel_format(axis="y", useOffset=False)
    wide = start == "cold"
    axes[0, 0].set_ylim((0.3, 1.2) if wide else (0.9, 1.06))
    axes[1, 0].set_ylim(0.92, 1.08)
    axes[2, 0].set_ylim(0.96, 1.04)
    axes[3, 0].set_ylim(6e5, 9e6)
    handles, labels = axes[3, 0].get_legend_handles_labels()
    handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.05, 0.955), ncol=3, frameon=False, fontsize=10.5)
    start_text = "cold start (t = 0)" if wide else "analytic start at 0.1 ns"
    fig.suptitle(rf"Test 3: Au ($\omega=+0.5$) · shocked region: density, pressure, velocity over the solver's, and the entropy · {start_text}",
                 x=0.05, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- entropy
def plot_entropy(path):
    """The entropy function p / rho^gamma of the shocked matter against the solver's, and the
    change of each element's value between 1 and 2 ns (zero for adiabatic flow)."""
    import matplotlib.pyplot as plt

    s = h.solver()
    t1, t2 = 1e-9, 2e-9
    g = h.GAMMA
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey="row")
    for j, start in enumerate(("cold", "seeded")):
        m_f = float(s.heat_solver.ablated_mass(time=t2))
        m_s1 = float(np.ravel(s.shock_solver.shocked_mass(time=t1))[0])
        entries = []                                       # (runs, colour, text, lw)
        for stem, tags, text, color in OLD[start]:
            entries.append(([_old(stem, tag) for tag in tags], color, text, 1.0))
        if _imc_runs(start):
            entries.append((_imc_runs(start), IMC_COLOR, _imc_text(start), 1.2))
        for n in _new(start):
            entries.append(([_load(n)], COLORS[n], t3.RUN_TEXT[n], 1.7))
        for runs, color, text, lw in entries:
            k1, k2 = hf._frame(runs[0], t1), hf._frame(runs[0], t2)
            mc = hf._mc(runs[0]["m"][k2])
            A1, A2 = (np.mean([r["p"][k] / r["rho"][k]**g for r in runs], axis=0) for k in (k1, k2))
            _, ratio, _ = _mean_ratios(s, runs, t2, 1.05 * m_f)
            axes[0, j].loglog(mc * 1e3, ratio["S"], color=color, lw=lw, label=text)
            keep = (mc > 1.05 * m_f) & (mc < 0.97 * m_s1)
            axes[1, j].semilogx(mc * 1e3, np.where(keep, 100.0 * (A2 / A1 - 1.0), np.nan), color=color, lw=lw)
        mc = hf._mc(entries[-1][0][0]["m"][hf._frame(entries[-1][0][0], t2)])
        ref, _ = hf._reference(s, mc, t2)
        axes[0, j].loglog(mc * 1e3, ref["entropy"], color=INK, lw=1.3, label="Analytic (shock solver)", zorder=5)
        axes[0, j].set_title(("Cold start (t = 0)" if start == "cold" else "Analytic start at 0.1 ns")
                             + r" · entropy $P/\rho^\gamma$ at 2 ns", loc="left", fontweight="bold")
        axes[1, j].axhline(0.0, color=INK, lw=1.0)
        axes[1, j].set_title("change of each element's entropy between 1 and 2 ns", loc="left", fontweight="bold")
        axes[1, j].set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
        for ax in axes[:, j]:
            ax.set_xlim(1.0, 330.0)
            hf._style(ax)
        axes[0, j].legend(frameon=False, fontsize=9.5, loc="lower right")
    axes[0, 0].set_ylabel(r"$P/\rho^\gamma$ [cgs]")
    axes[0, 0].set_ylim(6e5, 9e6)
    axes[1, 0].set_ylabel(r"$(P/\rho^\gamma)_{2\,{\rm ns}}\,/\,(P/\rho^\gamma)_{1\,{\rm ns}}-1$ [%]")
    axes[1, 0].set_ylim(-1.0, 3.0)
    fig.suptitle(r"Test 3: Au ($\omega=+0.5$) · entropy of the shocked region", x=0.05, ha="left",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- the coupling factor chi
CHI_COLORS = {1.0: "#2f9e8f", 10.0: "#b8860b", 100.0: "#8e44ad", 1000.0: "#12305e"}


def _chi_runs():
    """[(chi, run name)] of the 2T cold starts at 30 K with data, chi = 1 first."""
    runs = [(1.0, "cold_2T_30K")] + sorted((chi, n) for n, chi in t3.CHI.items())
    return [(chi, n) for chi, n in runs if (t3.OUT / f"{n}.npz").exists()]


def plot_chi(path, t=2e-9):
    """The 2T cold start against the coupling factor chi, with the 1T run and the solver:
    density and velocity over the solver's, the entropy itself, and the convergence in chi."""
    import matplotlib.pyplot as plt

    s = h.solver()
    m_lo = 1.05 * float(s.heat_solver.ablated_mass(time=t))
    curves = [(f"2T, $\\chi$ = {chi:g}", CHI_COLORS[chi], 1.5, _mean_ratios(s, [_load(n)], t, m_lo)) for chi, n in _chi_runs()]
    one = _mean_ratios(s, [_load("cold_1T_30K")], t, m_lo)
    n_chi = len(curves)
    fine = None
    if (t3.OUT / f"{t3.CHI100_FINE}.npz").exists():
        fine = _mean_ratios(s, [_load(t3.CHI100_FINE)], t, m_lo)
        curves.append((r"2T, $\chi$ = 100, finer surface mesh", "#000000", 1.5, fine))
    curves.append(("1T conduction", "#c0392b", 1.5, one))
    if _imc_runs("cold"):
        curves.append((f"IMC ({len(_imc_runs('cold'))} seeds)", ORANGE, 1.1, _mean_ratios(s, _imc_runs("cold"), t, m_lo)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    for ax, key, ylabel in ((axes[0, 0], "rho", r"$\rho\,/\,\rho_{\rm solver}$"), (axes[0, 1], "u", r"$u\,/\,u_{\rm solver}$"),
                            (axes[1, 0], "S", r"entropy $P/\rho^\gamma$ [cgs]")):
        if key == "S":
            mc = one[0]
            ax.loglog(mc * 1e3, hf._reference(s, mc, t)[0]["entropy"], color=INK, lw=1.3, label="Analytic (shock solver)", zorder=5)
        else:
            ax.axhline(1.0, color=INK, lw=1.0)
        for text, color, lw, (mc, r, _) in curves:
            ax.semilogx(mc * 1e3, r[key], color=color, lw=lw, label=text, ls="--" if "finer" in text else "-")
        ax.set_xlim(1.0, 330.0)
        ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
        ax.set_ylabel(ylabel)
        hf._style(ax)
    axes[0, 0].set_ylim(0.9, 1.3)
    axes[0, 1].set_ylim(0.975, 1.01)
    axes[0, 1].ticklabel_format(axis="y", useOffset=False)
    axes[1, 0].set_ylim(6e5, 9e6)
    axes[1, 0].legend(frameon=False, fontsize=10, loc="lower right")
    axes[0, 0].set_title("Density over the solver's", loc="left", fontweight="bold")
    axes[0, 1].set_title("Velocity over the solver's", loc="left", fontweight="bold")
    axes[1, 0].set_title("Entropy", loc="left", fontweight="bold")
    # convergence: entropy over the solver's at fixed masses, and the shock mass, against chi
    ax = axes[1, 1]
    masses = (5e-3, 12e-3, 30e-3, 100e-3)
    shades = ("#6ea8e5", "#2f6fc0", "#12305e", "#000000")
    chis = np.array([chi for chi, _ in _chi_runs()])
    data = [c[3] for c in curves[:n_chi]]
    for m, color in zip(masses, shades):
        vals = [np.interp(m, mc[np.isfinite(r["A"])], r["A"][np.isfinite(r["A"])]) for mc, r, _ in data]
        ok = np.isfinite(one[1]["A"])
        ax.semilogx(chis, vals, color=color, lw=1.5, marker="o", ms=5, label=f"entropy at {m * 1e3:g} mg/cm$^2$")
        ax.plot([chis[-1] * 3], [np.interp(m, one[0][ok], one[1]["A"][ok])], color=color, marker="s", ms=7, ls="none")
    ax.semilogx(chis, [c[2] for c in data], color=ORANGE, lw=1.5, marker="o", ms=5, label="shock mass")
    ax.plot([chis[-1] * 3], [one[2]], color=ORANGE, marker="s", ms=7, ls="none")
    ax.axhline(1.0, color=INK, lw=1.0)
    ax.set_xlim(chis[0] / 1.6, chis[-1] * 5)
    ax.set_xticks(list(chis) + [chis[-1] * 3])
    ax.set_xticklabels([f"{c:g}" for c in chis] + ["1T"])
    ax.minorticks_off()
    ax.set_xlabel(r"coupling factor $\chi$   (squares: the 1T run)")
    ax.set_ylabel("over the solver's")
    ax.set_title(r"Convergence in $\chi$", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    hf._style(ax)
    fig.suptitle(rf"Test 3: Au ($\omega=+0.5$) · cold start, new mesh, 30 K · 2T diffusion against the coupling factor $\chi$, 1T and IMC · {t * 1e9:g} ns",
                 x=0.05, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chi_table(t=2e-9) -> str:
    s = h.solver()
    m_lo = 1.05 * float(s.heat_solver.ablated_mass(time=t))
    rows = [(f"2T, χ = {chi:g}", n) for chi, n in _chi_runs()] + [("1T conduction", "cold_1T_30K")]
    if (t3.OUT / f"{t3.CHI100_FINE}.npz").exists():
        rows.insert(-1, ("2T, χ = 100, finer surface mesh", t3.CHI100_FINE))
    rows += [(t3.RUN_TEXT[n].split(",")[0], n) for n in t3.IMC_COLD if (t3.OUT / f"{n}.npz").exists()]
    lines = [f"**Cold start, new mesh, 30 K, {t * 1e9:g} ns: (p/ρ^γ) / solver's**", "",
             "| m [mg/cm²] | " + " | ".join(f"{m * 1e3:g}" for m in TABLE_MASSES)
             + " | shock mass | heat front mass | velocity (median) | steps | wall [min] |",
             "|---" * (TABLE_MASSES.size + 6) + "|"]
    for text, n in rows:
        run = _load(n)
        mc, r, shock = _mean_ratios(s, [run], t, m_lo)
        ok = np.isfinite(r["A"])
        keep = ok & (mc > 2e-3)
        lines.append(f"| {text} | " + " | ".join(f"{v:.3f}" for v in np.interp(TABLE_MASSES, mc[ok], r["A"][ok]))
                     + f" | {shock:.3f} | {run['m_heat'][-1] / float(s.heat_solver.ablated_mass(time=t)):.3f} | "
                     f"{np.nanmedian(r['u'][keep]):.4f} | {int(run['steps'])} | {float(run['wall']) / 60:.0f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- table
def table(t=2e-9) -> str:
    s = h.solver()
    p0, c3 = h.solver_pressure_law(s)
    lines = []
    for start, start_text in (("cold", "Cold start (t = 0)"), ("seeded", "Analytic start at 0.1 ns")):
        curves = _curves(s, start, t)
        for key, text in (("rho", "ρ / ρ_solver"), ("A", "(p/ρ^γ) / solver's"), ("p", "p / p_solver"), ("u", "u / u_solver")):
            lines += [f"**{start_text}, {t * 1e9:g} ns: {text}**", "",
                      "| m [mg/cm²] | " + " | ".join(f"{m * 1e3:g}" for m in TABLE_MASSES) + " | shock mass / solver's |",
                      "|---" * (TABLE_MASSES.size + 2) + "|"]
            for mc, r, shock, _, name, _ in curves:
                ok = np.isfinite(r[key])
                lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in np.interp(TABLE_MASSES, mc[ok], r[key][ok]))
                             + f" | {shock:.3f} |")
            lines.append("")
        lines += [f"**{start_text}, {t * 1e9:g} ns: the heat front and the run**", "",
                  "| run | heat front mass / solver's | front pressure (dense side) / solver's | cells | steps | wall [min] |",
                  "|---|---|---|---|---|---|"]
        extra = [*t3.IMC, t3.IMC_X4] if start == "seeded" else [t3.CHI100_FINE, *t3.IMC_COLD]
        for n in _new(start) + [n for n in extra if (t3.OUT / f"{n}.npz").exists()]:
            r = _load(n)
            m_f = float(s.heat_solver.ablated_mass(time=t))
            lines.append(f"| {t3.RUN_TEXT[n]} | {r['m_heat'][-1] / m_f:.3f} | {r['p_dense'][-1] / (p0 * t**c3):.3f} | "
                         f"{r['m'].shape[1]} | {int(r['steps'])} | {float(r['wall']) / 60:.0f} |")
        lines.append("")
    return "\n".join(lines)


def write_table():
    t3.FIGURES.mkdir(parents=True, exist_ok=True)
    text = table()
    (t3.FIGURES / "table.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def all_figures():
    import matplotlib
    matplotlib.use("Agg")

    t3.FIGURES.mkdir(parents=True, exist_ok=True)
    plot_startup(t3.FIGURES / "startup.png")
    for name in t3.RUNS:
        if name in (*t3.IMC, *t3.IMC_COLD) and not name.endswith("seed1"):
            continue                                   # one IMC seed's profiles
        if (t3.OUT / f"{name}.npz").exists():
            plot_profiles(name, t3.FIGURES / f"profiles_{name}.png")
            plot_profiles(name, t3.FIGURES / f"full_{name}.png", region="full")
    for start in ("cold", "seeded"):
        plot_ratios(start, t3.FIGURES / f"ratio_{start}.png")
    plot_entropy(t3.FIGURES / "entropy.png")
    if len(_chi_runs()) > 1:
        plot_chi(t3.FIGURES / "chi_convergence.png")
        (t3.FIGURES / "chi_table.md").write_text(chi_table() + "\n", encoding="utf-8")
    print(f"  wrote {len(list(t3.FIGURES.glob('*.png')))} figures in {t3.FIGURES.relative_to(ivd.REPO)}")

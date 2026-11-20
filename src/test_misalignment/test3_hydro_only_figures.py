"""Figures and table of the hydro-only runs of Test 3 (test3_hydro_only.py).

- drives.png: the pressure at the heat front of the rad-hydro runs against the solver's law,
  and the solver's piston velocity.
- profiles_<run>.png: density, pressure, velocity and temperature of one hydro-only run in the
  shocked region at 1, 1.5 and 2 ns, with the solver and the matching rad-hydro run.
- ratio_cold.png, ratio_seeded.png: every run of that start divided by the solver, at each time.
- entropy.png: p / rho^gamma against mass, its ratio to the solver's, and its change in time.
- table.md: the ratios at 2 ns.
"""
from __future__ import annotations

import numpy as np

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from test_misalignment import test3_hydro_only as h

K_PER_HEV = ivd.K_PER_HEV
BLUES = ("#6ea8e5", "#2f6fc0", "#12305e")           # 1, 1.5, 2 ns, as the repo's profile figures
INK, ORANGE = "#1a1a1a", "#e8683a"
RUN_COLORS = {"p_solver": "#12305e", "p_diffusion": "#2f9e8f", "p_imc": "#b8860b", "p_diffusion_raw": "#9aa0a6", "piston": "#8e44ad"}
TABLE_MASSES = np.array([2, 3, 5, 8, 12, 20, 30, 50, 100, 150, 200]) * 1e-3


def _mc(m):
    return m - 0.5 * np.diff(m, prepend=0.0)


def _frame(run, t):
    k = int(np.argmin(np.abs(run["t"] - t)))
    assert abs(run["t"][k] - t) < 0.02 * t, (run["t"], t)
    return k


def _reference(s, mc, t):
    """The PistonShock solution at the cell centres (zeros past the shock replaced by NaN)."""
    ps = s.shock_solver
    m_s = float(np.ravel(ps.shocked_mass(time=t))[0])
    sol = ps.solve(mass=mc, time=t)
    out = {k: np.where(mc < m_s, sol[k], np.nan) for k in ("density", "pressure", "velocity")}
    out["entropy"] = out["pressure"] / out["density"]**h.GAMMA
    return out, m_s


def _ratios(s, run, t, m_lo=0.0):
    """rho, p, u, entropy of a run over the solver's, between m_lo and just behind the nearer shock."""
    k = _frame(run, t)
    mc = _mc(run["m"][k])
    ref, m_s = _reference(s, mc, float(run["t"][k]))
    excess = run["rho"][k] / ivd._initial_density(run["m"][k], _case()) - 1.0
    m_s_run = mc[int(np.flatnonzero(excess >= 0.5 * np.nanmax(excess))[-1])]
    keep = (mc > m_lo) & (mc < 0.985 * min(m_s, m_s_run))
    A = run["p"][k] / run["rho"][k]**h.GAMMA
    r = dict(rho=run["rho"][k] / ref["density"], p=run["p"][k] / ref["pressure"],
             u=run["u"][k] / ref["velocity"], A=A / ref["entropy"])
    return mc, {key: np.where(keep, val, np.nan) for key, val in r.items()}, m_s_run / m_s


_CASE = []


def _case():
    if not _CASE:
        _CASE.append(ivd._hydro_case(ivd.HYDRO_CASES[h.LABEL]))
    return _CASE[0]


def _rad_hydro(start, tag):
    label = h.LABEL if start == "cold" else h.SEED_LABEL
    return np.load(ivd.DEFAULT_OUT / f"{label}_{tag}.npz")


def _rad_hydro_mean(start, tags, key, k):
    return np.mean([_rad_hydro(start, tag)[key][k] for tag in tags], axis=0)


def _style(ax):
    ivd.style_axes(ax)


# --------------------------------------------------------------------------- drives
def plot_drives(path):
    import matplotlib.pyplot as plt

    s = h.solver()
    p0, c3 = h.solver_pressure_law(s)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for start, tags, color, text in (("cold", ["diffusion"], BLUES[1], "diffusion, cold start"),
                                     ("cold", ["imc1", "imc2", "imc3"], ORANGE, "IMC (3 seeds), cold start"),
                                     ("seeded", ["diffusion"], BLUES[2], "diffusion, analytic start"),
                                     ("seeded", ["imc1", "imc2", "imc3"], "#b8860b", "IMC (3 seeds), analytic start")):
        label = h.LABEL if start == "cold" else h.SEED_LABEL
        t, p = h.front_pressure_series(label, tags)
        ax.semilogx(t * 1e9, p / (p0 * t**c3), color=color, lw=1.4, label=text + r", at $T=0.3\,T_s$")
        if tags == ["diffusion"]:
            kappa = h.front_pressure_factor(label, tags)
            ax.semilogx(t * 1e9, kappa * p / (p0 * t**c3), color=color, lw=1.4, ls="--",
                        label=text + rf", dense side ($\times${kappa:.3f})")
    ax.axhline(1.0, color=INK, lw=1.2, label=r"solver, $p_f = p_0 t^{-0.448}$")
    ax.set_xlabel("t [ns]")
    ax.set_ylabel(r"$p_f$ / solver's $p_f$")
    ax.set_title("Pressure at the heat front", loc="left", fontweight="bold")
    ax.set_ylim(0.75, 1.8)
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    t = np.logspace(-15, np.log10(2e-9), 300)
    ps = s.shock_solver
    ax.semilogx(t * 1e9, [ps.piston_velocity(time=x) * 1e-5 for x in t], color=INK, lw=1.4,
                label="solver's piston (the surface m = 0)")
    for name, color in (("cold_p_solver", RUN_COLORS["p_solver"]), ("cold_p_diffusion", RUN_COLORS["p_diffusion"])):
        f = h.OUT / f"{name}.npz"
        if f.exists():
            run = np.load(f)
            ax.semilogx(run["series_t"][1:] * 1e9, run["u_left"][1:] * 1e-5, color=color, lw=1.0,
                        label=f"surface of the run: {h.RUN_TEXT[name]}")
    ax.set_xlabel("t [ns]")
    ax.set_ylabel(r"$u$ [km/s]")
    ax.set_title("Velocity of the driven surface", loc="left", fontweight="bold")
    ax.set_xlim(1e-6, 3.0)
    ax.set_ylim(0.12, 0.26)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for ax in axes:
        _style(ax)
    fig.suptitle(r"Test 3: Au ($\omega=+0.5$) · the drives of the hydro-only runs", x=0.06, ha="left",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- profiles of one run
def plot_profiles(name, path):
    import matplotlib.pyplot as plt

    start, drive = h.RUNS[name]
    run = np.load(h.OUT / f"{name}.npz")
    s = h.solver()
    tags = ["imc1", "imc2", "imc3"] if drive == "p_imc" else ["diffusion"]
    rh = _rad_hydro(start, tags[0])
    rh_text = "IMC rad-hydro (3 seeds)" if drive == "p_imc" else "diffusion rad-hydro"
    case = _case()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4))
    panels = (("rho", r"$\rho$ [g/cm$^3$]", "Density", 1.0), ("p", r"$P$ [Mbar]", "Pressure", 1e-12),
              ("u", r"$u$ [km/s]", "Velocity", 1e-5), ("T_material", r"$T$ [HeV]", "Temperature", 1 / K_PER_HEV))
    lims = {key: [np.inf, -np.inf] for key, *_ in panels}
    for j, t_ns in enumerate(h.FIGURE_NS):
        t = t_ns * 1e-9
        k, kr = _frame(run, t), _frame(rh, t)
        mc = _mc(run["m"][k])
        ref, m_s = _reference(s, mc, t)
        ref["T"] = (ref["pressure"] * ref["density"]**(case.mu - 1.0) / (case.r * case.f_Kelvin))**(1.0 / case.beta_Rosen)
        m_f = float(s.heat_solver.ablated_mass(time=t))
        mcr = _mc(rh["m"][kr])
        beyond = mcr > 1.05 * m_f                                   # the rad-hydro run's shocked matter
        for ax, (key, _, _, unit) in zip(axes.flat, panels):
            y_rh = _rad_hydro_mean(start, tags, key, kr)
            ax.plot(mcr[beyond] * 1e3, y_rh[beyond] * unit, color=ORANGE, lw=1.0, alpha=0.9,
                    label=rh_text if j == 0 else None, zorder=2)
            ax.plot(mc * 1e3, run[key][k] * unit, color=BLUES[j], lw=1.6, label=f"Hydro only, {t_ns:g} ns", zorder=3)
            y_ref = {"rho": ref["density"], "p": ref["pressure"], "u": ref["velocity"], "T_material": ref["T"]}[key]
            edge = np.concatenate((mc[mc < m_s], [m_s, m_s]))
            y_edge = np.concatenate((y_ref[mc < m_s], [y_ref[mc < m_s][-1], 0.0]))
            ax.plot(edge * 1e3, y_edge * unit, color=INK, lw=1.3, label="Analytic (shock solver)" if j == 0 else None,
                    zorder=4)
            inside = (mc > 2e-3) & (mc < 0.99 * m_s)
            vals = np.concatenate((run[key][k][inside], y_ref[inside], y_rh[(mcr > 2e-3) & (mcr < 0.99 * m_s)])) * unit
            lims[key] = [min(lims[key][0], np.nanmin(vals)), max(lims[key][1], np.nanmax(vals))]
    for ax, (key, ylabel, title, _) in zip(axes.flat, panels):
        lo, hi = lims[key]
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xscale("log")
        ax.set_xlim(1.0, 330.0)
        ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [i for i, lab in enumerate(labels) if lab.startswith("Hydro")] + \
            [i for i, lab in enumerate(labels) if not lab.startswith("Hydro")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.06, 0.945), ncol=5, frameon=False, fontsize=10.5)
    start_text = "cold start" if start == "cold" else "analytic start at 0.1 ns"
    fig.suptitle(rf"Test 3: Au ($\omega=+0.5$) · {h.RUN_TEXT[name]} · {start_text} · shocked region",
                 x=0.06, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- ratios to the solver
def plot_ratios(start, path):
    import matplotlib.pyplot as plt

    s = h.solver()
    names = [n for n, (st, _) in h.RUNS.items() if st == start and (h.OUT / f"{n}.npz").exists()]
    rows = (("rho", r"$\rho\,/\,\rho_{\rm solver}$"), ("p", r"$P\,/\,P_{\rm solver}$"),
            ("u", r"$u\,/\,u_{\rm solver}$"), ("A", r"entropy $P/\rho^\gamma$ [cgs]"))
    fig, axes = plt.subplots(len(rows), len(h.FIGURE_NS), figsize=(16, 13), sharex=True, sharey="row")
    for j, t_ns in enumerate(h.FIGURE_NS):
        t = t_ns * 1e-9
        m_f = float(s.heat_solver.ablated_mass(time=t))
        curves = []
        for tags, color, text in ((["diffusion"], BLUES[1], "diffusion rad-hydro"),
                                  (["imc1", "imc2", "imc3"], ORANGE, "IMC rad-hydro (3 seeds)")):
            runs = [_rad_hydro(start, tag) for tag in tags]
            parts = [_ratios(s, r, t, m_lo=1.05 * m_f) for r in runs]
            mean = {key: np.mean([p[1][key] for p in parts], axis=0) for key, _ in rows}
            curves.append((parts[0][0], mean, color, text, 1.0, 2))
        for n in names:
            mc, r, _ = _ratios(s, np.load(h.OUT / f"{n}.npz"), t, m_lo=1e-3)
            color = "#c0392b" if n in h.T_INITIAL else RUN_COLORS[h.RUNS[n][1]]
            curves.append((mc, r, color, h.RUN_TEXT[n], 1.7, 3))
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            if key == "A":                    # the entropy itself (ratio x the solver's), against the solver's
                for mc, r, color, text, lw, z in curves:
                    ax.loglog(mc * 1e3, r[key] * _reference(s, mc, t)[0]["entropy"], color=color, lw=lw, label=text, zorder=z)
                mc = curves[-1][0]
                ax.loglog(mc * 1e3, _reference(s, mc, t)[0]["entropy"], color=INK, lw=1.3,
                          label="Analytic (shock solver)", zorder=5)
            else:
                ax.axhline(1.0, color=INK, lw=1.0, zorder=1)
                for mc, r, color, text, lw, z in curves:
                    ax.semilogx(mc * 1e3, r[key], color=color, lw=lw, label=text, zorder=z)
                ax.ticklabel_format(axis="y", useOffset=False)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"{t_ns:g} ns", loc="left", fontweight="bold")
            if i == len(rows) - 1:
                ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
            ax.set_xlim(1.0, 330.0)
            _style(ax)
    wide = start == "cold"
    axes[0, 0].set_ylim((0.3, 1.15) if wide else (0.88, 1.06))
    axes[1, 0].set_ylim(0.9, 1.1)
    axes[2, 0].set_ylim(0.96, 1.04)
    axes[3, 0].set_ylim(6e5, 9e6)
    handles, labels = axes[3, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.05, 0.955), ncol=3, frameon=False, fontsize=10.5)
    start_text = "cold start (t = 0)" if start == "cold" else "analytic start at 0.1 ns"
    fig.suptitle(rf"Test 3: Au ($\omega=+0.5$) · shocked region: density, pressure, velocity over the solver's, and the entropy · {start_text}",
                 x=0.05, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- table
def table(t=2e-9) -> str:
    s = h.solver()
    m_f = float(s.heat_solver.ablated_mass(time=t))
    lines = []
    for start, start_text in (("cold", "Cold start (t = 0)"), ("seeded", "Analytic start at 0.1 ns")):
        entries = []
        for tags, text in ((["diffusion"], "diffusion rad-hydro"), (["imc1", "imc2", "imc3"], "IMC rad-hydro (3 seeds)")):
            parts = [_ratios(s, _rad_hydro(start, tag), t, m_lo=1.05 * m_f) for tag in tags]
            entries.append((text, parts[0][0], {k: np.mean([p[1][k] for p in parts], axis=0) for k in ("rho", "p", "u", "A")},
                            np.mean([p[2] for p in parts]), ""))
        for n, (st, _) in h.RUNS.items():
            f = h.OUT / f"{n}.npz"
            if st == start and f.exists():
                run = np.load(f)
                mc, r, shock = _ratios(s, run, t, m_lo=1e-3)
                err = float(run["max_energy_error"])
                entries.append((h.RUN_TEXT[n].replace("$", "").replace("_f", "_f").replace("_p", "_p"), mc, r, shock,
                                f"{int(run['steps'])} steps" + ("" if np.isnan(err) else f", energy error {err:.0e}")))
        for key, text in (("rho", "ρ / ρ_solver"), ("A", "(p/ρ^γ) / solver's"), ("p", "p / p_solver"), ("u", "u / u_solver")):
            lines += [f"**{start_text}, {t * 1e9:g} ns: {text}**", "",
                      "| m [mg/cm²] | " + " | ".join(f"{m * 1e3:g}" for m in TABLE_MASSES) + " | shock mass / solver's |",
                      "|---" * (TABLE_MASSES.size + 2) + "|"]
            for name, mc, r, shock, _ in entries:
                vals = np.interp(TABLE_MASSES, mc[np.isfinite(r[key])], r[key][np.isfinite(r[key])])
                lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {shock:.3f} |")
            lines.append("")
        lines += [f"- {name}: {note}" for name, _, _, _, note in entries if note] + [""]
    return "\n".join(lines)


def write_table():
    h.FIGURES.mkdir(parents=True, exist_ok=True)
    text = table()
    (h.FIGURES / "table.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def all_figures():
    import matplotlib
    matplotlib.use("Agg")

    h.FIGURES.mkdir(parents=True, exist_ok=True)
    plot_drives(h.FIGURES / "drives.png")
    print("  wrote drives.png")
    for name in h.RUNS:
        if (h.OUT / f"{name}.npz").exists():
            plot_profiles(name, h.FIGURES / f"profiles_{name}.png")
            print(f"  wrote profiles_{name}.png")
    for start in ("cold", "seeded"):
        plot_ratios(start, h.FIGURES / f"ratio_{start}.png")
        print(f"  wrote ratio_{start}.png")
    plot_entropy(h.FIGURES / "entropy.png")
    print("  wrote entropy.png")


# --------------------------------------------------------------------------- entropy
def plot_entropy(path):
    """The entropy function A = p / rho^gamma against mass: its value, its ratio to the solver's,
    and its change between 1 and 2 ns (zero for adiabatic flow)."""
    import matplotlib.pyplot as plt

    s = h.solver()
    t1, t2 = 1e-9, 2e-9
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey="row")
    shown = {"cold": ("cold_p_solver", "cold_p_diffusion", "cold_piston"),
             "seeded": ("seeded_p_solver", "seeded_p_diffusion", "seeded_p_solver_30K")}
    for j, start in enumerate(("cold", "seeded")):
        m_f = float(s.heat_solver.ablated_mass(time=t2))
        curves = []                                         # (mc, A at t2, A at t1, ratio at t2, color, text, lw)
        for tags, color, text in ((["diffusion"], BLUES[1], "diffusion rad-hydro"),
                                  (["imc1", "imc2", "imc3"], ORANGE, "IMC rad-hydro (3 seeds)")):
            runs = [_rad_hydro(start, tag) for tag in tags]
            k1, k2 = _frame(runs[0], t1), _frame(runs[0], t2)
            mc = _mc(runs[0]["m"][k2])
            A = [np.mean([r["p"][k] / r["rho"][k]**h.GAMMA for r in runs], axis=0) for k in (k1, k2)]
            ratio = np.mean([_ratios(s, r, t2, m_lo=1.05 * m_f)[1]["A"] for r in runs], axis=0)
            curves.append((mc, A[1], A[0], ratio, color, text, 1.1))
        for n in shown[start]:
            run = np.load(h.OUT / f"{n}.npz")
            k1, k2 = _frame(run, t1), _frame(run, t2)
            mc = _mc(run["m"][k2])
            A = [run["p"][k] / run["rho"][k]**h.GAMMA for k in (k1, k2)]
            color = "#c0392b" if n in h.T_INITIAL else RUN_COLORS[h.RUNS[n][1]]
            curves.append((mc, A[1], A[0], _ratios(s, run, t2, m_lo=1e-3)[1]["A"], color, h.RUN_TEXT[n], 1.7))
        ref, m_s2 = _reference(s, curves[0][0], t2)
        m_s1 = float(np.ravel(s.shock_solver.shocked_mass(time=t1))[0])
        ax = axes[0, j]
        for mc, A2, _, ratio, color, text, lw in curves:
            ax.loglog(mc * 1e3, np.where(np.isfinite(ratio), A2, np.nan), color=color, lw=lw, label=text)
        ax.loglog(curves[0][0] * 1e3, ref["entropy"], color=INK, lw=1.3, label="Analytic (shock solver)", zorder=5)
        ax.set_title(("Cold start (t = 0)" if start == "cold" else "Analytic start at 0.1 ns") + r" · entropy $P/\rho^\gamma$ at 2 ns",
                     loc="left", fontweight="bold")
        ax = axes[1, j]
        ax.axhline(0.0, color=INK, lw=1.0)
        for mc, A2, A1, _, color, text, lw in curves:
            lo = 1.05 * m_f if "rad-hydro" in text else 1e-3
            keep = (mc > lo) & (mc < 0.97 * m_s1)           # matter already shocked at 1 ns
            ax.semilogx(mc * 1e3, np.where(keep, 100.0 * (A2 / A1 - 1.0), np.nan), color=color, lw=lw)
        ax.set_title("change of each element's entropy between 1 and 2 ns", loc="left", fontweight="bold")
        ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
        for ax in axes[:, j]:
            ax.set_xlim(1.0, 330.0)
            _style(ax)
    axes[0, 0].set_ylabel(r"$P/\rho^\gamma$ [cgs]")
    axes[1, 0].set_ylabel(r"$(P/\rho^\gamma)_{2\,{\rm ns}}\,/\,(P/\rho^\gamma)_{1\,{\rm ns}}-1$ [%]")
    axes[1, 0].set_ylim(-1.0, 3.0)
    axes[0, 0].set_ylim(6e5, 9e6)
    for j in (0, 1):
        axes[0, j].legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.suptitle(r"Test 3: Au ($\omega=+0.5$) · entropy of the shocked region", x=0.05, ha="left",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)

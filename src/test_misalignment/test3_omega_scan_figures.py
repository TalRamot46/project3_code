"""Figures and table of test3_omega_scan.py (md/misalignment/03_startup.md section 5).

- entropy_by_omega.png: the entropy p / rho^gamma of IMC and of 2T diffusion against the solver's,
  one panel per omega, at 1 ns.
- convergence.png: the same entropies over the solver's, all omega on one axis: against the
  fraction of the shocked mass, against the time the shock passed the element, and the depth of
  the deficit against omega.
- layer.png: what the ablated layer does against time for all omega (front pressure and heat
  front over the solver's, surface matter temperature) and its temperatures at 1 ns.
- table.md: the numbers.
"""
from __future__ import annotations

import numpy as np

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from test_misalignment import test3_hydro_only as h
from test_misalignment import test3_hydro_only_figures as hf
from test_misalignment import test3_omega_scan as o

INK = hf.INK
SOLVER, IMC, TWO_T = INK, "#d9822b", "#2f9e8f"
COLORS = {0.5: "#12305e", 0.4: "#2f6fc0", 0.3: "#2f9e8f", 0.2: "#7aa33a", 0.1: "#d9822b", 0.05: "#c0392b"}
TEXT = {"imc": "IMC (mean of 3 seeds)", "2T": r"2T diffusion, $\chi=1$"}
SHORT = {"imc": "IMC", "2T": r"2T, $\chi=1$"}
T_NS = o.T_END * 1e9


def shock_exponent(s) -> float:
    """a in m_s ~ t^a."""
    m = [float(np.ravel(s.shock_solver.shocked_mass(time=t))[0]) for t in (0.5e-9, 1e-9)]
    return float(np.log(m[1] / m[0]) / np.log(2.0))


def compare(omega: float, physics: str, t: float = o.T_END):
    """The runs of (omega, physics) at t, averaged over the seeds, against the solver of omega.

    Returns None without runs; else a dict with the cell-centre mass mc, the solver's shocked mass
    m_s and heat front m_f, the entropies A and A_solver, the ratios A_ratio and rho_ratio (NaN
    outside the shocked region common to run and solver), shock = run's shock mass / solver's, and
    the shock arrival time t_shock of every element."""
    runs = o.load(omega, physics)
    if not runs:
        return None
    s, case = o.solver(omega), o.base_case(omega)
    A, rho, shock = [], [], []
    for r in runs:
        k = int(np.argmin(np.abs(r["t"] - t)))
        mc = hf._mc(r["m"][k])
        ref, m_s = hf._reference(s, mc, float(r["t"][k]))
        excess = r["rho"][k] / ivd._initial_density(r["m"][k], case) - 1.0
        shock.append(mc[int(np.flatnonzero(excess >= 0.5 * np.nanmax(excess))[-1])] / m_s)
        A.append(r["p"][k] / r["rho"][k]**o.GAMMA)
        rho.append(r["rho"][k] / ref["density"])
    A, rho = np.mean(A, axis=0), np.mean(rho, axis=0)
    m_f = float(s.heat_solver.ablated_mass(time=t))
    keep = (mc > 2.0 * m_f) & (mc < 0.985 * min(shock) * m_s)
    return dict(mc=mc, m_s=m_s, m_f=m_f, A=A, A_solver=ref["entropy"], keep=keep,
                A_ratio=np.where(keep, A / ref["entropy"], np.nan), rho_ratio=np.where(keep, rho, np.nan),
                shock=float(np.mean(shock)), t_shock=t * (mc / m_s)**(1.0 / shock_exponent(s)), n=len(runs))


def _at_time(c, t_shock):
    """Entropy over the solver's of the element the shock passed at t_shock."""
    ok = np.isfinite(c["A_ratio"])
    return float(np.interp(t_shock, c["t_shock"][ok], c["A_ratio"][ok], left=np.nan, right=np.nan))


# --------------------------------------------------------------------------- one panel per omega
def plot_entropy_by_omega(path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, w in zip(axes.ravel(), o.OMEGAS):
        drawn = False
        for physics, color in (("imc", IMC), ("2T", TWO_T)):
            c = compare(w, physics)
            if c is None:
                continue
            if not drawn:
                ax.loglog(c["mc"] * 1e3, c["A_solver"], color=SOLVER, lw=2.2, label="solver")
                ax.axvline(c["m_f"] * 1e3, color=hf.ORANGE, lw=1.0, ls=":")
                lo, hi = np.nanmin(c["A_solver"][c["keep"]]), np.nanmax(c["A_solver"][c["keep"]])
                ax.set_xlim(0.5 * c["m_f"] * 1e3, 1.05 * c["m_s"] * 1e3)
                ax.set_ylim(lo / 2.5, hi * 2.5)
                drawn = True
            show = c["mc"] > 1.3 * c["m_f"]
            ax.loglog(c["mc"][show] * 1e3, c["A"][show], color=color, lw=1.5, label=TEXT[physics])
        ax.set_title(rf"$\omega={w:+.2f}$", loc="left", fontweight="bold", fontsize=12)
        ax.set_xlabel(r"m [mg/cm$^2$]")
        ax.set_ylabel(r"$p/\rho^{\gamma}$ [cgs]")
        hf._style(ax)
    axes[0, 0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle(rf"Au, $T_s$ = 1 HeV, 30 K, cold start · entropy against the solver's at {T_NS:g} ns "
                 "(dotted: the solver's heat front)", x=0.04, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- all omega together
def plot_convergence(path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for row, physics in enumerate(("imc", "2T")):
        a, b, d = axes[row]
        depth, far, shock = [], [], []
        for w in o.OMEGAS:
            c = compare(w, physics)
            if c is None:
                continue
            a.plot(c["mc"] / c["m_s"], c["A_ratio"], color=COLORS[w], lw=1.5, label=rf"$\omega={w:+.2f}$")
            b.semilogx(c["t_shock"] * 1e9, c["A_ratio"], color=COLORS[w], lw=1.5)
            depth.append((w, np.nanmin(c["A_ratio"])))
            far.append((w, _at_time(c, 0.5 * o.T_END)))
            shock.append((w, c["shock"]))
        for data, marker, text in ((depth, "o", "deepest entropy deficit"),
                                   (far, "s", f"entropy of the element shocked at {0.5 * T_NS:g} ns"),
                                   (shock, "^", "shock mass")):
            if data:
                d.plot(*zip(*data), marker=marker, color=INK, lw=1.2, ms=7, mfc="white" if marker != "o" else INK,
                       label=text)
        a.set_xlabel(r"m / m$_{shock}$ (solver)")
        a.set_xlim(0, 1)
        b.set_xlabel("time the shock passed the element [ns]")
        b.set_xlim(2e-3, T_NS)
        d.set_xlabel(r"$\omega$")
        d.set_xlim(0, 0.55)
        for ax, title in ((a, "entropy / solver's"), (b, "the same, against the time of shocking"),
                          (d, "against the density exponent")):
            ax.axhline(1.0, color=INK, lw=1.0)
            ax.set_ylim(0.7, 1.05)
            ax.set_title(f"{SHORT[physics]} · {title}", loc="left", fontweight="bold", fontsize=10.5)
            ax.set_ylabel("simulation / solver")
            hf._style(ax)
        d.legend(frameon=False, fontsize=9.5, loc="lower left")
    axes[0, 0].legend(frameon=False, fontsize=10, loc="lower right", ncol=2)
    fig.suptitle(rf"Au, $T_s$ = 1 HeV, 30 K, cold start · the cold-start entropy deficit against $\omega$, at {T_NS:g} ns",
                 x=0.04, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- the ablated layer
def plot_layer(path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5))
    for row, physics in enumerate(("imc", "2T")):
        for w in o.OMEGAS:
            runs = o.load(w, physics)
            if not runs:
                continue
            s = o.solver(w)
            p0, c3 = h.solver_pressure_law(s)
            r = runs[0]
            t = r["series_t"]
            ok = np.isfinite(r["p_dense"]) & (t > 0) & (t <= o.T_END * 1.001)
            t = t[ok]
            m_f = np.array([float(s.heat_solver.ablated_mass(time=x)) for x in t])
            m_s = np.array([float(np.ravel(s.shock_solver.shocked_mass(time=x))[0]) for x in t])
            kw = dict(color=COLORS[w], lw=1.4)
            axes[row, 0].semilogx(t * 1e9, r["p_dense"][ok] / (p0 * t**c3), label=rf"$\omega={w:+.2f}$", **kw)
            axes[row, 1].semilogx(t * 1e9, r["m_heat"][ok] / m_f, **kw)
            axes[row, 2].semilogx(t * 1e9, r["T_surface"][ok] / ivd.K_PER_HEV, **kw)
            k = int(np.argmin(np.abs(r["t"] - o.T_END)))
            mc = hf._mc(r["m"][k])
            axes[row, 3].semilogx(mc * 1e3, r["T_material"][k] / ivd.K_PER_HEV, **kw)
            axes[row, 3].semilogx(mc * 1e3, r["T_rad"][k] / ivd.K_PER_HEV, ls="--", **kw)
        for ax, title, ylim in zip(axes[row], ("front pressure (dense side) / solver's",
                                               "heat front mass / solver's", "surface matter temperature [HeV]",
                                               f"T at {T_NS:g} ns [HeV]: matter solid, radiation dashed"),
                                   ((0.6, 1.3), (0.8, 2.0), (0.0, 1.1), (0.0, 1.1))):
            ax.set_ylim(*ylim)
            ax.set_title(f"{SHORT[physics]} · {title}", loc="left", fontweight="bold", fontsize=10)
            hf._style(ax)
        for ax in axes[row, :2]:
            ax.axhline(1.0, color=INK, lw=1.0)
        for ax in axes[row, :3]:
            ax.set_xlim(1e-4, T_NS)
            ax.set_xlabel("t [ns]")
        axes[row, 3].set_xlim(1e-3, 3.0)
        axes[row, 3].set_xlabel(r"m [mg/cm$^2$]")
    axes[0, 0].legend(frameon=False, fontsize=9.5, loc="lower right", ncol=2)
    fig.suptitle(r"Au, $T_s$ = 1 HeV, 30 K, cold start · the ablated layer against $\omega$ (IMC: seed 1)",
                 x=0.04, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- table
def table() -> str:
    times = (0.05, 0.1, 0.2, 0.5, 0.8)
    out = [f"Cold start, 30 K, at {T_NS:g} ns. Entropy p/rho^gamma over the solver's, of the mass element the shock "
           "passed at the given time; IMC is the mean of the seeds.", ""]
    for physics in ("imc", "2T"):
        out += [f"**{TEXT[physics]}**", "",
                "| omega | m_shock [mg/cm2] | m_shock / m_heat | deepest deficit | "
                + " | ".join(f"shocked at {x:g} ns" for x in times) + " | shock mass / solver | steps |",
                "|---|---|---|---|" + "---|" * len(times) + "---|---|"]
        for w in o.OMEGAS:
            c = compare(w, physics)
            if c is None:
                continue
            steps = int(np.mean([int(r["steps"]) for r in o.load(w, physics)]))
            out.append(f"| {w:+.2f} | {c['m_s'] * 1e3:.1f} | {c['m_s'] / c['m_f']:.0f} | {np.nanmin(c['A_ratio']):.3f} | "
                       + " | ".join(f"{_at_time(c, x * 1e-9):.3f}" for x in times)
                       + f" | {c['shock']:.3f} | {steps:,} |")
        out.append("")
    return "\n".join(out)


def all_figures():
    import matplotlib
    matplotlib.use("Agg")
    o.FIGURES.mkdir(parents=True, exist_ok=True)
    plot_entropy_by_omega(o.FIGURES / "entropy_by_omega.png")
    plot_convergence(o.FIGURES / "convergence.png")
    plot_layer(o.FIGURES / "layer.png")
    (o.FIGURES / "table.md").write_text(table() + "\n", encoding="utf-8")
    print(table())
    print(f"  wrote   {o.FIGURES}")

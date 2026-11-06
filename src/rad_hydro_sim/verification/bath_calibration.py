"""The semi-analytic AblationSolver against diffusion and IMC under the three bath drives
(md/phase_c/bath_calibration.md): tables per bath, the full-profile figures per bath, and one
overlay per case comparing the baths.

    uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full test4_full     # tables
    uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full ... --profiles  # figures/<bath>/
    uv run python -m rad_hydro_sim.verification.bath_calibration --cases test1_full ... --overlay   # figures/overlay/
    uv run python -m rad_hydro_sim.verification.bath_calibration --cases --omega-scan=-0.75 --seeds 1

Baths (imc_vs_diffusion --bath): "paper", the manuscript's Marshak relation; "calibrated", the
surface matter temperature matched to the solver's T_s(t); "calibrated_flux", the net energy flux
matched to the solver's s(0, t).  The analytic curve is the same for all three.

The omega scan adds Test 4 twins (gold, 1 HeV, tau = 0) with other omegas on a uniform-mass mesh,
paper bath only.  Attempts at a better patch are legacy (rad_hydro_sim.verification.legacy).
"""
from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from rad_hydro_sim.verification import imc_vs_diffusion as ivd

SOLVERS = ivd.REPO / "results" / "patching_poc"       # pickled AblationSolvers (~20 min each to build)
OUT = ivd.REPO / "results" / "bath_calibration"
SIM_OUT = ivd.DEFAULT_OUT
TIMES = np.array([0.5, 1.0, 1.5, 2.0]) * 1e-9
# simulation runs per bath, and their figure folders
BATHS = {"paper": ("paper_bath", SIM_OUT),
         "calibrated": ("calibrated_T_bath", SIM_OUT.with_name(SIM_OUT.name + "_calibrated")),
         "calibrated_flux": ("calibrated_flux_bath", SIM_OUT.with_name(SIM_OUT.name + "_calibrated_flux"))}
BATH_TEXT = {"paper": "paper bath", "calibrated": "bath calibrated to the surface temperature",
             "calibrated_flux": "bath calibrated to the energy flux"}
BATH_SHORT = {"paper": "paper bath", "calibrated": "T-calibrated bath", "calibrated_flux": "flux-calibrated bath"}
FIG = ivd.REPO / "md" / "phase_c" / "figures"


def omega_label(omega: float) -> str:
    return f"omega{omega:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")


def register_omega_case(omega: float, N: int = 400) -> str:
    """A Test 4 twin with another omega: preset + HydroCase registered at run time."""
    from rad_hydro_sim.problems import presets_config as P

    label = omega_label(omega) + ("" if N == 400 else f"_n{N}")
    key = f"POC_{label.upper()}"
    if label in ivd.HYDRO_CASES:
        return label
    base = P.PRESET_TEST_CASES[P.CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL]
    case = replace(base, omega=omega, title=rf"Au, $\omega={omega:+.2f}$")
    solver = _solver_for_case(case, omega_label(omega))
    m_s = float(np.ravel(solver.shock_solver.shocked_mass(time=base.t_sec_end))[0])
    # a 15% margin in mass beyond the analytic shock at t_end
    x_max = float(solver.shock_solver.initial_position(mass=1.15 * m_s))
    P.PRESET_TEST_CASES[key] = replace(case, x_max=x_max)
    setattr(P, key, key)
    ivd.HYDRO_CASES[label] = ivd.HydroCase(key, rf"Au, $\omega={omega:+.2f}$", N=N, store_every=40,
                                           config=(("mesh", "uniform_mass"),),
                                           imc=ivd.HYDRO_CASES["test4_full"].imc)
    return label


def _solver_for_case(case, label: str):
    from menahem_new.ablation_solver_og import AblationSolver
    from rad_hydro_sim.verification.menahem_comparison import _ablation_kwargs_from_case

    path = SOLVERS / f"solver_{label}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    solver = AblationSolver(**_ablation_kwargs_from_case(case))
    SOLVERS.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(solver, f)
    return solver


def solver_for(label: str):
    # the omega-scan cases at other N share their omega's solver
    return _solver_for_case(ivd._hydro_case(ivd.HYDRO_CASES[label]), label.split("_n")[0])


class Analytic:
    """The AblationSolver with the front values the tables compare (solve() as AblationSolver's)."""

    def __init__(self, label: str):
        self.base = solver_for(label)
        hs = self.base.heat_solver
        self._p0s = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3

    def front_values(self, t):
        hs, ps = self.base.heat_solver, self.base.shock_solver
        return dict(m_heat=float(hs.ablated_mass(time=t)), m_shock=float(np.ravel(ps.shocked_mass(time=t))[0]),
                    p_abl=self._p0s * t**hs.c3, u_front=float(ps.piston_velocity(time=t)))

    def solve(self, *, mass, time):
        return self.base.solve(mass=mass, time=time)


def baths_of(label: str):
    """The baths with a diffusion run of this case."""
    return [b for b, (_, out) in BATHS.items() if (out / f"{label}_diffusion.npz").exists()]


def sim_fronts(label: str, out: Path = SIM_OUT):
    """{run name: {t: front values}} of the cached diffusion / IMC runs."""
    runs = {}
    for tag in ["diffusion", "imc1", "imc2", "imc3"]:
        path = out / f"{label}_{tag}.npz"
        if path.exists():
            run = np.load(path)
            runs[tag] = {t: ivd.front_values(run, label, t) for t in TIMES}
    return runs


def table(label: str, analytic) -> str:
    """Fronts of diffusion (and the IMC mean) under each bath against the analytic solution."""
    baths = baths_of(label)
    sims = {b: sim_fronts(label, BATHS[b][1]) for b in baths}
    rows = [f"### {label}", "", "| t [ns] | quantity | analytic | " + " | ".join(
        f"diffusion, {BATH_SHORT[b]} | IMC, {BATH_SHORT[b]}" for b in baths) + " |",
            "|---|---|---|" + "---|---|" * len(baths)]
    for t in TIMES:
        ref = analytic.front_values(t)
        for q, unit in (("m_heat", 1e3), ("m_shock", 1e3), ("p_abl", 1e-12)):
            cells = [f"{ref[q] * unit:.4g}"]
            for b in baths:
                d = sims[b]["diffusion"][t][q]
                imc = [s[t][q] for k, s in sims[b].items() if k.startswith("imc")]
                cells.append(f"{d * unit:.4g} ({100 * (d / ref[q] - 1):+.1f}%)")
                cells.append(f"{np.mean(imc) * unit:.4g} ({100 * (np.mean(imc) / ref[q] - 1):+.1f}%)" if imc else "-")
            rows.append(f"| {t * 1e9:.1f} | {q} | " + " | ".join(cells) + " |")
    rows += ["", "(m in mg/cm^2, p in Mbar; the % is the simulation vs. the analytic solution)"]
    return "\n".join(rows)


def figure_name(label: str) -> str:
    return ivd.FIGURE_NAMES[label][0] if label in ivd.FIGURE_NAMES else f"{label}_profiles.png"


def plot_profile_set(labels, baths=tuple(BATHS)):
    """md/phase_c/figures/<bath>/<case>_profiles.png for every case with runs in that bath."""
    import matplotlib
    matplotlib.use("Agg")

    for label in labels:
        analytic = None
        for bath in baths:
            folder, out = BATHS[bath]
            if not (out / f"{label}_diffusion.npz").exists():
                continue
            analytic = analytic or Analytic(label)
            path = FIG / folder / figure_name(label)
            path.parent.mkdir(parents=True, exist_ok=True)
            ivd.plot_hydro_profiles(out, label, (1, 2, 3), path, solver=analytic, method=BATH_SHORT[bath])
            print(f"  wrote {path.relative_to(ivd.REPO)}", flush=True)


# overlay: one hue per bath (reference palette, fixed order), ink for the analytic curve
BATH_COLOR = {"paper": "#2a78d6", "calibrated": "#eb6834", "calibrated_flux": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e4e3dd"
FIELDS = (("rho", "density", r"$\rho$ [g/cm$^3$]", 1.0), ("p", "pressure", r"$P$ [Mbar]", 1e-12),
          ("u", "velocity", r"$u$ [km/s]", 1e-5), ("T_material", "temperature", r"$T$ [HeV]", 1.0 / ivd.K_PER_HEV))


def profile_score(run, k, analytic, t):
    """Mass-weighted L1 distance between a run's profile at frame k and an analytic one,
    sum |f_run - f_an| dm / sum |f_an| dm, averaged over rho, p, u, T."""
    m = run["m"][k]
    dm = np.diff(m, prepend=0.0)
    sol = analytic.solve(mass=m - 0.5 * dm, time=t)
    errs = []
    for key, ref, _, _ in FIELDS:
        an = np.nan_to_num(np.asarray(sol[ref], dtype=float))
        errs.append(np.sum(np.abs(run[key][k] - an) * dm) / np.sum(np.abs(an) * dm))
    return float(np.mean(errs))


def _label_short_log_axes(ax):
    """On a log axis spanning under ~1.5 decades matplotlib labels no tick; label the 2, 3, 5 ones."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        if axis.get_scale() == "log" and lo > 0 and hi / lo < 30:
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(which="both", top=False, right=False, colors=MUTED, labelcolor=INK, labelsize=10)


def plot_overlay(label: str, path: Path, t: float = 2e-9, region="full", scale=None):
    """Diffusion under every bath of one case at t against the analytic solution; the bath
    closest to it by profile_score drawn bold, the baths ranked beside the profiles.

    ``region`` / ``scale`` as in imc_vs_diffusion.plot_hydro_profiles: "full", "heat" or "shock";
    "linear", "semilogx", "semilogy" or "loglog" (None: the case's log_mass_axis).  The ranking is always over
    the full domain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    analytic = Analytic(label)
    runs = {b: np.load(BATHS[b][1] / f"{label}_diffusion.npz") for b in baths_of(label)}
    k = {b: int(np.argmin(np.abs(r["t"] - t))) for b, r in runs.items()}
    rows = []
    for b, r in runs.items():
        tb = float(r["t"][k[b]])
        fv, ref = ivd.front_values(r, label, tb), analytic.front_values(tb)
        rows.append(dict(bath=b, score=profile_score(r, k[b], analytic, tb),
                         **{q: 100 * (fv[q] / ref[q] - 1) for q in ("m_heat", "m_shock", "p_abl")}))
    rows.sort(key=lambda row: row["score"])
    best = rows[0]

    fig = plt.figure(figsize=(17, 9.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=(1, 1, 0.82), wspace=0.28, hspace=0.32,
                  left=0.055, right=0.985, top=0.86, bottom=0.08)
    axes = {key: fig.add_subplot(gs[i // 2, i % 2]) for i, (key, *_) in enumerate(FIELDS)}
    m_end = max(r["m"][k[b]][-1] for b, r in runs.items())
    m_dense = np.unique(np.concatenate((np.linspace(1e-12, m_end, 1500),
                                        np.geomspace(1e-4 * min(r["m"][k[b]][0] for b, r in runs.items()), m_end, 1500))))
    t_an = float(runs[best["bath"]]["t"][k[best["bath"]]])
    fronts = [ivd.front_values(r, label, float(r["t"][k[b]])) for b, r in runs.items()]
    shock = max([analytic.front_values(t_an)["m_shock"]] + [fv["m_shock"] for fv in fronts])
    heat = max([analytic.front_values(t_an)["m_heat"]] + [fv["m_heat"] for fv in fronts])
    heat_min = min([analytic.front_values(t_an)["m_heat"]] + [fv["m_heat"] for fv in fronts])
    if scale is None:
        scale = "semilogx" if ivd.HYDRO_CASES[label].log_mass_axis else "linear"
    log_x, log_y = scale in ("semilogx", "loglog"), scale in ("semilogy", "loglog")
    x_first = 0.5e3 * min(r["m"][k[b]][0] for b, r in runs.items())
    x_lo, x_hi = {"full": (x_first if log_x else 0.0, 1.15e3 * shock),
                  "heat": (x_first if log_x else 0.0, 1.3e3 * heat),
                  "shock": (0.85e3 * heat_min, 1.1e3 * shock)}[region]
    m_dense = np.unique(np.concatenate((m_dense, np.linspace(max(x_lo, 1e-9) * 1e-3, x_hi * 1e-3, 1500))))
    sol = analytic.solve(mass=m_dense, time=t_an)
    keep = ivd.front_spike_mask(m_dense, sol, analytic.front_values(t_an)["m_heat"])   # no patch spike at the front
    for key, ref, ylabel, fscale in FIELDS:
        ax = axes[key]
        for b, r in runs.items():
            hl = b == best["bath"]
            # the closest bath wide underneath, the others thin on top: where baths agree, all stay visible
            ax.plot(r["m"][k[b]] * 1e3, r[key][k[b]] * fscale, color=BATH_COLOR[b], lw=3.4 if hl else 1.3,
                    alpha=0.8 if hl else 0.95, zorder=2 if hl else 3)
        ax.plot(m_dense[keep] * 1e3, np.asarray(sol[ref], dtype=float)[keep] * fscale, color=INK, ls="--", lw=1.8,
                zorder=4)
        # y limits from the simulations inside the mass window
        sim = np.concatenate([r[key][k[b]][(r["m"][k[b]] * 1e3 >= x_lo) & (r["m"][k[b]] * 1e3 <= x_hi)] * fscale
                              for b, r in runs.items()])
        if log_y and key != "u":
            pos = sim[sim > 0]
            ax.set_yscale("log")
            ax.set_ylim(0.7 * max(pos.min(), 1e-9 * pos.max()), 1.6 * pos.max())
        elif log_y:               # u changes sign: symmetric log, linear within 1 km/s
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(1.5 * min(sim.min(), -1.0), 1.5 * max(sim.max(), 1.0))
        else:
            lo, hi = sim.min(), sim.max()
            ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.12 * (hi - lo))
        if log_x:
            ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        _label_short_log_axes(ax)
        ax.set_ylabel(ylabel, color=INK, fontsize=12)
        ax.set_xlabel(r"mass $m$ [mg/cm$^2$]", color=INK, fontsize=11)
        ax.set_title(ref.capitalize(), color=INK, fontsize=13, loc="left", fontweight="bold")
        _style(ax)

    handles = [Line2D([0], [0], color=BATH_COLOR[b], lw=2.4, label=f"Diffusion, {BATH_SHORT[b]}") for b in runs]
    handles += [Line2D([0], [0], color=INK, ls="--", lw=1.8, label="Analytic (AblationSolver)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.925), ncol=len(handles),
               frameon=False, fontsize=10.5, handlelength=2.6, labelcolor=INK)

    tab = fig.add_subplot(gs[:, 2])
    tab.axis("off")
    tab.set_title(f"Baths at {t_an * 1e9:.1f} ns, best first", color=INK, fontsize=13, loc="left", fontweight="bold")
    short = {"paper": "paper", "calibrated": "T-calibrated", "calibrated_flux": "flux-calibrated"}
    cell = [[short[r["bath"]], f"{100 * r['score']:.1f}%", f"{r['m_heat']:+.1f}%", f"{r['m_shock']:+.1f}%",
             f"{r['p_abl']:+.1f}%"] for r in rows]
    h = 0.085 * (len(rows) + 1)
    t_ = tab.table(cellText=cell, colLabels=("bath", "profile\nL1", "heat\nfront", "shock", "front\np"),
                   cellLoc="center", colWidths=(0.32, 0.17, 0.17, 0.17, 0.17), bbox=(0.0, 1.0 - h, 1.0, h))
    t_.auto_set_font_size(False)
    t_.set_fontsize(11)
    for (i, j), c in t_.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_text_props(color=INK)
        if i == 0:
            c.set_text_props(color=MUTED, fontweight="bold")
        elif i == 1:
            c.set_facecolor("#eef4fc")
            c.set_text_props(color=INK, fontweight="bold")
        if i > 0 and j == 0:
            c.set_text_props(color=BATH_COLOR[rows[i - 1]["bath"]], fontweight="bold" if i == 1 else "normal")
    note = ("Best = smallest profile L1: the mass-weighted\n"
            "|diffusion - analytic| over the domain, relative to\n"
            "the analytic profile, averaged over rho, p, u and T.\n"
            "Front columns: diffusion vs. analytic\n"
            "(+ = diffusion ahead / higher).\n\n"
            "The analytic curve does not depend on the bath.\n"
            "IMC agrees with diffusion to a few % (per-bath\n"
            "figures) and is left out for readability.")
    tab.text(0.0, 1.0 - h - 0.04, note, va="top", ha="left", fontsize=10, color=MUTED,
             transform=tab.transAxes, linespacing=1.5)
    # one modest title line, as on the per-bath figures; the closest bath is the table's first row
    region_text = {"full": "full domain", "heat": "heat region", "shock": "shocked region"}[region]
    fig.suptitle(" · ".join((ivd.short_case_name(label), region_text, f"all baths at {t_an * 1e9:.1f} ns",
                                   ivd.VIEW_SCALE_TEXT[scale])),
                 x=0.055, ha="left", fontsize=16, fontweight="bold", color=INK, y=0.985)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return rows


VIEW_CASES = ("test3_smooth", "test4_full")
REGIONS = ("full", "heat", "shock")
SCALES = ("linear", "semilogx", "semilogy", "loglog")


def plot_views(labels=VIEW_CASES, baths=tuple(BATHS)):
    """Every region x scale view of each case, in each bath folder and (cases with more than one
    bath) in overlay/: md/phase_c/figures/<folder>/<case>/<region>_<scale>.png."""
    import matplotlib
    matplotlib.use("Agg")

    for label in labels:
        stem = figure_name(label).replace("_profiles.png", "")
        analytic = None
        for bath in baths:
            folder, out = BATHS[bath]
            if not (out / f"{label}_diffusion.npz").exists():
                continue
            analytic = analytic or Analytic(label)
            for region in REGIONS:
                for scale in SCALES:
                    path = FIG / folder / stem / f"{region}_{scale}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ivd.plot_hydro_profiles(out, label, (1, 2, 3), path, solver=analytic, method=BATH_SHORT[bath],
                                            region=region, scale=scale)
                    print(f"  wrote {path.relative_to(ivd.REPO)}", flush=True)
        if len(baths_of(label)) > 1:
            for region in REGIONS:
                for scale in SCALES:
                    path = FIG / "overlay" / stem / f"{region}_{scale}.png"
                    plot_overlay(label, path, region=region, scale=scale)
                    print(f"  wrote {path.relative_to(ivd.REPO)}", flush=True)


def main(argv=None):
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["test4_full"])
    ap.add_argument("--omega-scan", nargs="*", type=float, default=[])
    ap.add_argument("--N", type=int, default=400, help="cells of the omega-scan diffusion runs")
    ap.add_argument("--seeds", nargs="*", type=int, default=[], help="IMC seeds of the omega-scan runs")
    ap.add_argument("--run-only", action="store_true", help="only run the omega-scan simulations")
    ap.add_argument("--profiles", action="store_true", help="the full-profile figures, md/phase_c/figures/<bath>/")
    ap.add_argument("--overlay", action="store_true",
                    help="cases with more than one bath: one figure comparing them, md/phase_c/figures/overlay/")
    ap.add_argument("--baths", nargs="*", default=list(BATHS), choices=list(BATHS))
    ap.add_argument("--views", action="store_true",
                    help="every region (full, heat, shock) x scale (linear, semilogx, semilogy, loglog) view of --cases, "
                         "md/phase_c/figures/<folder>/<case>/")
    args = ap.parse_args(argv)
    labels = list(args.cases)
    for w in args.omega_scan:
        label = register_omega_case(w, N=args.N)
        ivd.run_hydro(SIM_OUT, label, seeds=tuple(args.seeds), physical=False)
        labels.append(label)
    if args.run_only:
        return
    if args.profiles:
        plot_profile_set(labels, args.baths)
        return
    if args.views:
        plot_views(labels, args.baths)
        return
    if args.overlay:
        for label in labels:
            if len(baths_of(label)) < 2:
                print(f"  {label}: one bath only, no overlay")
                continue
            path = FIG / "overlay" / figure_name(label)
            rows = plot_overlay(label, path)
            print(f"  wrote {path.relative_to(ivd.REPO)}: " +
                  ", ".join(f"{r['bath']} {100 * r['score']:.2f}%" for r in rows), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for label in labels:
        text = table(label, Analytic(label))
        print(text, flush=True)
        (OUT / f"{label}_table.md").write_text(text + "\n")


if __name__ == "__main__":
    main()

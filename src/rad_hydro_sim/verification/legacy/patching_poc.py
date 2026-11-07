"""LEGACY: the impulse-matched and pressure-matched patches (fix C) against diffusion and IMC.

Superseded: the impulse match lowers the whole-profile error with the paper bath for omega < 0
and Cu/Al, but flips the shock error (+7 ... +13% for omega <= -0.5), worsens the front pressure
and leaves a pressure step at the front; with the flux-calibrated bath it wins only for Al
(md/phase_c/legacy/analytic_patching_impulse.md).  The current comparison is
rad_hydro_sim.verification.bath_calibration (the original AblationSolver under each bath).

    uv run python -m rad_hydro_sim.verification.legacy.patching_poc --cases test4_full            # tables
    uv run python -m rad_hydro_sim.verification.legacy.patching_poc --cases test4_full --profiles
    uv run python -m rad_hydro_sim.verification.legacy.patching_poc --cases test4_full --overlay

Patchings (menahem_new.legacy.matched_piston_patch): original, C_pressure, C_impulse.
Figures go to results/phase_c/figures/legacy/impulse_patch/; tables to results/patching_poc/.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rad_hydro_sim.verification import imc_vs_diffusion as ivd

from rad_hydro_sim.verification.bath_calibration import (  # noqa: E402  (shared with the current comparison)
    SIM_OUT, TIMES, figure_name, omega_label, profile_score, register_omega_case, sim_fronts, solver_for)

OUT = ivd.REPO / "results" / "patching_poc"
PATCHINGS = ("original", "C_pressure", "C_impulse")
# simulation runs per bath (imc_vs_diffusion --bath), and their figure folders
BATHS = {"paper": ("paper_bath", SIM_OUT),
         "calibrated": ("calibrated_T_bath", SIM_OUT.with_name(SIM_OUT.name + "_calibrated")),
         "calibrated_flux": ("calibrated_flux_bath", SIM_OUT.with_name(SIM_OUT.name + "_calibrated_flux"))}
BATH_TEXT = {"paper": "paper bath", "calibrated": "bath calibrated to the surface temperature",
             "calibrated_flux": "bath calibrated to the energy flux"}
# analytic method per figure folder: (patching, what the figure calls it)
METHODS = {"original_patch": ("original", "original patch"),
           "impulse_patch": ("C_impulse", "impulse-matched patch")}
FIG = ivd.REPO / "results" / "phase_c" / "figures" / "legacy" / "impulse_patch"


def patched_solvers(label: str):
    """{patching: object with front_values(t) and solve(mass, time)} for one case."""
    from menahem_new.legacy.matched_piston_patch import MatchedPistonPatch

    base = solver_for(label)
    hs, ps = base.heat_solver, base.shock_solver
    p0s = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3

    class Original:
        def front_values(self, t):
            return dict(m_heat=float(hs.ablated_mass(time=t)),
                        m_shock=float(np.ravel(ps.shocked_mass(time=t))[0]),
                        p_abl=p0s * t**hs.c3, u_front=float(ps.piston_velocity(time=t)))

        def solve(self, *, mass, time):
            return base.solve(mass=mass, time=time)

    return {"original": Original(),
            "C_pressure": MatchedPistonPatch(base, front="massless", match="pressure"),
            "C_impulse": MatchedPistonPatch(base, front="massless", match="impulse", t_end=TIMES[-1])}


def table(label: str, solvers, sims, patchings=PATCHINGS, title=None) -> str:
    rows = [f"### {title or label}", "",
            "| t [ns] | quantity | diffusion | IMC | " + " | ".join(patchings) + " |",
            "|---|---|---|---|" + "---|" * len(patchings)]
    for t in TIMES:
        ref = {p: solvers[p].front_values(t) for p in patchings}
        for q, unit in (("m_heat", 1e3), ("m_shock", 1e3), ("p_abl", 1e-12)):
            d = sims["diffusion"][t][q]
            imc = [s[t][q] for k, s in sims.items() if k.startswith("imc")]
            cells = [f"{d * unit:.4g}", f"{np.mean(imc) * unit:.4g}" if imc else "-"]
            cells += [f"{ref[p][q] * unit:.4g} ({100 * (d / ref[p][q] - 1):+.1f}%)" for p in patchings]
            rows.append(f"| {t * 1e9:.1f} | {q} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("(m in mg/cm^2, p in Mbar; the % is diffusion vs. that patching)")
    return "\n".join(rows)


def plot_profile_set(labels, baths=tuple(BATHS)):
    """One full-profile figure per (bath, analytic method, case) with runs in that bath:
    results/phase_c/figures/<bath>_bath/<method>/<case>_profiles.png, the analytic curve in black."""
    import matplotlib
    matplotlib.use("Agg")

    for label in labels:
        solvers = None
        for bath in baths:
            folder, out = BATHS[bath]
            if not (out / f"{label}_diffusion.npz").exists():
                continue
            solvers = solvers or patched_solvers(label)
            for method, (patching, name) in METHODS.items():
                path = FIG / folder / method / figure_name(label)
                path.parent.mkdir(parents=True, exist_ok=True)
                ivd.plot_hydro_profiles(out, label, (1, 2, 3), path, solver=solvers[patching],
                                        analytic_label=f"Analytic ({name})",
                                        method=f"{BATH_TEXT[bath]}; analytic: {name}")
                print(f"  wrote {path.relative_to(ivd.REPO)}", flush=True)


# overlay figure: one hue per bath (reference palette, fixed order), ink for the analytic curves
BATH_COLOR = {"paper": "#2a78d6", "calibrated": "#eb6834", "calibrated_flux": "#1baf7a"}
BATH_SHORT = {"paper": "paper bath", "calibrated": "T-calibrated bath", "calibrated_flux": "flux-calibrated bath"}
METHOD_STYLE = {"original_patch": "--", "impulse_patch": "-"}
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e4e3dd"
FIELDS = (("rho", "density", r"$\rho$ [g/cm$^3$]", 1.0), ("p", "pressure", r"$P$ [Mbar]", 1e-12),
          ("u", "velocity", r"$u$ [km/s]", 1e-5), ("T_material", "temperature", r"$T$ [HeV]", 1.0 / ivd.K_PER_HEV))


def plot_overlay(label: str, path: Path, t: float = 2e-9):
    """Every (bath, analytic method) option for one case at t, overlaid, the best one by
    profile_score drawn bold, with the ranking and the front errors beside the profiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    solvers = patched_solvers(label)
    runs = {b: np.load(out / f"{label}_diffusion.npz") for b, (_, out) in BATHS.items()
            if (out / f"{label}_diffusion.npz").exists()}
    k = {b: int(np.argmin(np.abs(r["t"] - t))) for b, r in runs.items()}
    rows = []
    for b, r in runs.items():
        tb = float(r["t"][k[b]])
        fv = ivd.front_values(r, label, tb)
        for method, (patching, name) in METHODS.items():
            ref = solvers[patching].front_values(tb)
            rows.append(dict(bath=b, method=method, score=profile_score(r, k[b], solvers[patching], tb),
                             **{q: 100 * (fv[q] / ref[q] - 1) for q in ("m_heat", "m_shock", "p_abl")}))
    rows.sort(key=lambda row: row["score"])
    best = rows[0]

    fig = plt.figure(figsize=(17, 9.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=(1, 1, 0.92), wspace=0.28, hspace=0.32,
                  left=0.055, right=0.985, top=0.86, bottom=0.08)
    axes = {key: fig.add_subplot(gs[i // 2, i % 2]) for i, (key, *_) in enumerate(FIELDS)}
    m_dense = np.unique(np.concatenate([np.linspace(1e-12, r["m"][k[b]][-1], 1200) for b, r in runs.items()] +
                                       [np.geomspace(1e-4 * r["m"][k[b]][0], r["m"][k[b]][-1], 1200)
                                        for b, r in runs.items()]))
    t_an = float(runs[best["bath"]]["t"][k[best["bath"]]])
    analytic = {method: solvers[patching].solve(mass=m_dense, time=t_an) for method, (patching, _) in METHODS.items()}
    shock = max(max(solvers[p].front_values(t_an)["m_shock"] for p, _ in METHODS.values()),
                max(ivd.front_values(r, label, float(r["t"][k[b]]))["m_shock"] for b, r in runs.items()))
    log_axis = ivd.HYDRO_CASES[label].log_mass_axis
    for key, ref, ylabel, scale in FIELDS:
        ax = axes[key]
        for b, r in runs.items():
            hl = b == best["bath"]
            ax.plot(r["m"][k[b]] * 1e3, r[key][k[b]] * scale, color=BATH_COLOR[b], lw=2.4 if hl else 1.2,
                    alpha=1.0 if hl else 0.55, zorder=3 if hl else 2)
        for method, sol in analytic.items():
            hl = method == best["method"]
            ax.plot(m_dense * 1e3, np.asarray(sol[ref], dtype=float) * scale, color=INK if hl else MUTED,
                    ls=METHOD_STYLE[method], lw=2.2 if hl else 1.2, zorder=4 if hl else 1)
        sim = np.concatenate([r[key][k[b]] * scale for b, r in runs.items()])
        lo, hi = sim.min(), sim.max()
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.12 * (hi - lo))
        if log_axis:
            ax.set_xscale("log")
            ax.set_xlim(0.5e3 * min(r["m"][k[b]][0] for b, r in runs.items()), 1.15e3 * shock)
        else:
            ax.set_xlim(0.0, 1.15e3 * shock)
        ax.set_ylabel(ylabel, color=INK, fontsize=12)
        ax.set_xlabel(r"mass $m$ [mg/cm$^2$]", color=INK, fontsize=11)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=10)
        ax.set_title(ref.capitalize(), color=INK, fontsize=13, loc="left", fontweight="bold")

    # legend: the encodings, in one row under the title
    handles = [Line2D([0], [0], color=BATH_COLOR[b], lw=2.4, label=f"Diffusion, {BATH_SHORT[b]}") for b in runs]
    handles += [Line2D([0], [0], color=INK, ls=METHOD_STYLE[m], lw=2.0, label=f"Analytic, {n}")
                for m, (_, n) in METHODS.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.925), ncol=len(handles),
               frameon=False, fontsize=10.5, handlelength=2.6, labelcolor=INK)

    # ranking table
    tab = fig.add_subplot(gs[:, 2])
    tab.axis("off")
    tab.set_title(f"All options at {t_an * 1e9:.1f} ns, best first", color=INK, fontsize=13, loc="left",
                  fontweight="bold")
    cols = ("bath", "analytic", "profile\nL1", "heat\nfront", "shock", "front\np")
    table_name = {"paper": "paper", "calibrated": "T-calib.", "calibrated_flux": "flux-calib.",
                  "original_patch": "original", "impulse_patch": "impulse"}
    cell = [[table_name[r["bath"]], table_name[r["method"]],
             f"{100 * r['score']:.1f}%", f"{r['m_heat']:+.1f}%", f"{r['m_shock']:+.1f}%", f"{r['p_abl']:+.1f}%"]
            for r in rows]
    t_ = tab.table(cellText=cell, colLabels=cols, loc="upper left", cellLoc="center",
                   colWidths=(0.2, 0.17, 0.16, 0.16, 0.15, 0.16), bbox=(0.0, 1.0 - 0.075 * (len(rows) + 1), 1.0,
                                                                         0.075 * (len(rows) + 1)))
    t_.auto_set_font_size(False)
    t_.set_fontsize(10.5)
    for (i, j), c in t_.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_text_props(color=INK)
        if i == 0:
            c.set_text_props(color=MUTED, fontweight="bold")
            c.set_facecolor("white")
        elif i == 1:
            c.set_facecolor("#eef4fc")
            c.set_text_props(color=INK, fontweight="bold")
        if i > 0 and j == 0:   # the bath's color as a swatch: a colored left edge
            c.set_text_props(color=BATH_COLOR[rows[i - 1]["bath"]], fontweight="bold" if i == 1 else "normal")
    note = ("Best = smallest profile L1: the mass-weighted\n"
            "|diffusion - analytic| over the domain, relative to\n"
            "the analytic profile, averaged over rho, p, u and T.\n"
            "Front columns: diffusion vs. analytic\n"
            "(+ = diffusion ahead / higher).\n\n"
            "Bold curves: the best option's bath (diffusion) and\n"
            "analytic method. The analytic curve does not depend\n"
            "on the bath. IMC agrees with diffusion to a few %\n"
            "(per-bath figures) and is left out for readability.")
    tab.text(0.0, 1.0 - 0.075 * (len(rows) + 1) - 0.04, note, va="top", ha="left", fontsize=9.5, color=MUTED,
             transform=tab.transAxes, linespacing=1.5)
    verdict = f"Best: {BATH_SHORT[best['bath']]}, {METHODS[best['method']][1]}"
    if len(rows) > 1 and rows[1]["score"] - best["score"] < 1e-3:   # within 0.1 percentage point: no winner
        verdict = (f"Tie: {BATH_SHORT[best['bath']]} with the {METHODS[best['method']][1]} or the "
                   f"{METHODS[rows[1]['method']][1]}" if rows[1]["bath"] == best["bath"] else
                   f"Tie: {BATH_SHORT[best['bath']]}, {METHODS[best['method']][1]} and "
                   f"{BATH_SHORT[rows[1]['bath']]}, {METHODS[rows[1]['method']][1]}")
    fig.suptitle(f"{ivd.HYDRO_CASES[label].title}: every bath and patch at {t_an * 1e9:.1f} ns.  {verdict}",
                 x=0.055, ha="left", fontsize=15, fontweight="bold", color=INK, y=0.985)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return rows


def momentum_budget(label: str, t: float = 2e-9) -> dict:
    """The target's momentum beyond the cut m_f at t (per cm^2), as each closure has it and as
    the paper-bath diffusion run has it (md/phase_c/legacy/patching_fixes_explained.md, Sec. 4).

    push      int_0^t p_f dt', what the front pushes with;
    massive   the original patch: push - int_0^m_f u dm of PistonShock(p0) (the slab rides on);
    massless  the impulse match: the momentum its march delivered beyond the cut;
    sim       sum of u dm over the run's cells beyond the analytic m_f;
    theta     (sim - massless) / (massive - massless): 1 = massive closure, 0 = massless.
    """
    from menahem_new.legacy.matched_piston_patch import MatchedPistonPatch

    base = solver_for(label)
    hs, ps0 = base.heat_solver, base.shock_solver
    C = MatchedPistonPatch(base, front="massless", match="impulse", t_end=t)
    p0s = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3
    m_f = float(hs.ablated_mass(time=t))
    push = p0s * t**(1 + hs.c3) / (1 + hs.c3)
    massive = push - C._layer_momentum(ps0, t, m_f)
    ps = C._shock(C.amplitude(t))
    massless = ps.p0 * t**(1 + hs.c3) / (1 + hs.c3) - C._layer_momentum(ps, t, m_f)
    r = np.load(SIM_OUT / f"{label}_diffusion.npz")
    k = int(np.argmin(np.abs(r["t"] - t)))
    m = r["m"][k]
    dm = np.diff(m, prepend=0.0)
    beyond = (m - 0.5 * dm) > m_f
    sim = float(np.sum(r["u"][k][beyond] * dm[beyond]))
    p_ref = p0s * t**hs.c3
    return dict(push=push, massive=massive, massless=massless, sim=sim,
                theta=(sim - massless) / (massive - massless),
                p_cut_massive=C._at_cut(ps0, t, m_f)[0] / p_ref, p_cut_massless=C._at_cut(ps, t, m_f)[0] / p_ref,
                p_eff_over_pf=ps.p0 / ps0.p0, march=C.march)


def main(argv=None):
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["test4_full"])
    ap.add_argument("--omega-scan", nargs="*", type=float, default=[])
    ap.add_argument("--N", type=int, default=400, help="cells of the omega-scan diffusion runs")
    ap.add_argument("--seeds", nargs="*", type=int, default=[], help="IMC seeds of the omega-scan runs")
    ap.add_argument("--run-only", action="store_true", help="only run the omega-scan simulations")
    ap.add_argument("--profiles", action="store_true",
                    help="the full-profile figures, results/phase_c/figures/<bath>/<method>/")
    ap.add_argument("--baths", nargs="*", default=list(BATHS), choices=list(BATHS))
    ap.add_argument("--overlay", action="store_true",
                    help="one figure per case with every bath and patch, results/phase_c/figures/overlay/")
    ap.add_argument("--momentum", action="store_true",
                    help="the target's momentum beyond the cut: massive / massless closures vs the run")
    args = ap.parse_args(argv)
    labels = list(args.cases)
    for w in args.omega_scan:
        label = register_omega_case(w, N=args.N)
        ivd.run_hydro(SIM_OUT, label, seeds=tuple(args.seeds), physical=False)
        labels.append(label)
    if args.run_only:
        return
    if args.momentum:
        for label in labels:
            b = momentum_budget(label)
            print(f"{label}: push {b['push']:.4g}  massive {b['massive']:.4g}  massless {b['massless']:.4g}  "
                  f"sim {b['sim']:.4g}  theta {b['theta']:.2f}  (g/cm/s per cm^2)", flush=True)
        return
    if args.profiles:
        plot_profile_set(labels, args.baths)
        return
    if args.overlay:
        for label in labels:
            path = FIG / "overlay" / figure_name(label)
            rows = plot_overlay(label, path)
            print(f"  wrote {path.relative_to(ivd.REPO)}; best: {rows[0]['bath']} + {rows[0]['method']}; "
                  + ", ".join(f"{r['bath']}/{r['method']} {100 * r['score']:.2f}%" for r in rows), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for label in labels:
        solvers = patched_solvers(label)
        for bath in args.baths:
            folder, out = BATHS[bath]
            if not (out / f"{label}_diffusion.npz").exists():
                continue
            text = table(label, solvers, sim_fronts(label, out), title=f"{label}, {BATH_TEXT[bath]}")
            print(text, flush=True)
            (OUT / f"{label}_{folder}_table.md").write_text(text + "\n")


if __name__ == "__main__":
    main()

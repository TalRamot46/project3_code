"""Gold, 1 HeV, tau = 0 (Test 4's material and drive) for omega = -0.8 ... +0.8: the heat front
(x_f, m_f) and the shock (x_s, m_s) against time, diffusion and IMC with the paper bath and with
the flux-calibrated bath, against the semi-analytic AblationSolver (md/phase_c/omega_sweep/).

    uv run python -m rad_hydro_sim.verification.omega_sweep --run -0.8 -0.6      # simulations
    uv run python -m rad_hydro_sim.verification.omega_sweep --figures             # whatever has run

--run per omega: diffusion with the paper bath; then, unless --no-flux, the flux calibration
(imc_vs_diffusion.calibrate_bath, up to 7 diffusion runs) and diffusion with that bath; IMC
(one seed, paper bath) only with --imc.  Runs are cached in results/omega_sweep{,_calibrated_flux}.

The heat solution does not depend on omega, so every omega reuses Test 4's cached heat solver and
builds only its own PistonShock (seconds instead of ~20 min).
"""
from __future__ import annotations

import argparse
import copy
import logging
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from rad_hydro_sim.verification import bath_calibration as bc

OMEGAS = tuple(round(float(w), 1) + 0.0 for w in np.arange(-0.8, 0.81, 0.2))   # + 0.0: no "-0.0"
OUT = ivd.REPO / "results" / "omega_sweep"
OUT_FLUX = OUT.with_name(OUT.name + "_calibrated_flux")
FIG = ivd.REPO / "md" / "phase_c" / "omega_sweep"
GRADED_FROM = 0.4      # from here on the shocked mass is >~ 100x the ablated one: graded mesh as Test 3
CELLS_PER_LAYER = 40   # uniform-mass mesh: cells across the ablated layer at t_end


def label_of(omega: float) -> str:
    return "sweep_" + bc.omega_label(omega)


def solver_of(omega: float):
    """AblationSolver for gold at this omega: Test 4's heat solver with its own PistonShock."""
    import rad_hydro_sim.verification.menahem_comparison  # noqa: F401  (scipy shims, PistonShock patch)
    from menahem_new.piston_shock_og import PistonShock

    path = OUT / "solvers" / f"solver_{label_of(omega)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    base = bc.solver_for("test4_full")
    hs, ps = base.heat_solver, base.shock_solver
    solver = copy.copy(base)
    solver.shock_solver = PistonShock(rho0=ps.rho0, omega=float(omega), p0=hs.Pf * hs.A**hs.a3 * hs.B**hs.b3,
                                      tau=hs.c3, gamma=ps.gamma)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(solver, f)
    return solver


def register(omega: float) -> str:
    """Preset + HydroCase for this omega (Test 4's case with omega changed), registered at run time."""
    from rad_hydro_sim.problems import presets_config as P

    label = label_of(omega)
    if label in ivd.HYDRO_CASES:
        return label
    base = P.PRESET_TEST_CASES[P.CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL]
    solver = solver_of(omega)
    t_end = base.t_sec_end
    m_s = float(np.ravel(solver.shock_solver.shocked_mass(time=t_end))[0])
    m_f = float(solver.heat_solver.ablated_mass(time=t_end))
    key = label.upper()
    title = rf"Au, $\omega={omega:+.1f}$"
    # a 15% margin in mass beyond the analytic shock at t_end
    P.PRESET_TEST_CASES[key] = replace(base, omega=float(omega), title=title,
                                       x_max=float(solver.shock_solver.initial_position(mass=1.15 * m_s)),
                                       bath_correction=str(ivd.BATH_CORRECTION_DIR / f"{key}.txt"))
    setattr(P, key, key)
    if omega < GRADED_FROM:
        n = int(np.clip(np.ceil(CELLS_PER_LAYER * 1.15 * m_s / m_f), 400, 2000))
        spec = ivd.HydroCase(key, title, N=n, store_every=40, config=(("mesh", "uniform_mass"),),
                             imc=ivd.HYDRO_CASES["test4_full"].imc)
    else:   # rho ~ x^-omega piles the mass at the surface: Test 3's graded mesh and settings
        spec = replace(ivd.HYDRO_CASES["test3_full"], preset=key, title=title, log_mass_axis=False)
        if omega >= 0.6:   # the compatible energy update fails to converge at the default 2 ps cap
            spec = replace(spec, dt_max=2e-13)
    ivd.HYDRO_CASES[label] = spec
    return label


def analytic_fronts(omega: float, times):
    """{x_f, m_f, x_s, m_s} of the AblationSolver at times (the heat front sits on the shock
    solution's tracer of m_f, as in AblationSolver.solve)."""
    solver = solver_of(omega)
    hs, ps = solver.heat_solver, solver.shock_solver
    out = {k: [] for k in ("x_f", "m_f", "x_s", "m_s")}
    for t in times:
        m_f = float(hs.ablated_mass(time=t))
        out["m_f"].append(m_f)
        out["x_f"].append(float(ps.solve(mass=np.array([m_f, 1.0001 * m_f]), time=t)["position"][0]))
        out["m_s"].append(float(np.ravel(ps.shocked_mass(time=t))[0]))
        out["x_s"].append(float(np.ravel(ps.shock_position(time=t))[0]))
    return {k: np.asarray(v) for k, v in out.items()}


def run(omega: float, flux=True, imc=False):
    label = register(omega)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_FLUX.mkdir(parents=True, exist_ok=True)
    ivd.run_hydro(OUT, label, seeds=(1,) if imc else (), physical=False)
    if flux:
        cal = ivd.BATH_CORRECTION_DIR / f"{label.upper()}_flux.txt"
        if not cal.exists():
            print(f"[{label}] calibrating the bath to the energy flux", flush=True)
            ivd.calibrate_bath(label, target="flux")
        ivd.run_hydro(OUT_FLUX, label, seeds=(), physical=False, bath="calibrated_flux")


def sim_series(omega: float):
    """{run name: {t, x_f, m_f, x_s, m_s}} of the runs of this omega that exist."""
    label = label_of(omega)
    found = {}
    for name, path in (("diffusion, paper bath", OUT / f"{label}_diffusion.npz"),
                       ("IMC, paper bath", OUT / f"{label}_imc1.npz"),
                       ("diffusion, flux-calibrated bath", OUT_FLUX / f"{label}_diffusion.npz")):
        if path.exists():
            r = np.load(path)
            # before ~1% of t_end there is no shock yet and the front finder returns the domain edge
            ok = r["series_t"] > 0.01 * r["series_t"][-1]
            found[name] = dict(t=r["series_t"][ok], m_f=r["m_heat"][ok], m_s=r["m_shock"][ok],
                               x_f=r["x_heat"][ok], x_s=r["x_shock"][ok])
    return found


# reference palette (dataviz skill): one hue per bath in fixed order, ink for the analytic curve
# where the baths agree (omega >= 0) the curves coincide: the paper bath is drawn wide underneath,
# the flux-calibrated bath narrow on top, so both stay visible
STYLE = {"diffusion, paper bath": dict(color="#2a78d6", lw=3.4, alpha=0.75),
         "IMC, paper bath": dict(color="#184f95", lw=1.4, ls=(0, (4, 2))),
         "diffusion, flux-calibrated bath": dict(color="#1baf7a", lw=1.6)}
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e4e3dd"
PANELS = (("x_f", r"heat front $x_f$ [$\mu$m]", 1e4), ("m_f", r"heat front $m_f$ [mg/cm$^2$]", 1e3),
          ("x_s", r"shock $x_s$ [$\mu$m]", 1e4), ("m_s", r"shock $m_s$ [mg/cm$^2$]", 1e3))


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=10)


def at(series, key, t):
    sel = np.isfinite(series[key])
    return float(np.interp(t, series["t"][sel], series[key][sel]))


def plot_omega(omega: float, path: Path):
    """The four fronts against time for one omega, with the deviation from the analytic solution
    at t_end in the corner of each panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sims = sim_series(omega)
    t_end = max(s["t"][-1] for s in sims.values())
    tg = np.linspace(0.02, 1.0, 120) * t_end
    an = analytic_fronts(omega, tg)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    for ax, (key, ylabel, scale) in zip(axes.flat, PANELS):
        ax.plot(tg * 1e9, an[key] * scale, color=INK, lw=1.8, ls="--", zorder=4)
        lines = []
        for name, s in sims.items():
            ax.plot(s["t"] * 1e9, s[key] * scale, **STYLE[name])
            dev = 100 * (at(s, key, t_end) / an[key][-1] - 1)
            lines.append((STYLE[name]["color"], f"{name.split(',')[0]}{' (flux)' if 'flux' in name else ''}: {dev:+.1f}%"))
        for i, (col, txt) in enumerate(lines):
            ax.text(0.03, 0.95 - 0.075 * i, txt, transform=ax.transAxes, va="top", fontsize=9.5, color=col)
        ax.text(0.03, 0.95 - 0.075 * len(lines), f"vs. analytic at {t_end * 1e9:g} ns", transform=ax.transAxes,
                va="top", fontsize=9, color=MUTED)
        ax.set_xlim(0, t_end * 1e9)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("t [ns]", color=INK, fontsize=11)
        ax.set_ylabel(ylabel, color=INK, fontsize=11)
        _style(ax)
    handles = [Line2D([0], [0], label=n, **STYLE[n]) for n in sims]
    handles.append(Line2D([0], [0], color=INK, lw=1.8, ls="--", label="analytic (AblationSolver)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=len(handles), frameon=False,
               fontsize=10, labelcolor=INK, handlelength=2.6)
    spec = ivd.HYDRO_CASES[label_of(omega)]
    fig.suptitle(rf"Gold, 1 HeV, $\omega = {omega:+.1f}$: heat front and shock against time"
                 f"  ({spec.N} cells{', graded mesh' if omega >= GRADED_FROM else ', uniform in mass'})",
                 x=0.06, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary(omegas, path: Path, t: float = 2e-9):
    """Deviation from the analytic solution at t against omega, one panel per front quantity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = {}
    for w in omegas:
        sims = sim_series(w)
        an = analytic_fronts(w, [t])
        rows[w] = {name: {key: 100 * (at(s, key, t) / an[key][0] - 1) for key, *_ in PANELS}
                   for name, s in sims.items() if s["t"][-1] >= 0.999 * t}
    names = [n for n in STYLE if any(n in r for r in rows.values())]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    for ax, (key, ylabel, _) in zip(axes.flat, PANELS):
        for name in names:
            pts = [(w, r[name][key]) for w, r in rows.items() if name in r]
            st = dict(STYLE[name])
            ax.plot(*zip(*pts), marker="o", ms=6, **st)
        ax.axhline(0, color=MUTED, lw=1)
        ax.margins(x=0.04, y=0.08)
        ax.set_title(ylabel.split(" [")[0], color=INK, fontsize=12, loc="left", fontweight="bold")
        ax.set_ylabel("simulation vs. analytic [%]", color=INK, fontsize=10.5)
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel(r"$\omega$", color=INK, fontsize=12)
        ax.set_xticks(list(omegas))
    handles = [Line2D([0], [0], marker="o", ms=6, label=n, **STYLE[n]) for n in names]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=len(handles), frameon=False,
               fontsize=10.5, labelcolor=INK)
    fig.suptitle(rf"Gold, 1 HeV, $\tau = 0$: fronts at {t * 1e9:g} ns against the semi-analytic solution, by $\omega$",
                 x=0.06, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return rows


def main(argv=None):
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", type=float, default=None, help="omegas to simulate")
    ap.add_argument("--no-flux", action="store_true", help="skip the flux calibration and its run")
    ap.add_argument("--imc", action="store_true", help="also IMC (one seed, paper bath)")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    if args.run is not None:
        for w in args.run:
            run(w, flux=not args.no_flux, imc=args.imc)
    if args.figures:
        done = [w for w in OMEGAS if (OUT / f"{label_of(w)}_diffusion.npz").exists()]
        for w in done:
            register(w)
            path = FIG / f"fronts_omega{w:+.1f}.png".replace("+", "p").replace("-", "m")
            plot_omega(w, path)
            print(f"  wrote {path.relative_to(ivd.REPO)}", flush=True)
        rows = plot_summary(done, FIG / "fronts_vs_omega.png")
        for w, r in rows.items():
            print(f"  omega {w:+.1f}: " + "; ".join(f"{n}: " + ", ".join(f"{k} {v:+.1f}%" for k, v in d.items())
                                                for n, d in r.items()), flush=True)


if __name__ == "__main__":
    main()

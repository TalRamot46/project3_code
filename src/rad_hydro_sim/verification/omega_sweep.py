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

import scipy.optimize

import rad_hydro_sim.verification.menahem_comparison  # noqa: F401  (scipy shims, PistonShock patch)
from menahem_new.piston_shock_og import PistonShock
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


class BracketedPistonShock(PistonShock):
    """PistonShock whose shock root xi_s is found by brentq over a scanned bracket.

    Menahem's find_xsi_s_fsolve starts fsolve from a fixed guess; for omega >= 0.75 (with this
    drive) the root function is flat at -1 there and fsolve fails, although the root exists
    (omega = 0.8: xi_s = 4.137).  For omega <= 0.7 both give the same root to 1e-6.
    """

    def find_xsi_s_fsolve(self):
        xs = np.geomspace(0.5, 50.0, 60)
        f = [self.fxsi_s(x) for x in xs]
        i = next(j for j in range(len(xs) - 1) if f[j] * f[j + 1] < 0)
        return scipy.optimize.brentq(self.fxsi_s, xs[i], xs[i + 1], xtol=1e-12)


def solver_of(omega: float):
    """AblationSolver for gold at this omega: Test 4's heat solver with its own PistonShock."""

    path = OUT / "solvers" / f"solver_{label_of(omega)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    base = bc.solver_for("test4_full")
    hs, ps = base.heat_solver, base.shock_solver
    solver = copy.copy(base)
    solver.shock_solver = BracketedPistonShock(rho0=ps.rho0, omega=float(omega), p0=hs.Pf * hs.A**hs.a3 * hs.B**hs.b3,
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
        if omega >= 0.6:
            # The first step (1e-10 t_end, not CFL-limited) heats the 1e-15 cm surface cell ~2000x;
            # its sound speed then puts c_s dt / dx ~ 6 and the compatible energy update diverges.
            spec = replace(spec, config=tuple(spec.config) + (("dt_first_frac", 1e-12),))
    ivd.HYDRO_CASES[label] = spec
    return label


def register_fine(omega: float) -> str:
    """The same case on a finer mesh, label <label>_fine, sharing the preset (and its flux table):
    2.5x the cells of the uniform-mass mesh below GRADED_FROM; above it, the graded mesh with
    Test 3's fine-mesh cap of 0.3 mg/cm^2 per cell (test3_smooth)."""
    label = register(omega)
    fine = label + "_fine"
    if fine in ivd.HYDRO_CASES:
        return fine
    spec = ivd.HYDRO_CASES[label]
    if omega < GRADED_FROM:
        spec = replace(spec, N=int(round(2.5 * spec.N)), store_every=100)   # ~1e3 frames, not ~3e3
    else:
        spec = replace(spec, config=tuple(spec.config) + (("max_cell_mass", 3e-4),), store_every=2000)
    ivd.HYDRO_CASES[fine] = spec
    return fine


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


def run(omega: float, flux=True, imc=False, imc_flux=False, fine=False, paper=True, seeds=(1,)):
    """Paper-bath diffusion (and IMC with imc), the flux calibration and flux-bath diffusion (and
    IMC with imc_flux), IMC with each of seeds; fine: the same on register_fine's mesh, with the
    coarse mesh's flux table.  Every run is cached by file (<label>_diffusion.npz, _imc<seed>.npz)."""
    label = register(omega)
    run_label = register_fine(omega) if fine else label
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_FLUX.mkdir(parents=True, exist_ok=True)
    if paper:
        ivd.run_hydro(OUT, run_label, seeds=tuple(seeds) if imc else (), physical=False)
    if flux:
        cal = ivd.BATH_CORRECTION_DIR / f"{label.upper()}_flux.txt"
        if not cal.exists():
            print(f"[{label}] calibrating the bath to the energy flux", flush=True)
            ivd.calibrate_bath(label, target="flux")
        ivd.run_hydro(OUT_FLUX, run_label, seeds=tuple(seeds) if imc_flux else (), physical=False,
                      bath="calibrated_flux")


def run_parallel(omegas, seeds, jobs: int, paper=True, flux=True):
    """Every (omega, bath, seed) IMC run as its own process, `jobs` at a time, the slowest omegas
    first (+0.6: ~2 h a seed, +0.4: ~50 min, the rest minutes).  First, one process per omega and
    bath for the diffusion run (a no-op when cached), which the seeds of that pair share.  Logs:
    results/omega_sweep/logs/run_<w>_<bath>_<run>.log."""
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    baths = [b for b, on in (("paper", paper), ("flux", flux)) if on]
    only = {"paper": "--no-flux", "flux": "--no-paper"}
    imc_flag = {"paper": "--imc", "flux": "--imc-flux"}
    diffusion = [((f"--run={w}", only[b]), f"run_{w:+.1f}_{b}_diffusion.log") for w in omegas for b in baths]
    imc = sorted((((f"--run={w}", only[b], imc_flag[b], "--seeds", str(s)), f"run_{w:+.1f}_{b}_imc{s}.log")
                  for w in omegas for b in baths for s in seeds), key=lambda job: -float(job[0][0][6:]))

    def launch(job):
        args, log = job
        with open(logs / log, "w", encoding="utf-8") as f:
            code = subprocess.run([sys.executable, "-m", "rad_hydro_sim.verification.omega_sweep", *args],
                                  stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"  {'done  ' if code == 0 else 'FAILED'} {' '.join(args)}  (log {log})", flush=True)
        return code

    for stage in (diffusion, imc):
        with ThreadPoolExecutor(max(1, jobs)) as pool:
            codes = list(pool.map(launch, stage))
        if any(codes):
            print(f"  {sum(c != 0 for c in codes)} of {len(stage)} runs failed; see the logs", flush=True)


IMC_SEEDS = range(1, 10)   # the seeds looked for on disk


def _series(r):
    # before ~1% of t_end there is no shock yet and the front finder returns the domain edge
    ok = r["series_t"] > 0.01 * r["series_t"][-1]
    return dict(t=r["series_t"][ok], m_f=r["m_heat"][ok], m_s=r["m_shock"][ok], x_f=r["x_heat"][ok],
                x_s=r["x_shock"][ok])


def _imc_mean(out: Path, label: str):
    """The mean front series over the IMC seeds on disk (each seed interpolated to the first one's
    times), with n_seeds; None without any."""
    runs = [_series(np.load(out / f"{label}_imc{s}.npz")) for s in IMC_SEEDS if (out / f"{label}_imc{s}.npz").exists()]
    if not runs:
        return None
    t = runs[0]["t"]
    mean = {k: np.mean([np.interp(t, r["t"], r[k]) for r in runs], axis=0) for k in ("m_f", "m_s", "x_f", "x_s")}
    return dict(t=t, **mean, n_seeds=len(runs))


def sim_series(omega: float, fine: bool = False):
    """{run name: {t, x_f, m_f, x_s, m_s, n_seeds}} of the runs of this omega that exist (fine: on
    the register_fine mesh); IMC is the mean over its seeds."""
    label = label_of(omega) + ("_fine" if fine else "")
    found = {}
    for name, out, kind in (("diffusion, paper bath", OUT, "diffusion"), ("IMC, paper bath", OUT, "imc"),
                            ("diffusion, flux-calibrated bath", OUT_FLUX, "diffusion"),
                            ("IMC, flux-calibrated bath", OUT_FLUX, "imc")):
        if kind == "imc":
            mean = _imc_mean(out, label)
            if mean is not None:
                found[name] = mean
        elif (out / f"{label}_diffusion.npz").exists():
            found[name] = dict(_series(np.load(out / f"{label}_diffusion.npz")), n_seeds=0)
    return found


def legend_name(name: str, counts) -> str:
    """'IMC, paper bath (mean of 3 seeds)' when every omega shown has the same number of seeds > 1."""
    counts = set(counts)
    n = counts.pop() if len(counts) == 1 else 0
    return f"{name} (mean of {n} seeds)" if n > 1 else name


# reference palette (dataviz skill): one hue per bath in fixed order, ink for the analytic curve
# where the baths agree (omega >= 0) the curves coincide: the paper bath is drawn wide underneath,
# the flux-calibrated bath narrow on top, so both stay visible
STYLE = {"diffusion, paper bath": dict(color="#2a78d6", lw=3.4, alpha=0.75),
         "IMC, paper bath": dict(color="#184f95", lw=1.4, ls=(0, (4, 2))),
         "diffusion, flux-calibrated bath": dict(color="#1baf7a", lw=1.6),
         "IMC, flux-calibrated bath": dict(color="#127a55", lw=1.4, ls=(0, (4, 2)))}
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
        if (omega, key) in UNRESOLVED:
            ax.text(0.97, 0.5, UNRESOLVED[(omega, key)], transform=ax.transAxes, ha="right", va="center",
                    fontsize=9.5, color="#a33")
        ax.set_xlim(0, t_end * 1e9)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("t [ns]", color=INK, fontsize=11)
        ax.set_ylabel(ylabel, color=INK, fontsize=11)
        _style(ax)
    handles = [Line2D([0], [0], label=legend_name(n, [s["n_seeds"]]), **STYLE[n]) for n, s in sims.items()]
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


def omega_tag(omega: float) -> str:
    """'omegam0.2' for -0.2, 'omegap0.2' for +0.2: the file names of this folder."""
    return f"omega{omega:+.1f}".replace("+", "p").replace("-", "m")


PROFILE_BATHS = (("paper bath", OUT, "paper_bath"), ("flux-calibrated bath", OUT_FLUX, "calibrated_flux_bath"))


def plot_profiles(omegas):
    """rho, p, u, T against mass at 1, 1.5 and 2 ns over the full domain, linear axes: diffusion and
    IMC (one seed) under each bath against this omega's solver, as the per-bath figures of
    md/phase_c/figures (imc_vs_diffusion.plot_hydro_profiles):
    md/phase_c/omega_sweep/profiles/<bath>/omega<w>.png."""
    import matplotlib
    matplotlib.use("Agg")

    for w in omegas:
        label = register(w)
        solver = solver_of(w)
        for bath, out, folder in PROFILE_BATHS:
            if not (out / f"{label}_diffusion.npz").exists():
                continue
            seeds = tuple(s for s in IMC_SEEDS if (out / f"{label}_imc{s}.npz").exists())
            path = FIG / "profiles" / folder / f"{omega_tag(w)}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            ivd.plot_hydro_profiles(out, label, seeds, path, solver=solver, method=bath, region="full",
                                    scale="linear")
            print(f"  wrote {path.relative_to(ivd.REPO)}", flush=True)


NO_SIMULATION = {0.8: "no simulation: the ablated layer lies\nwithin 1e-24 cm of the surface (rho > 1e20)"}
# quantities the mesh does not resolve: at omega = +0.6 the heated cell holding the heat front is
# ~0.5 um wide, wider than x_f itself (0.19 um), so its position is not defined to better than that
UNRESOLVED = {(0.6, "x_f"): "x_f not resolved: the heated cell at the\nfront (0.5 um) is wider than x_f (0.19 um)"}


def plot_summary(omegas, path: Path, t: float = 2e-9):
    """Deviation from the analytic solution at t against omega, one panel per front quantity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows, seeds = {}, {}
    for w in omegas:
        sims = sim_series(w)
        an = analytic_fronts(w, [t])
        rows[w] = {name: {key: 100 * (at(s, key, t) / an[key][0] - 1) for key, *_ in PANELS}
                   for name, s in sims.items() if s["t"][-1] >= 0.999 * t}
        for name, s in sims.items():
            seeds.setdefault(name, []).append(s["n_seeds"])
    names = [n for n in STYLE if any(n in r for r in rows.values())]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    for ax, (key, ylabel, _) in zip(axes.flat, PANELS):
        for name in names:
            pts = [(w, r[name][key]) for w, r in rows.items() if name in r and (w, key) not in UNRESOLVED]
            st = dict(STYLE[name])
            ax.plot(*zip(*pts), marker="o", ms=6, **st)
        for (w, k), note in UNRESOLVED.items():
            if k == key and w in rows:
                ax.axvspan(w - 0.07, w + 0.07, color=GRID, alpha=0.6, lw=0)
                ax.text(w, 0.5, note, transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="center",
                        fontsize=8.5, color=MUTED)
        ax.axhline(0, color=MUTED, lw=1)
        ax.margins(x=0.04, y=0.08)
        ax.set_title(ylabel.split(" [")[0], color=INK, fontsize=12, loc="left", fontweight="bold")
        ax.set_ylabel("simulation vs. analytic [%]", color=INK, fontsize=10.5)
        _style(ax)
    # the omegas with no simulation, marked where they would be
    missing = [w for w in OMEGAS if w not in rows]
    for ax in axes.flat:
        for w in missing:
            ax.axvspan(w - 0.07, w + 0.07, color=GRID, alpha=0.6, lw=0)
        ax.set_xlim(min(OMEGAS) - 0.1, max(OMEGAS) + 0.1)
    for w in missing:
        axes[0, 0].text(w, 0.5, NO_SIMULATION.get(w, "not run"), transform=axes[0, 0].get_xaxis_transform(),
                        rotation=90, ha="center", va="center", fontsize=8.5, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel(r"$\omega$", color=INK, fontsize=12)
        ax.set_xticks(list(OMEGAS))
    handles = [Line2D([0], [0], marker="o", ms=6, label=legend_name(n, seeds[n]), **STYLE[n]) for n in names]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=len(handles), frameon=False,
               fontsize=10.5, labelcolor=INK)
    fig.suptitle(rf"Gold, 1 HeV, $\tau = 0$: fronts at {t * 1e9:g} ns against the semi-analytic solution, by $\omega$",
                 x=0.06, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return rows


def plot_convergence(omegas, path: Path, t: float = 2e-9):
    """Deviation from the solver at t against omega on the default mesh (solid) and the
    register_fine mesh (open markers, dotted), diffusion with each bath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    baths = ("diffusion, paper bath", "diffusion, flux-calibrated bath")
    rows = {}
    for w in omegas:
        an = analytic_fronts(w, [t])
        coarse, fine = sim_series(w), sim_series(w, fine=True)
        rows[w] = {b: {key: (100 * (at(coarse[b], key, t) / an[key][0] - 1),
                             100 * (at(fine[b], key, t) / an[key][0] - 1)) for key, *_ in PANELS}
                   for b in baths if b in coarse and b in fine}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    for ax, (key, ylabel, _) in zip(axes.flat, PANELS):
        for b in baths:
            pts = [(w, r[b][key]) for w, r in rows.items() if b in r and (w, key) not in UNRESOLVED]
            if not pts:
                continue
            color = STYLE[b]["color"]
            ws = [p[0] for p in pts]
            ax.plot(ws, [p[1][0] for p in pts], color=color, lw=2.0, marker="o", ms=6)
            ax.plot(ws, [p[1][1] for p in pts], color=color, lw=1.4, ls=":", marker="o", ms=7,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6)
        ax.axhline(0, color=MUTED, lw=1)
        ax.margins(x=0.04, y=0.08)
        ax.set_title(ylabel.split(" [")[0], color=INK, fontsize=12, loc="left", fontweight="bold")
        ax.set_ylabel("simulation vs. analytic [%]", color=INK, fontsize=10.5)
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel(r"$\omega$", color=INK, fontsize=12)
        ax.set_xticks(sorted(rows))
    handles = [Line2D([0], [0], color=STYLE[b]["color"], lw=2.0, marker="o", ms=6, label=f"{b}, default mesh")
               for b in baths]
    handles += [Line2D([0], [0], color=STYLE[b]["color"], lw=1.4, ls=":", marker="o", ms=7, markerfacecolor="white",
                       markeredgecolor=STYLE[b]["color"], markeredgewidth=1.6, label=f"{b}, fine mesh") for b in baths]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.96), ncol=2, frameon=False,
               fontsize=10, labelcolor=INK)
    fig.suptitle(rf"Mesh convergence of the gold $\omega$ sweep at {t * 1e9:g} ns: default vs. fine mesh "
                 r"(2.5$\times$ the cells; $\omega \geq 0.4$: cells $\leq$ 0.3 mg/cm$^2$)",
                 x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return rows


def main(argv=None):
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", type=float, default=None, help="omegas to simulate")
    ap.add_argument("--no-flux", action="store_true", help="skip the flux calibration and its run")
    ap.add_argument("--imc", action="store_true", help="also IMC (one seed, paper bath)")
    ap.add_argument("--imc-flux", action="store_true", help="also IMC (one seed) with the flux-calibrated bath")
    ap.add_argument("--fine", action="store_true", help="the finer mesh of register_fine")
    ap.add_argument("--no-paper", action="store_true", help="skip the paper-bath runs")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1], help="IMC seeds of --imc / --imc-flux")
    ap.add_argument("--jobs", type=int, default=0,
                    help="with --run: IMC with --seeds under both baths (unless --no-paper / --no-flux), "
                         "one process per (omega, bath, seed), this many at once (run_parallel)")
    ap.add_argument("--figures", action="store_true", help="fronts, summary, convergence and full profiles")
    ap.add_argument("--profiles", action="store_true", help="only the full profiles, profiles/<bath>/")
    args = ap.parse_args(argv)
    if args.run is not None and args.jobs > 0:
        run_parallel(args.run, args.seeds, args.jobs, paper=not args.no_paper, flux=not args.no_flux)
    elif args.run is not None:
        for w in args.run:
            run(w, flux=not args.no_flux, imc=args.imc, imc_flux=args.imc_flux, fine=args.fine,
                paper=not args.no_paper, seeds=args.seeds)
    done = [w for w in OMEGAS if (OUT / f"{label_of(w)}_diffusion.npz").exists()]
    if args.profiles or args.figures:
        plot_profiles(done)
    if args.figures:
        for w in done:
            register(w)
            path = FIG / f"fronts_{omega_tag(w)}.png"
            plot_omega(w, path)
            print(f"  wrote {path.relative_to(ivd.REPO)}", flush=True)
        rows = plot_summary(done, FIG / "fronts_vs_omega.png")
        conv = plot_convergence(done, FIG / "mesh_convergence.png")
        for w, r in conv.items():
            print(f"  convergence omega {w:+.1f}: " + "; ".join(
                f"{b}: " + ", ".join(f"{k} {c:+.1f} -> {fi:+.1f}%" for k, (c, fi) in d.items()) for b, d in r.items()),
                flush=True)
        for w, r in rows.items():
            print(f"  omega {w:+.1f}: " + "; ".join(f"{n}: " + ", ".join(f"{k} {v:+.1f}%" for k, v in d.items())
                                                for n, d in r.items()), flush=True)


if __name__ == "__main__":
    main()

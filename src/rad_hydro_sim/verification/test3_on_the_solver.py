"""Test 3 on the solver: the runs of md/phase_c/test3_on_the_solver.md.

What separates the cold-start Test 3 runs from the semi-analytic solution, and the setting of
each cure:

- the ablated layer is unresolved at t -> 0        -> mesh="log_mass" (a self-similar surface mesh)
- a shock crossing cells of growing mass           -> 0.3% growth per cell in the shocked region
- matter and radiation out of equilibrium (< 0.1 ns) -> force_black="conduction" (the solver's
  one-temperature equation, T_s imposed), or an analytic start at 0.1 ns for the 2T / IMC runs
- the seeded shock inside a cell                   -> seed_shock_on_face
- the 300 K target (Mach 11.5 instead of infinity)  -> T_initial = 30 K

    python -m rad_hydro_sim.verification.test3_on_the_solver --run cold_1T_30K
    python -m rad_hydro_sim.verification.test3_on_the_solver --run-all --jobs 2
    python -m rad_hydro_sim.verification.test3_on_the_solver --figures --table
    python -m rad_hydro_sim.verification.test3_on_the_solver --slider cold_1T_30K
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from rad_hydro_sim.verification import imc_vs_diffusion as ivd

REPO = ivd.REPO
OUT = REPO / "results" / "test3_on_the_solver"                               # run data (untracked)
FIGURES = REPO / "results" / "phase_c" / "figures" / "test3_on_the_solver"   # figures and table (tracked)
BASE = "test3_smooth"                       # the preset, domain and snapshot times
SNAP_NS = (0.5, 1.0, 1.5, 2.0)
T_SEED = 1e-10
SLIDER_FROM = 1e-11

# The mesh.  Surface: the cell mass grows 5% per cell from 4e-4 mg/cm^2 to the old surface cell
# (0.0216 mg/cm^2), which it keeps to 1.5 mg/cm^2 (the heat front at 2 ns is at 1.46).  Beyond:
# 0.3% per cell up to 0.3 mg/cm^2.  1760 cells.
MESH = (("mesh", "log_mass"), ("log_mass_first", 4e-7),
        ("log_mass_segments", ((1.05, 2.16e-5, 1.5e-3), (1.003, 3e-4, float("inf")))),
        # the first cell is ~1e-15 cm thick when it is heated: the first steps must be ~1e-23 s
        ("dt_first_frac", 2.5e-15))

# The seeded runs keep the old equal surface cells (the layer spans 14 of them at 0.1 ns, and the
# solver's own table does not resolve the first ~1e-3 mg/cm^2 of the profile it would seed there).
MESH_SEEDED = (("mesh", "log_mass"), ("log_mass_first", 2.16e-5),
               ("log_mass_segments", ((1.0, 2.16e-5, 1.5e-3), (1.003, 3e-4, float("inf")))))

# name -> (start, physics, target temperature [K])
RUNS = {
    "cold_1T_30K": ("cold", "1T", 30.0),
    "cold_1T_300K": ("cold", "1T", 300.0),
    "cold_2T_30K": ("cold", "2T", 30.0),
    "seeded_2T_30K": ("seeded", "2T", 30.0),
    "seeded_2T_300K": ("seeded", "2T", 300.0),
    "seeded_2T_30K_smeared": ("seeded", "2T", 30.0),
}
# the 2T cold start with the matter-radiation coupling multiplied by chi (the preset has chi = 1):
# toward the one-temperature limit
CHI = {"cold_2T_30K_chi10": 10.0, "cold_2T_30K_chi100": 100.0, "cold_2T_30K_chi1000": 1000.0}
RUNS.update({name: ("cold", "2T", 30.0) for name in CHI})
# IMC from the recommended analytic start (smeared shock, 30 K); run with --t-end, it is slow
IMC = {"seeded_IMC_30K_smeared_seed1": 1, "seeded_IMC_30K_smeared_seed2": 2}
RUNS.update({name: ("seeded", "IMC", 30.0) for name in IMC})
SMEARED = {"seeded_2T_30K_smeared", *IMC}          # the seeded shock spread over the scheme's shock width
RUN_TEXT = {
    "cold_1T_30K": "1T conduction, new mesh, 30 K",
    "cold_1T_300K": "1T conduction, new mesh, 300 K",
    "cold_2T_30K": "2T diffusion, new mesh, 30 K",
    "seeded_2T_30K": "2T diffusion, new mesh, shock on a face, 30 K",
    "seeded_2T_300K": "2T diffusion, new mesh, shock on a face, 300 K",
    "seeded_2T_30K_smeared": "2T diffusion, new mesh, smeared shock, 30 K",
}


RUN_TEXT.update({name: f"IMC seed {seed}, new mesh, smeared shock, 30 K" for name, seed in IMC.items()})
RUN_TEXT.update({name: rf"2T diffusion, $\chi$ = {chi:g}, new mesh, 30 K" for name, chi in CHI.items()})


def case_and_config(name: str, t_end: float = 2e-9):
    start, physics, T0 = RUNS[name]
    spec = ivd.HYDRO_CASES[BASE]
    case = replace(ivd._hydro_case(spec, t_end), T_initial_Kelvin=T0)
    cfg = dict(N=spec.N, store_every=2000, show_plot=False, energy_diagnostics=True,
               dt_ramp_max=spec.dt_max, bath_drive="paper", **ivd.HYDRO_MATCHED)
    cfg.update(dict(MESH_SEEDED if start == "seeded" else MESH))
    if name in CHI:
        case = replace(case, chi=CHI[name])
    if name in IMC:
        cfg.update(dict(spec.imc), radiation_solver="imc", imc_seed=IMC[name])
    if physics == "1T":
        # the solver's equation: one temperature, T(0, t) = T_s(t) on the surface, no bath
        case = replace(case, force_black="conduction")
        for key in ("material_update", "right_rad_bc"):
            cfg.pop(key)
    if start == "seeded":
        case = replace(case, initial_condition="analytic_ablation", t_sec_start=T_SEED)
        cfg["seed_shock_on_face"] = True
        cfg["seed_shock_smeared"] = name in SMEARED
    return case, cfg


def dense_side(m, T, p, n=30):
    """(heat front mass, pressure on its dense side): the T = 0.3 T_s point and the largest
    pressure in the n cells beyond it (test3_hydro_only.md section 2)."""
    mc = m - 0.5 * np.diff(m, prepend=0.0)
    level = ivd.HEAT_FRONT_LEVEL * T.max()
    below = np.flatnonzero(T < level)
    below = below[below > int(np.argmax(T))]
    if not below.size:
        return np.nan, np.nan
    j = int(below[0])
    return float(np.interp(level, [T[j], T[j - 1]], [mc[j], mc[j - 1]])), float(p[j:j + n].max())


def run(name: str, t_end: float = 2e-9, rerun: bool = False):
    from hydro_sim.problems.simulation_config import SimulationConfig
    from rad_hydro_sim.simulation.iterator import simulate_rad_hydro

    path = OUT / f"{name}.npz"
    if path.exists() and not rerun:
        print(f"  cached  {path.name}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    case, cfg = case_and_config(name, t_end)
    wall = time.time()
    _, _, meta, hist = simulate_rad_hydro(case, SimulationConfig(**cfg))
    wall = time.time() - wall
    snaps = [x * 1e-9 for x in SNAP_NS if x * 1e-9 <= t_end * (1 + 1e-9)] or [t_end]
    idx = [int(np.argmin(np.abs(hist.t - ts))) for ts in snaps]
    rho_init = ivd._initial_density(hist.m[0], case)
    front = np.array([dense_side(hist.m[k], hist.T_material[k], hist.p[k]) for k in range(hist.t.size)])
    shock = []
    for k in range(hist.t.size):
        excess = hist.rho[k] / rho_init - 1.0
        mc = hist.m[k] - 0.5 * np.diff(hist.m[k], prepend=0.0)
        shock.append(mc[int(np.flatnonzero(excess >= 0.5 * excess.max())[-1])])
    # frames for the slider: ~220 times, log-spaced from 0.01 ns (or the start) to the end
    targets = np.geomspace(max(SLIDER_FROM, float(hist.t[0]) * 1.0001), hist.t[-1], 220)
    fr = np.unique([int(np.argmin(np.abs(hist.t - x))) for x in targets])
    frames = {f"frames_{key}": getattr(hist, attr)[fr].astype(np.float32)
              for key, attr in (("rho", "rho"), ("p", "p"), ("u", "u"), ("T", "T_material"))}
    b = meta["energy_budget"]
    err = float(np.max(np.abs(b["relative_error"]))) if RUNS[name][1] != "1T" else np.nan
    np.savez(path, t=hist.t[idx], m=hist.m[idx], x=hist.x[idx], rho=hist.rho[idx], p=hist.p[idx],
             u=hist.u[idx], T_material=hist.T_material[idx], T_rad=hist.T[idx],
             series_t=hist.t, m_heat=front[:, 0], p_dense=front[:, 1], m_shock=np.array(shock),
             T_surface=hist.T_material[:, :5].max(axis=1), steps=len(b["t"]) - 1, wall=wall,
             max_energy_error=err, t_start=hist.t[0], frames_t=hist.t[fr], **frames)
    print(f"  ran     {path.name}: {hist.m.shape[1]} cells, {len(b['t']) - 1} steps, wall {wall:.0f} s", flush=True)


def slider(name: str):
    """The profiles of a run against the solver with a time slider (0.01 ns, or the start, to 2 ns).

    Density, pressure, velocity, temperature over the whole domain, and the entropy
    p / rho^gamma of the shocked matter, each against the solver's curve.  The axes are fixed: they are set
    once from the shocked matter of all frames and never change with the slider, so a spike
    can be followed in time.  Zoom with the toolbar; the zoom is kept while the slider moves.
    Keys: 'f' fits the y ranges to the mass window in view, 'x' switches the mass axis between
    log and linear, 'y' switches density / pressure / temperature between log and linear, and
    the left / right arrows step one frame.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    from rad_hydro_sim.verification import test3_hydro_only as h
    from rad_hydro_sim.verification import test3_hydro_only_figures as hf

    run = np.load(OUT / f"{name}.npz")
    s = h.solver()
    ps = s.shock_solver
    mc = hf._mc(run["m"][0])
    times = run["frames_t"]
    panels = (("rho", r"$\rho$ [g/cm$^3$]", 1.0, True), ("p", r"$P$ [Mbar]", 1e-12, True),
              ("u", r"$u$ [km/s]", 1e-5, False), ("T", r"$T$ [HeV]", 1.0 / ivd.K_PER_HEV, True),
              ("A", r"entropy $P/\rho^\gamma$ [cgs]", 1.0, True))

    def frame(k):
        t = float(times[k])
        m_f = float(s.heat_solver.ablated_mass(time=t))
        m_s = float(np.ravel(ps.shocked_mass(time=t))[0])
        shocked = (mc > 1.05 * m_f) & (mc < 0.985 * m_s)
        with np.errstate(all="ignore"):
            ref = ps.solve(mass=mc, time=t)
            A = run["frames_p"][k] / run["frames_rho"][k]**h.GAMMA
            A_ref = ref["pressure"] / ref["density"]**h.GAMMA
            try:                                      # the whole solution: ablated layer, shocked matter, target
                ok = mc > 2e-3 * m_f                  # the solver's table does not resolve the first cells
                sol = s.solve(mass=mc[ok], time=t)
                ana = {key: np.full(mc.size, np.nan) for key in ("rho", "p", "u", "T")}
                for key, name_ in (("rho", "density"), ("p", "pressure"), ("u", "velocity"), ("T", "temperature")):
                    ana[key][ok] = np.where(mc[ok] < m_s, sol[name_], np.nan)
            except AssertionError:
                ana = {key: np.full(mc.size, np.nan) for key in ("rho", "p", "u", "T")}
        ana["A"] = np.where((mc > 1.05 * m_f) & (mc < m_s), A_ref, np.nan)
        sim = dict(rho=run["frames_rho"][k], p=run["frames_p"][k], u=run["frames_u"][k], T=run["frames_T"][k],
                   A=np.where(shocked, A, np.nan))
        return t, sim, ana, shocked

    # fixed y ranges: the shocked matter of every 10th frame (and the last)
    lo = {key: np.inf for key, *_ in panels}
    hi = {key: -np.inf for key, *_ in panels}
    for k in sorted(set(range(0, len(times), 10)) | {len(times) - 1}):
        _, sim, ana, shocked = frame(k)
        for key, _, unit, _ in panels:
            vals = (ana if key == "A" else sim)[key][shocked] * unit     # entropy: the solver's range
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo[key], hi[key] = min(lo[key], vals.min()), max(hi[key], vals.max())

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.ravel()
    axes[5].axis("off")
    lines = {}
    t, sim, ana, _ = frame(len(times) - 1)
    for ax, (key, ylabel, unit, log) in zip(axes, panels):
        lines[key] = (ax.plot(mc * 1e3, sim[key] * unit, color="#2f6fc0", lw=1.5, marker=".", ms=2.5, label="Simulation")[0],
                      ax.plot(mc * 1e3, ana[key] * unit, color="#1a1a1a", lw=1.2, label="Analytic")[0])
        ax.set_xscale("log")
        ax.set_xlim(0.1, 330.0)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(lo[key] / 1.5, hi[key] * 1.5)
        else:
            pad = 0.08 * (hi[key] - lo[key])
            ax.set_ylim(lo[key] - pad, hi[key] + pad)
        ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]")
        ax.set_ylabel(ylabel)
        ivd.style_axes(ax)
    axes[0].legend(frameon=False, loc="lower left")
    title = fig.suptitle("", x=0.05, ha="left", fontweight="bold", fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.13, wspace=0.25, hspace=0.25)
    control = Slider(fig.add_axes([0.12, 0.04, 0.76, 0.03]), "frame", 0, len(times) - 1,
                     valinit=len(times) - 1, valstep=1)
    current = {}

    def update(_=None):
        k = int(control.val)
        t, sim, ana, _ = frame(k)
        current.update(sim=sim, ana=ana)
        for key, _, unit, _ in panels:
            lines[key][0].set_ydata(sim[key] * unit)
            lines[key][1].set_ydata(ana[key] * unit)
        title.set_text(rf"Test 3: Au ($\omega=+0.5$) · {RUN_TEXT[name]} · t = {t * 1e9:.4f} ns "
                       f"(frame {k + 1} of {len(times)})")
        control.valtext.set_text(f"{t * 1e9:.3f} ns")
        fig.canvas.draw_idle()

    def fit():
        for ax, (key, _, unit, _) in zip(axes, panels):
            x_lo, x_hi = ax.get_xlim()
            keep = (mc * 1e3 > x_lo) & (mc * 1e3 < x_hi)
            vals = np.concatenate((current["sim"][key][keep], current["ana"][key][keep])) * unit
            vals = vals[np.isfinite(vals) & ((vals > 0) | (ax.get_yscale() != "log"))]
            if vals.size:
                if ax.get_yscale() == "log":
                    ax.set_ylim(vals.min() / 1.1, vals.max() * 1.1)
                else:
                    pad = 0.08 * max(vals.max() - vals.min(), 1e-6 * abs(vals.max()))
                    ax.set_ylim(vals.min() - pad, vals.max() + pad)

    def on_key(event):
        if event.key == "f":
            fit()
        elif event.key == "x":
            for ax in axes[:5]:
                ax.set_xscale("linear" if ax.get_xscale() == "log" else "log")
        elif event.key == "y":
            for ax, (_, _, _, log) in zip(axes, panels):
                if log:
                    ax.set_yscale("linear" if ax.get_yscale() == "log" else "log")
            fit()
        elif event.key in ("left", "right"):
            control.set_val(int(np.clip(control.val + (1 if event.key == "right" else -1), 0, len(times) - 1)))
            return
        else:
            return
        fig.canvas.draw_idle()

    control.on_changed(update)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()
    return fig, control


def run_all(jobs: int):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    OUT.mkdir(parents=True, exist_ok=True)

    def launch(name):
        with open(OUT / f"{name}.log", "w", encoding="utf-8") as f:
            return subprocess.run([sys.executable, "-m", "rad_hydro_sim.verification.test3_on_the_solver",
                                   "--run", name], stdout=f, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        print(dict(zip(RUNS, pool.map(launch, RUNS))))


def main(argv=None):
    import logging
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", choices=list(RUNS))
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--t-end", type=float, default=2e-9)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--slider", choices=list(RUNS), help="the profiles of a run with a time slider")
    args = ap.parse_args(argv)
    for name in args.run or ():
        run(name, t_end=args.t_end, rerun=args.rerun)
    if args.run_all:
        run_all(args.jobs)
    if args.slider:
        slider(args.slider)
    if args.figures or args.table:
        from rad_hydro_sim.verification import test3_on_the_solver_figures as figs
        if args.figures:
            figs.all_figures()
        if args.table:
            figs.write_table()


if __name__ == "__main__":
    main()

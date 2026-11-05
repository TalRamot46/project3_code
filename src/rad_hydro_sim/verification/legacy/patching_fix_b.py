"""LEGACY (numerical): fix B of md/phase_c/why_the_solver_differs.md Sec. 3.3.

Not an analytic benchmark: every curve here involves ``menahem_new.legacy.moving_front_shock``,
a 1-D hydro integration of the target.  Kept to reproduce md/phase_c/legacy/patching_poc_fix_B.md
and the figures in md/phase_c/figures/legacy/fix_B/.  The analytic patchings live in
``rad_hydro_sim.verification.patching_poc``.

    uv run python -m rad_hydro_sim.verification.legacy.patching_fix_b --cases test4_full   # tables
    uv run python -m rad_hydro_sim.verification.legacy.patching_fix_b --cases test4_full --verify
    uv run python -m rad_hydro_sim.verification.legacy.patching_fix_b --figures
    uv run python -m rad_hydro_sim.verification.legacy.patching_fix_b --cases test4_full --profiles

Patchings: original, B_massless, B_massive (moving-front target), C_pressure, C_impulse,
and B_simdrive (B driven by the diffusion run's own front history).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from rad_hydro_sim.verification.patching_poc import (
    OUT, SIM_OUT, TIMES, patched_solvers, register_omega_case, sim_fronts, solver_for, table)

PATCHINGS = ("original", "B_massless", "B_massive", "C_pressure", "C_impulse", "B_simdrive")
LEGACY_OUT = OUT / "legacy"
FIG = ivd.REPO / "md" / "phase_c" / "figures" / "legacy" / "fix_B"


def patched_solvers_b(label: str, n_cells: int = 3000):
    """The analytic patchings plus B with both front closures."""
    from menahem_new.legacy.moving_front_shock import PatchedAblationSolver

    out = patched_solvers(label)
    base = solver_for(label)
    for front in ("massless", "massive"):
        out[f"B_{front}"] = PatchedAblationSolver(base, times=TIMES, front=front, n_cells=n_cells)
    return out


class SimDriven:
    """Diagnostic: the moving-front target driven by a simulation's own front history.

    m_f(t) and the pressure at the front come from the run's front series (the pressure
    there already contains whatever the ablated layer does), so the front is "massless".
    Agreement with the run's shock tests the target model alone, free of the rigid heat
    solution's m_f(t) and p_f(t).
    """

    def __init__(self, label: str, run, n_cells: int = 3000):
        from menahem_new.legacy.moving_front_shock import MovingFrontShock, shock_mass

        ps = solver_for(label).shock_solver
        t, mh, pa = run["series_t"], run["m_heat"], run["p_abl"]
        ok = np.isfinite(mh) & np.isfinite(pa) & (t > 0)
        t, mh, pa = t[ok], mh[ok], pa[ok]
        # before the first frame with a resolved front: the power laws through the first 5%
        early = t < t[0] + 0.05 * (t[-1] - t[0])
        km, cm = np.polyfit(np.log(t[early]), np.log(mh[early]), 1)
        kp, cp = np.polyfit(np.log(t[early]), np.log(pa[early]), 1)
        m_f = lambda tt: float(np.interp(tt, t, mh)) if tt >= t[0] else float(np.exp(cm) * tt**km)
        p_f = lambda tt: float(np.interp(tt, t, pa)) if tt >= t[0] else float(np.exp(cp) * tt**kp)
        m_max = 1.4 * float(np.ravel(ps.shocked_mass(time=TIMES[-1]))[0])
        self.shock = MovingFrontShock(rho0=ps.rho0, omega=ps.omega, gamma=ps.gamma, m_front=m_f, p_front=p_f,
                                      m_max=m_max, n_cells=n_cells, front="massless")
        snaps, _ = self.shock.run(TIMES)
        self.values = {}
        for tt, sn in zip(TIMES, snaps):
            M = sn["M"]
            rho_init = np.diff(M) / np.diff(self.shock.initial_position(M))
            self.values[tt] = dict(m_heat=m_f(tt), m_shock=shock_mass(M, sn["v"], rho_init), p_abl=p_f(tt))

    def front_values(self, t):
        return self.values[t]


def verify_moving_front(label: str, n_cells: int = 3000) -> str:
    """The moving-front target driven by PistonShock's own pressure at the cut, p_PS(m_f(t), t),
    must reproduce PistonShock beyond the cut (the restriction of PistonShock to m > m_f solves
    the same equations with the same boundary data).  Returns one line of shock errors."""
    from menahem_new.legacy.moving_front_shock import MovingFrontShock, shock_mass

    base = solver_for(label)
    hs, ps = base.heat_solver, base.shock_solver
    m_f = lambda t: float(hs.ablated_mass(time=t))
    p_cut = lambda t: float(ps.solve(mass=np.array([m_f(t), 1.0001 * m_f(t)]), time=t)["pressure"][0])
    m_s = lambda t: float(np.ravel(ps.shocked_mass(time=t))[0])
    sh = MovingFrontShock(rho0=ps.rho0, omega=ps.omega, gamma=ps.gamma, m_front=m_f, p_front=p_cut,
                          m_max=1.4 * m_s(TIMES[-1]), n_cells=n_cells, front="massless")
    snaps, _ = sh.run(TIMES)
    errs = [shock_mass(sn["M"], sn["v"], np.diff(sn["M"]) / np.diff(sh.initial_position(sn["M"]))) / m_s(sn["t"]) - 1
            for sn in snaps]
    return f"{label}: moving front driven by p_PS(m_f) vs PistonShock, shock at " +         ", ".join(f"{t * 1e9:g} ns {100 * e:+.2f}%" for t, e in zip(TIMES, errs))


# reference palette (dataviz skill), categorical slots in fixed order; ink for the simulations
STYLE = {
    "original": dict(color="#2a78d6", ls="-", marker="o"),
    "B_massless": dict(color="#eb6834", ls="-", marker="s"),
    "B_massive": dict(color="#1baf7a", ls="-", marker="^"),
    "C_pressure": dict(color="#eda100", ls="--", marker="v"),
    "C_impulse": dict(color="#e87ba4", ls="--", marker="D"),
    "B_simdrive": dict(color="#555550", ls=":", marker="x"),
}
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e4e3dd"
# the gold cases of the omega scan, and whether their front series is the snapshot's definition
GOLD_SCAN = {-0.75: "omegam075", -0.5: "test4_full", -0.25: "omegam025", 0.0: "test1_full",
             0.25: "omegap025_n2000", 0.5: "test3_smooth"}
# omega = +0.25 at N = 400 leaves ~8 cells in the ablated layer (heat front +7%, IMC and diffusion
# 13% apart); N = 2000 resolves it.
CLEAN_SERIES = {"test4_full", "test3_smooth", "omegam075", "omegam050", "omegam025", "omegap025_n2000"}


def _axes_style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_omega_scan(path: Path, t: float = 2e-9):
    """Left: shock, diffusion vs each patching, against omega.  Right: the heat side,
    the run's heat front and front pressure against the rigid heat solution's."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for w, label in GOLD_SCAN.items():
        if label.startswith("omega"):
            register_omega_case(w, N=int(label.split("_n")[1]) if "_n" in label else 400)
        if not (SIM_OUT / f"{label}_diffusion.npz").exists():
            continue
        solvers = patched_solvers_b(label)
        if label in CLEAN_SERIES:
            solvers["B_simdrive"] = SimDriven(label, np.load(SIM_OUT / f"{label}_diffusion.npz"))
        d = sim_fronts(label)["diffusion"][t]
        ref = {p: solvers[p].front_values(t) for p in solvers}
        rows.append((w, d, ref))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for p in PATCHINGS:
        pts = [(w, 100 * (d["m_shock"] / ref[p]["m_shock"] - 1)) for w, d, ref in rows if p in ref]
        if pts:
            a1.plot(*zip(*pts), lw=2, ms=7, label=p.replace("_", " "), **STYLE[p])
    a1.axhline(0, color=MUTED, lw=1)
    a1.set_xlabel(r"$\omega$  (gold, 1 HeV, $\tau=0$)", color=INK)
    a1.set_ylabel("shock: diffusion vs patching [%]", color=INK)
    a1.set_title(f"Shock front at {t * 1e9:g} ns", color=INK, loc="left")
    a1.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    for q, name, st in (("m_heat", "heat front", dict(color="#2a78d6", marker="o")),
                        ("p_abl", "front pressure", dict(color="#eb6834", marker="s"))):
        pts = [(w, 100 * (d[q] / ref["original"][q] - 1)) for w, d, ref in rows]
        a2.plot(*zip(*pts), lw=2, ms=7, label=name, **st)
    a2.axhline(0, color=MUTED, lw=1)
    a2.set_xlabel(r"$\omega$", color=INK)
    a2.set_ylabel("diffusion vs rigid heat solution [%]", color=INK)
    a2.set_title("What the front really does (the heat side)", color=INK, loc="left")
    a2.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    for ax in (a1, a2):
        _axes_style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


def plot_profiles(label: str, path: Path, t: float = 2e-9, shown=("original", "B_massless", "B_massive", "C_impulse")):
    """Pressure and velocity against mass at t: runs vs patchings (the jump at m_f)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solvers = patched_solvers_b(label)
    run = np.load(SIM_OUT / f"{label}_diffusion.npz")
    k = int(np.argmin(np.abs(run["t"] - t)))
    m = run["m"][k]
    mc = m - 0.5 * np.diff(m, prepend=0.0)
    imc = [np.load(SIM_OUT / f"{label}_imc{s}.npz") for s in (1, 2, 3) if (SIM_OUT / f"{label}_imc{s}.npz").exists()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, unit, ylab in ((axes[0], "p", 1e-12, "pressure [Mbar]"), (axes[1], "u", 1e-5, "velocity [km/s]")):
        ax.plot(mc * 1e3, run[key][k] * unit, color=INK, lw=2.5, label="diffusion")
        if imc:
            ax.plot(mc * 1e3, imc[0][key][k] * unit, color=MUTED, lw=1.2, ls="--", label="IMC")
        for p in shown:
            sol = solvers[p].solve(mass=m, time=float(run["t"][k]))
            val = sol["pressure" if key == "p" else "velocity"]
            st = dict(STYLE[p]); st.pop("marker")
            ax.plot(m * 1e3, np.asarray(val) * unit, lw=1.6, label=p.replace("_", " "), **st)
        m_f = solvers["original"].front_values(float(run["t"][k]))["m_heat"]
        ax.axvline(m_f * 1e3, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, 1.25 * solvers["original"].front_values(float(run["t"][k]))["m_shock"] * 1e3)
        if key == "u":
            ax.set_ylim(-60, 1.6 * np.nanmax(run["u"][k][mc > m_f]) * unit)
        ax.set_xlabel(r"mass [mg/cm$^2$]", color=INK)
        ax.set_ylabel(ylab, color=INK)
        _axes_style(ax)
    axes[0].set_title(f"{ivd.HYDRO_CASES[label].title}, {t * 1e9:g} ns", color=INK, loc="left")
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


class _AtSnapshot:
    """A patching evaluated at the integrated time nearest the requested one (within 1%):
    the runs' stored frames sit up to ~0.3% off the nominal times."""

    def __init__(self, obj):
        self.obj = obj

    def solve(self, *, mass, time):
        t = float(TIMES[np.argmin(np.abs(TIMES - time))])
        assert abs(t - time) <= 1e-2 * time, f"no patched solution near t = {time:g}"
        return self.obj.solve(mass=mass, time=t)


def plot_patched_profiles(label: str, path: Path, log_axis=None):
    """imc_vs_diffusion's <label>_profiles figure (rho, p, u, T at 1, 1.5, 2 ns: diffusion,
    IMC) with the corrected patchings on top: B massless (black) and C impulse (dotted);
    the original AblationSolver gray dashed."""
    import matplotlib
    matplotlib.use("Agg")

    solvers = patched_solvers_b(label)
    patched = [("Analytic, B: moving front (massless)", _AtSnapshot(solvers["B_massless"]),
                dict(color="black", lw=2.0)),
               ("Analytic, C: impulse-matched", _AtSnapshot(solvers["C_impulse"]),
                dict(color="black", lw=1.6, linestyle=":"))]
    ivd.plot_hydro_profiles(SIM_OUT, label, (1, 2, 3), path, solver=solver_for(label), patched=patched,
                            log_axis=log_axis)
    print(f"  wrote {path}", flush=True)


def main(argv=None):
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["test4_full"])
    ap.add_argument("--omega-scan", nargs="*", type=float, default=[])
    ap.add_argument("--N", type=int, default=400, help="cells of the omega-scan diffusion runs")
    ap.add_argument("--verify", action="store_true", help="the moving-front check against PistonShock")
    ap.add_argument("--figures", action="store_true", help="the omega-scan and Test 4 figures")
    ap.add_argument("--profiles", action="store_true", help="profile figures with B and C on top")
    ap.add_argument("--linear", action="store_true", help="--profiles on a linear mass axis")
    args = ap.parse_args(argv)
    labels = list(args.cases) + [register_omega_case(w, N=args.N) for w in args.omega_scan]
    if args.verify:
        for label in labels:
            print(verify_moving_front(label), flush=True)
        return
    if args.figures:
        plot_omega_scan(FIG / "patching_omega_scan.png")
        plot_profiles("test4_full", FIG / "patching_test4_profiles.png")
        return
    if args.profiles:
        for label in labels:
            name = f"{label}_profiles{'_linear' if args.linear else ''}.png"
            plot_patched_profiles(label, FIG / "profiles" / name, log_axis=False if args.linear else None)
        return
    LEGACY_OUT.mkdir(parents=True, exist_ok=True)
    for label in labels:
        solvers = patched_solvers_b(label)
        solvers["B_simdrive"] = SimDriven(label, np.load(SIM_OUT / f"{label}_diffusion.npz"))
        text = table(label, solvers, sim_fronts(label), PATCHINGS)
        print(text, flush=True)
        (LEGACY_OUT / f"{label}_table.md").write_text(text + "\n")


if __name__ == "__main__":
    main()

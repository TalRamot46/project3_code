"""Proof of concept: better patching of the semi-analytic ablation solution (fix B / C of
md/phase_c/why_the_solver_differs.md Sec. 3.3), against the cached diffusion and IMC runs.

    uv run python -m rad_hydro_sim.verification.patching_poc --cases test4_full test1_full
    uv run python -m rad_hydro_sim.verification.patching_poc --omega-scan -0.75 -0.25 0.25

Patchings compared (menahem_new.ablation_patching):
  original     AblationSolver: PistonShock with the piston at m = 0, cut at m_f.
  B_massless   numerical target driven at the moving front m_f(t) with p_f.
  B_massive    the same, with the ablated layer riding on the front (p_b = p_f - m_f du_f/dt).
  C_pressure   PistonShock with its amplitude matched so that p = p_f at the cut m_f (quasi-static).
  C_impulse    PistonShock with its amplitude matched to the momentum the moving (massless)
               front has delivered beyond the cut, marched in time.
  B_simdrive   diagnostic: B driven by the diffusion run's own front m_heat(t), p_abl(t).

The omega scan adds Test 4 variants (gold, 1 HeV, tau = 0) with other omegas, run with
diffusion only on a coarser uniform-mass mesh; the runs are cached in --out like the others.
"""
from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from rad_hydro_sim.verification import imc_vs_diffusion as ivd

OUT = ivd.REPO / "results" / "patching_poc"
SIM_OUT = ivd.DEFAULT_OUT
TIMES = np.array([0.5, 1.0, 1.5, 2.0]) * 1e-9
PATCHINGS = ("original", "B_massless", "B_massive", "C_pressure", "C_impulse", "B_simdrive")


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

    path = OUT / f"solver_{label}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    solver = AblationSolver(**_ablation_kwargs_from_case(case))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(solver, f)
    return solver


def solver_for(label: str):
    # the omega-scan cases at other N share their omega's solver
    return _solver_for_case(ivd._hydro_case(ivd.HYDRO_CASES[label]), label.split("_n")[0])


def patched_solvers(label: str, n_cells: int = 3000):
    """{patching: object with front_values(t) and solve(mass, time)} for one case."""
    from menahem_new.ablation_patching import MatchedPistonPatch, PatchedAblationSolver

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

    out = {"original": Original()}
    for front in ("massless", "massive"):
        out[f"B_{front}"] = PatchedAblationSolver(base, times=TIMES, front=front, n_cells=n_cells)
    out["C_pressure"] = MatchedPistonPatch(base, front="massless", match="pressure")
    out["C_impulse"] = MatchedPistonPatch(base, front="massless", match="impulse", t_end=TIMES[-1])
    return out


class SimDriven:
    """Diagnostic: the moving-front target driven by a simulation's own front history.

    m_f(t) and the pressure at the front come from the run's front series (the pressure
    there already contains whatever the ablated layer does), so the front is "massless".
    Agreement with the run's shock tests the target model alone, free of the rigid heat
    solution's m_f(t) and p_f(t).
    """

    def __init__(self, label: str, run, n_cells: int = 3000):
        from menahem_new.ablation_patching import MovingFrontShock, shock_mass

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
    from menahem_new.ablation_patching import MovingFrontShock, shock_mass

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


def sim_fronts(label: str):
    """{run name: {t: front values}} of the cached diffusion / IMC runs."""
    runs = {}
    for tag in ["diffusion", "imc1", "imc2", "imc3"]:
        path = SIM_OUT / f"{label}_{tag}.npz"
        if path.exists():
            run = np.load(path)
            runs[tag] = {t: ivd.front_values(run, label, t) for t in TIMES}
    return runs


def table(label: str, solvers, sims) -> str:
    rows = [f"### {label}", "",
            "| t [ns] | quantity | diffusion | IMC | " + " | ".join(PATCHINGS) + " |",
            "|---|---|---|---|" + "---|" * len(PATCHINGS)]
    for t in TIMES:
        ref = {p: solvers[p].front_values(t) for p in PATCHINGS}
        for q, unit in (("m_heat", 1e3), ("m_shock", 1e3), ("p_abl", 1e-12)):
            d = sims["diffusion"][t][q]
            imc = [s[t][q] for k, s in sims.items() if k.startswith("imc")]
            cells = [f"{d * unit:.4g}", f"{np.mean(imc) * unit:.4g}" if imc else "-"]
            cells += [f"{ref[p][q] * unit:.4g} ({100 * (d / ref[p][q] - 1):+.1f}%)" for p in PATCHINGS]
            rows.append(f"| {t * 1e9:.1f} | {q} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("(m in mg/cm^2, p in Mbar; the % is diffusion vs. that patching)")
    return "\n".join(rows)


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
        solvers = patched_solvers(label)
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

    solvers = patched_solvers(label)
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


def plot_patched_profiles(label: str, path: Path):
    """imc_vs_diffusion's <label>_profiles figure (rho, p, u, T at 1, 1.5, 2 ns: diffusion,
    IMC) with the corrected patchings on top: B massless (black) and C impulse (dotted);
    the original AblationSolver gray dashed."""
    import matplotlib
    matplotlib.use("Agg")

    solvers = patched_solvers(label)
    patched = [("Analytic, B: moving front (massless)", _AtSnapshot(solvers["B_massless"]),
                dict(color="black", lw=2.0)),
               ("Analytic, C: impulse-matched", _AtSnapshot(solvers["C_impulse"]),
                dict(color="black", lw=1.6, linestyle=":"))]
    ivd.plot_hydro_profiles(SIM_OUT, label, (1, 2, 3), path, solver=solver_for(label), patched=patched)
    print(f"  wrote {path}", flush=True)


def main(argv=None):
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["test4_full"])
    ap.add_argument("--omega-scan", nargs="*", type=float, default=[])
    ap.add_argument("--N", type=int, default=400, help="cells of the omega-scan diffusion runs")
    ap.add_argument("--seeds", nargs="*", type=int, default=[], help="IMC seeds of the omega-scan runs")
    ap.add_argument("--verify", action="store_true", help="the moving-front check against PistonShock")
    ap.add_argument("--figures", action="store_true", help="the omega-scan and Test 4 profile figures")
    ap.add_argument("--profiles", action="store_true",
                    help="each case's profile figure with the corrected patchings, md/phase_c/figures/patched/")
    ap.add_argument("--run-only", action="store_true", help="only run the omega-scan simulations")
    args = ap.parse_args(argv)
    labels = list(args.cases)
    for w in args.omega_scan:
        label = register_omega_case(w, N=args.N)
        ivd.run_hydro(SIM_OUT, label, seeds=tuple(args.seeds), physical=False)
        labels.append(label)
    if args.run_only:
        return
    if args.verify:
        for label in labels:
            print(verify_moving_front(label), flush=True)
        return
    if args.profiles:
        for label in labels:
            plot_patched_profiles(label, ivd.REPO / "md" / "phase_c" / "figures" / "patched" / f"{label}_profiles.png")
        return
    if args.figures:
        FIG = ivd.REPO / "md" / "phase_c" / "figures"
        plot_omega_scan(FIG / "patching_omega_scan.png")
        plot_profiles("test4_full", FIG / "patching_test4_profiles.png")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for label in labels:
        solvers = patched_solvers(label)
        solvers["B_simdrive"] = SimDriven(label, np.load(SIM_OUT / f"{label}_diffusion.npz"))
        text = table(label, solvers, sim_fronts(label))
        print(text, flush=True)
        (OUT / f"{label}_table.md").write_text(text + "\n")


if __name__ == "__main__":
    main()

"""IMC vs. diffusion verification: runs, tables and figures (plan milestone M10, radiation-only part).

One command regenerates the Phase B results of md/phase_b/phase_b_results.md:

    uv run python -m rad_hydro_sim.verification.imc_vs_diffusion
    uv run python -m rad_hydro_sim.verification.imc_vs_diffusion --figures-dir md/phase_b/figures

Cases (all radiation-only, static mesh):
  t1       SiO2 foam (PRESET_SIO2_FOAM_RADIATION_ONLY), 100 cells, 3 ns. Outgoing flux of
           diffusion (original and Phase B flags), IMC through rad_hydro_sim, the standalone
           IMC (imc.problems.sio2) and the S_N twin (src/sn/sn.py).
  test1    paper Test 1 (PRESET_PAPER_TEST1_RADIATION_ONLY), domain cut to 1e-2 g/cm^2, 400 cells.
  test2    paper Test 2 (PRESET_PAPER_TEST2_RADIATION_ONLY), same mesh.
  thin100  Test 1 with opacity / 100 (g x 100), full preset domain, 400 cells.

Every run is cached as an .npz in --out (default results/imc_vs_diffusion) and reused unless
--rerun is given, so the tables and figures can be redone without rerunning the physics.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "results" / "imc_vs_diffusion"
K_PER_HEV = 1.16045e6
BLOCK_NS = 0.05                     # T1 flux averaging blocks
T1_END_NS = 3.0

# Configurations of the Phase B runs.
DIFFUSION_PHASE_B = dict(material_update="conservative", right_rad_bc="marshak_vacuum")
T1_COMMON = dict(N=100, store_every=10**9, show_plot=False, energy_diagnostics=True, dt_ramp_max=1e-12)
T1_IMC = dict(radiation_solver="imc", imc_n_emission=2500, imc_n_bath=4000, imc_max_census=120000)
STANDALONE_SEEDS = (1729, 2718, 3141)


@dataclass(frozen=True)
class GoldCase:
    preset: str          # attribute name in presets_config
    title: str
    N: int = 400
    dt_max: float = 2e-12
    t_end: float = 2e-9
    g_scale: float = 1.0
    x_max_mass: float | None = None   # domain as areal mass [g/cm^2]; None keeps the preset's


GOLD_CASES = {
    "test1": GoldCase("PRESET_PAPER_TEST1_RADIATION_ONLY", r"Test 1 ($\tau=0$), radiation only", x_max_mass=1e-2),
    "test2": GoldCase("PRESET_PAPER_TEST2_RADIATION_ONLY", r"Test 2 ($\tau=158/1285$), radiation only", x_max_mass=1e-2),
    "thin100": GoldCase("PRESET_PAPER_TEST1_RADIATION_ONLY", r"Test 1 setup, opacity $\div$100, radiation only",
                        g_scale=100.0),
}
ALL_CASES = ("t1", *GOLD_CASES)


def _cached(path: Path, rerun: bool) -> bool:
    if path.exists() and not rerun:
        print(f"  cached  {path.name}")
        return True
    return False


def _simulate(case, **cfg_opts):
    from hydro_sim.problems.simulation_config import SimulationConfig
    from rad_hydro_sim.simulation.iterator import simulate_rad_hydro

    t0 = time.time()
    _, state, meta, hist = simulate_rad_hydro(case, SimulationConfig(**cfg_opts))
    return state, meta, hist, time.time() - t0


# --------------------------------------------------------------------------- T1: runs
def _norm_cgs() -> float:
    """sigma_SB (165 eV)^4 in rad_hydro_sim's CGS units."""
    from rad_hydro_sim.simulation.radiation_step import a_Kelvin, c, eV_to_erg, k_B
    return 0.25 * a_Kelvin * c * (165.0 * eV_to_erg / k_B) ** 4


def run_t1(out: Path, seeds, rerun=False, standalone=True, sn_twin=True, end_ns=T1_END_NS):
    """T1 runs. Each file stores t_edges [ns] and the cumulative normalized outflow
    W(t) = int F_out / [sigma_SB (165 eV)^4] dt [ns], so block averages are exact energy averages."""
    from rad_hydro_sim.problems import presets_config as P

    case = replace(P.PRESET_TEST_CASES[P.PRESET_SIO2_FOAM_RADIATION_ONLY], t_sec_end=end_ns * 1e-9)
    norm = _norm_cgs()

    def rad_hydro_run(label, **opts):
        path = out / f"t1_{label}.npz"
        if _cached(path, rerun):
            return
        state, meta, _, wall = _simulate(case, **T1_COMMON, **opts)
        b = meta["energy_budget"]
        t_ns = np.asarray(b["t"]) * 1e9
        W = np.asarray(b["rad_out_right"]) / norm * 1e9
        np.savez(path, t_edges=t_ns, W=W, energy_error=np.asarray(b["relative_error"]),
                 T_final=state.T_material, wall=wall)
        print(f"  ran     {path.name:28s} wall {wall:6.1f} s  max|energy err| "
              f"{np.max(np.abs(b['relative_error'])):.1e}")

    rad_hydro_run("diffusion_original")
    rad_hydro_run("diffusion_phaseB", **DIFFUSION_PHASE_B)
    for s in seeds:
        rad_hydro_run(f"imc_seed{s}", **T1_IMC, imc_seed=s)

    if standalone:
        from imc.core import stepper
        from imc.problems import sio2 as problem

        settings = stepper.IMCSettings(n_emission=2500, n_bath=4000, max_census=120000)
        bath = problem.make_bath()
        norm_s = 0.25 * problem.C * problem.E100 * 1.65**4
        for s in STANDALONE_SEEDS:
            path = out / f"t1_standalone_seed{s}.npz"
            if _cached(path, rerun):
                continue
            t0 = time.time()
            r = problem.run_one(s, 0.001, end_ns, settings, bath)
            edges = _edges_from_midpoints(r["time_ns"], end_ns)
            W = np.concatenate(([0.0], np.cumsum(r["Fout"] * np.diff(edges)))) / norm_s
            np.savez(path, t_edges=edges, W=W, energy_error=r["energy_error"], wall=time.time() - t0)
            print(f"  ran     {path.name:28s} wall {time.time() - t0:6.1f} s")

    if sn_twin:
        from sn import sn

        path = out / "t1_sn.npz"
        if not _cached(path, rerun):
            t0 = time.time()
            drive = sn.Drive(Path(sn.__file__).resolve().parent.parent / "imc" / "data" / "back_pulse.txt")
            r = sn.run_one(drive, 0.001, end_ns, 16, "double-gauss", "current", 1e-10, 1000)
            edges = _edges_from_midpoints(r["time_ns"], end_ns)
            norm_n = sn.SIGMA_SB * (165.0 * sn.EV_TO_K) ** 4
            W = np.concatenate(([0.0], np.cumsum(r["Fout"] * np.diff(edges)))) / norm_n
            np.savez(path, t_edges=edges, W=W, energy_error=r["energy_error"], wall=time.time() - t0)
            print(f"  ran     {path.name:28s} wall {time.time() - t0:6.1f} s")


def _edges_from_midpoints(mid, end):
    """Step edges of a run with uniform steps (the last one possibly shorter)."""
    edges = np.empty(mid.size + 1)
    edges[0] = 0.0
    for k, m in enumerate(mid):
        edges[k + 1] = min(2.0 * m - edges[k], end)
    return edges


# --------------------------------------------------------------------------- T1: analysis
def _blocks(path: Path, edges):
    d = np.load(path)
    W = np.interp(edges, d["t_edges"], d["W"])
    return np.diff(W) / np.diff(edges)


def t1_blocks(out: Path, seeds, end_ns=T1_END_NS):
    edges = np.arange(0.0, end_ns + 0.5 * BLOCK_NS, BLOCK_NS)
    tc = 0.5 * (edges[:-1] + edges[1:])
    res = {"tc": tc}
    imc = np.array([_blocks(out / f"t1_imc_seed{s}.npz", edges) for s in seeds])
    res["IMC via rad_hydro_sim"] = imc.mean(0)
    res["IMC s.e."] = imc.std(0, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else np.zeros_like(tc)
    stand = [out / f"t1_standalone_seed{s}.npz" for s in STANDALONE_SEEDS]
    if all(p.exists() for p in stand):
        res["standalone IMC"] = np.mean([_blocks(p, edges) for p in stand], axis=0)
    if (out / "t1_sn.npz").exists():
        res["S_N twin"] = _blocks(out / "t1_sn.npz", edges)
    res["diffusion, Phase B flags"] = _blocks(out / "t1_diffusion_phaseB.npz", edges)
    res["diffusion, original"] = _blocks(out / "t1_diffusion_original.npz", edges)
    return res


def t1_table(res) -> str:
    lines = ["| Solver | Breakout (flux >= 0.01) | Peak |", "|---|---|---|"]
    tc = res["tc"]
    for key, y in res.items():
        if key in ("tc", "IMC s.e."):
            continue
        cross = np.flatnonzero(y >= 0.01)
        k = int(np.argmax(y))
        bo = f"{tc[cross[0]]:.3f} ns" if cross.size else "-"
        lines.append(f"| {key} | {bo} | {y[k]:.3f} at {tc[k]:.3f} ns |")
    return "\n".join(lines)


def plot_t1(res, path: Path, end_ns=T1_END_NS):
    import matplotlib.pyplot as plt

    tc = res["tc"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    m, se = res["IMC via rad_hydro_sim"], res["IMC s.e."]
    ax.fill_between(tc, m - 2 * se, m + 2 * se, color="tab:blue", alpha=0.3)
    ax.plot(tc, m, color="tab:blue", lw=1.8, label=r"IMC via rad_hydro_sim (mean of seeds, $\pm$2 s.e.)")
    styles = {"standalone IMC": ("--", "tab:cyan", 1.2), "S_N twin": (":", "black", 1.5),
              "diffusion, Phase B flags": ("-", "tab:red", 1.5), "diffusion, original": ("-", "tab:orange", 1.2)}
    labels = {"standalone IMC": "standalone IMC (imc.problems.sio2)", "S_N twin": r"S$_N$ twin (src/sn/sn.py)"}
    for key, (ls, col, lw) in styles.items():
        if key in res:
            ax.plot(tc, res[key], ls, color=col, lw=lw, label=labels.get(key, key))
    ax.set_xlim(0.8, end_ns)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("time [ns]")
    ax.set_ylabel(r"$F_{\rm out}/[\sigma_{\rm SB}(165\,{\rm eV})^4]$")
    ax.set_title(rf"T1: SiO$_2$ foam, 1 mm, 0.05 g/cm$^3$ ({BLOCK_NS:g} ns blocks)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- gold: runs
def _gold_case(spec: GoldCase, t_end=None):
    from rad_hydro_sim.problems import presets_config as P

    case = replace(P.PRESET_TEST_CASES[getattr(P, spec.preset)], t_sec_end=t_end or spec.t_end)
    case = replace(case, g_Kelvin=case.g_Kelvin * spec.g_scale)
    if spec.x_max_mass is not None:
        case = replace(case, x_max=spec.x_max_mass / case.rho0)
    return case


def run_gold(out: Path, label: str, seeds, rerun=False, N=None, t_end=None):
    spec = GOLD_CASES[label]
    case = _gold_case(spec, t_end)
    snap = np.array([0.25, 0.5, 1.0]) * case.t_sec_end
    common = dict(N=N or spec.N, store_every=1, show_plot=False, energy_diagnostics=True,
                  dt_ramp_max=spec.dt_max, **DIFFUSION_PHASE_B)

    def run(tag, **opts):
        path = out / f"{label}_{tag}.npz"
        if _cached(path, rerun):
            return
        _, meta, hist, wall = _simulate(case, **common, **opts)
        idx = [int(np.argmin(np.abs(hist.t - ts))) for ts in snap]
        b = meta["energy_budget"]
        E_track = meta["imc_E_rad_track"][idx] if "imc_E_rad_track" in meta else hist.E_rad[idx]
        np.savez(path, t=hist.t[idx], m=hist.m[idx], x=hist.x[idx], T_material=hist.T_material[idx],
                 T_rad=hist.T[idx], E_rad=hist.E_rad[idx], E_rad_track=E_track,
                 energy_error=np.asarray(b["relative_error"]), wall=wall)
        print(f"  ran     {path.name:28s} wall {wall:6.1f} s  steps {len(b['t']) - 1}  "
              f"max|energy err| {np.max(np.abs(b['relative_error'])):.1e}")

    run("diffusion")
    for s in seeds:
        run(f"imc{s}", radiation_solver="imc", imc_seed=s)


# --------------------------------------------------------------------------- gold: analysis
def _front(m, T, level):
    """Mass where T first drops below level, linear between cells."""
    j = np.flatnonzero(T < level)
    if not j.size or j[0] == 0:
        return np.nan
    j = j[0]
    return np.interp(level, [T[j], T[j - 1]], [m[j], m[j - 1]])


def _load_gold(out: Path, label: str, seeds):
    d = np.load(out / f"{label}_diffusion.npz")
    imc = [np.load(out / f"{label}_imc{s}.npz") for s in seeds]
    return d, imc


def gold_table(out: Path, label: str, seeds, front_level=0.5, hot_level=0.3) -> str:
    d, imc = _load_gold(out, label, seeds)
    Ti = np.array([r["T_material"] for r in imc]) / K_PER_HEV
    Tm = Ti.mean(0)
    Tse = Ti.std(0, ddof=1) / np.sqrt(len(imc)) if len(imc) > 1 else np.zeros_like(Tm)
    Td = d["T_material"] / K_PER_HEV
    lines = [f"| t [ns] | Heat front (T = {front_level} HeV), diffusion -> IMC [g/cm^2] "
             f"| T median / max difference (T > {hot_level} HeV) | IMC s.e. (median) |",
             "|---|---|---|---|"]
    for k in range(len(d["t"])):
        m = d["m"][k]
        hot = Td[k] > hot_level
        fd, fi = _front(m, Td[k], front_level), _front(m, Tm[k], front_level)
        rel = np.abs(Tm[k] - Td[k])[hot] / Td[k][hot]
        se = np.median(Tse[k][hot] / Tm[k][hot])
        lines.append(f"| {d['t'][k] * 1e9:.2f} | {fd:.4g} -> {fi:.4g} ({(fi - fd) / fd:+.1%}) "
                     f"| {np.median(rel):.1%} / {rel.max():.1%} | {se:.2%} |")
    err_d = np.max(np.abs(d["energy_error"]))
    err_i = max(np.max(np.abs(r["energy_error"])) for r in imc)
    lines.append(f"\nmax |energy error|: diffusion {err_d:.1e}, IMC {err_i:.1e}")
    return "\n".join(lines)


def plot_gold(out: Path, label: str, seeds, path: Path):
    import matplotlib.pyplot as plt
    from rad_hydro_sim.simulation.radiation_step import a_Kelvin

    d, imc = _load_gold(out, label, seeds)
    Ti = np.array([r["T_material"] for r in imc]) / K_PER_HEV
    Ei = np.array([r["E_rad_track"] for r in imc])      # track-length (step-averaged) estimate
    Tm = Ti.mean(0)
    Tse = Ti.std(0, ddof=1) / np.sqrt(len(imc)) if len(imc) > 1 else np.zeros_like(Tm)
    Trad_i = (Ei.mean(0) / a_Kelvin) ** 0.25 / K_PER_HEV
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, col in enumerate(["tab:green", "tab:orange", "tab:purple"]):
        m = d["m"][k] * 1e3
        t_ns = d["t"][k] * 1e9
        axes[0].fill_between(m, Tm[k] - 2 * Tse[k], Tm[k] + 2 * Tse[k], color=col, alpha=0.3)
        axes[0].plot(m, d["T_material"][k] / K_PER_HEV, "k-", lw=0.9)
        axes[0].plot(m, Tm[k], color=col, lw=1.4, label=f"IMC, t = {t_ns:.2f} ns")
        axes[1].plot(m, (d["E_rad"][k] / a_Kelvin) ** 0.25 / K_PER_HEV, "k-", lw=0.9)
        axes[1].plot(m, Trad_i[k], color=col, lw=1.4, label=f"IMC, t = {t_ns:.2f} ns")
    hot = np.flatnonzero(d["T_material"][-1] / K_PER_HEV > 0.02)
    xmax = 1e3 * d["m"][-1][hot[-1]] * 1.3 if hot.size else None
    for ax, ylab in zip(axes, ["material temperature [HeV]", "radiation temperature [HeV]"]):
        ax.plot([], [], "k-", lw=0.9, label="diffusion (Phase B flags)")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(r"mass coordinate m [mg/cm$^2$]")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{GOLD_CASES[label].title}: IMC ({len(imc)} seeds, band = $\\pm$2 s.e.) vs diffusion",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- driver
FIGURE_NAMES = {"t1": "t1_sio2_outgoing_flux.png", "test1": "test1_profiles.png",
                "test2": "test2_profiles.png", "thin100": "thin100_profiles.png"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", choices=ALL_CASES, default=list(ALL_CASES))
    p.add_argument("--seeds", default="1,2,3", help="IMC seeds for the rad_hydro_sim runs")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="run cache and report directory")
    p.add_argument("--figures-dir", type=Path, default=None, help="where to write the PNGs (default: --out)")
    p.add_argument("--rerun", action="store_true", help="ignore cached runs")
    p.add_argument("--no-runs", action="store_true", help="only analyze cached runs")
    p.add_argument("--skip-standalone", action="store_true", help="T1 without the standalone IMC")
    p.add_argument("--skip-sn", action="store_true", help="T1 without the S_N twin")
    p.add_argument("--t1-end", type=float, default=T1_END_NS, help="T1 end time [ns] (smoke tests)")
    p.add_argument("--gold-n", type=int, default=None, help="override the gold cell count (smoke tests)")
    p.add_argument("--gold-t-end", type=float, default=None, help="override the gold end time [s] (smoke tests)")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.figures_dir or out
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for case in args.cases:
        print(f"[{case}]", flush=True)
        if case == "t1":
            if not args.no_runs:
                run_t1(out, seeds, args.rerun, not args.skip_standalone, not args.skip_sn, args.t1_end)
            res = t1_blocks(out, seeds, args.t1_end)
            plot_t1(res, fig_dir / FIGURE_NAMES[case], args.t1_end)
            report.append(f"## T1: SiO2 foam\n\n{t1_table(res)}\n")
        else:
            if not args.no_runs:
                run_gold(out, case, seeds, args.rerun, args.gold_n, args.gold_t_end)
            plot_gold(out, case, seeds, fig_dir / FIGURE_NAMES[case])
            report.append(f"## {case}\n\n{gold_table(out, case, seeds)}\n")

    text = "# IMC vs. diffusion report\n\n" + "\n".join(report)
    (out / "report.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"figures in {fig_dir}\nreport  in {out / 'report.md'}")


if __name__ == "__main__":
    main()

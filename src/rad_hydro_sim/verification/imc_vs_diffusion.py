"""IMC vs. diffusion verification: runs, tables and figures (plan milestone M10, Phases B and C).

One command regenerates the results of md/phase_b/phase_b_results.md and
md/phase_c/phase_c_results.md:

    uv run python -m rad_hydro_sim.verification.imc_vs_diffusion
    uv run python -m rad_hydro_sim.verification.imc_vs_diffusion --cases t1 test1 test2 thin100 \
        --figures-dir md/phase_b/figures
    uv run python -m rad_hydro_sim.verification.imc_vs_diffusion --cases test1_full test2_full \
        --figures-dir md/phase_c/figures

Radiation-only cases (Phase B, static mesh):
  t1       SiO2 foam (PRESET_SIO2_FOAM_RADIATION_ONLY), 100 cells, 3 ns. Outgoing flux of
           diffusion (original and Phase B flags), IMC through rad_hydro_sim, the standalone
           IMC (imc.problems.sio2) and the S_N twin (src/sn/sn.py).
  test1    paper Test 1 (PRESET_PAPER_TEST1_RADIATION_ONLY), domain cut to 1e-2 g/cm^2, 400 cells.
  test2    paper Test 2 (PRESET_PAPER_TEST2_RADIATION_ONLY), same mesh.
  thin100  Test 1 with opacity / 100 (g x 100), full preset domain, 400 cells.

Full rad-hydro cases (Phase C, Lagrangian mesh; M5 and M5b):
  test1_full  paper Test 1 (PRESET_PAPER_TEST1), 400 cells, 2 ns: diffusion and IMC with the
              matched coupling (the manuscript's equations), each also with rad_coupling=
              "physical", against the semi-analytic AblationSolver.
  test2_full  paper Test 2 (PRESET_PAPER_TEST2), same.

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


@dataclass(frozen=True)
class HydroCase:
    preset: str          # attribute name in presets_config
    title: str
    N: int = 400
    dt_max: float = 2e-12
    t_end: float = 2e-9


HYDRO_CASES = {
    "test1_full": HydroCase("PRESET_PAPER_TEST1", r"Test 1 ($\tau=0$)"),
    "test2_full": HydroCase("PRESET_PAPER_TEST2", r"Test 2 ($\tau=158/1285$)"),
}
# Phase C configuration: the Phase B flags plus the two that full rad-hydro needs.
HYDRO_MATCHED = dict(DIFFUSION_PHASE_B, rad_energy_per_mass=True, hydro_energy="compatible")
HYDRO_SNAP_NS = (0.5, 1.0, 1.5, 2.0)
ALL_CASES = ("t1", *GOLD_CASES, *HYDRO_CASES)


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


# --------------------------------------------------------------------------- full rad-hydro: runs
def _hydro_case(spec: HydroCase, t_end=None):
    from rad_hydro_sim.problems import presets_config as P

    return replace(P.PRESET_TEST_CASES[getattr(P, spec.preset)], t_sec_end=t_end or spec.t_end)


def _fronts_of_frame(m, rho, p, rho0):
    """(heat front, shock front, ablation pressure) of one profile, read off the density.

    rho0 is each cell's initial density (an array: rho0 r^-omega for omega != 0).

    The density is smooth in both solvers (IMC noise enters it only through the
    pressure), unlike T.  Behind the ablation (heat) front the matter has expanded,
    rho << rho0; ahead of it, between it and the shock, it is compressed, rho > rho0.
    So the heat front is where rho first climbs back to rho0 from the surface, and
    the shock is where the compression rho - rho0 falls through half its peak,
    searching from the cold side.  Both are interpolated between cell centres; the
    ablation pressure is p at the heat front.  (run_comparison's detect_front, the
    steepest log-drop of T and p, locks onto the shock on coarse meshes.)
    """
    mc = m - 0.5 * np.diff(m, prepend=0.0)                    # cell-centre mass coordinates
    excess = rho / rho0 - 1.0                                  # relative compression
    up = np.flatnonzero(excess >= 0.0)
    if not up.size or up[0] == 0:
        return np.nan, np.nan, np.nan
    j = up[0]
    m_heat = float(np.interp(0.0, [excess[j - 1], excess[j]], [mc[j - 1], mc[j]]))
    half = 0.5 * excess[j:].max()
    k = j + int(np.flatnonzero(excess[j:] >= half)[-1])      # last cell above half the peak
    m_shock = (float(np.interp(half, [excess[k + 1], excess[k]], [mc[k + 1], mc[k]]))
               if k + 1 < rho.size else float(mc[k]))
    return m_heat, m_shock, float(np.interp(m_heat, mc, p))


def run_hydro(out: Path, label: str, seeds, rerun=False, N=None, t_end=None, imc_only=False,
              physical=True):
    """Diffusion and IMC on one full rad-hydro case; matched coupling, optionally also physical.

    Each run stores the profiles at HYDRO_SNAP_NS, the front / ablation-pressure series of
    every stored frame, the energy budget (with the A9 v/c diagnostic) and the per-cell
    optical depth at the snapshots.
    """
    spec = HYDRO_CASES[label]
    case = _hydro_case(spec, t_end)
    snaps = np.array([s for s in HYDRO_SNAP_NS if s * 1e-9 <= case.t_sec_end * (1 + 1e-9)]) * 1e-9
    common = dict(N=N or spec.N, store_every=20, show_plot=False, energy_diagnostics=True,
                  dt_ramp_max=spec.dt_max, **HYDRO_MATCHED)

    def run(tag, **opts):
        path = out / f"{label}_{tag}.npz"
        if _cached(path, rerun):
            return
        _, meta, hist, wall = _simulate(case, **common, **opts)
        idx = [int(np.argmin(np.abs(hist.t - ts))) for ts in snaps]
        fronts = np.array([_fronts_of_frame(hist.m[k], hist.rho[k], hist.p[k], hist.rho[0])
                           for k in range(hist.t.size)])
        b = meta["energy_budget"]
        keep = np.unique(np.linspace(0, len(b["t"]) - 1, 2000).astype(int))   # thin the per-step budget
        E_track = meta["imc_E_rad_track"][idx] if "imc_E_rad_track" in meta else hist.E_rad[idx]
        k_t = (hist.rho[idx] ** (1.0 + case.lambda_)
               / (case.g_Kelvin * np.maximum(hist.T_material[idx], 1.0) ** case.alpha))
        dx = np.diff(hist.m[idx], axis=1, prepend=0.0) / hist.rho[idx]        # cell mass / rho
        imc = meta.get("imc", {})
        np.savez(path, t=hist.t[idx], m=hist.m[idx], x=hist.x[idx], rho=hist.rho[idx], p=hist.p[idx],
                 u=hist.u[idx], T_material=hist.T_material[idx], E_rad=hist.E_rad[idx], E_rad_track=E_track,
                 tau_cell=k_t * dx,
                 series_t=hist.t, m_heat=fronts[:, 0], m_shock=fronts[:, 1], p_abl=fronts[:, 2],
                 **{f"budget_{k}": np.asarray(b[k])[keep] for k in
                    ("t", "rad_in_left", "rad_out_right", "E_material", "E_kinetic", "E_radiation",
                     "relative_error", "max_v_over_c", "tau_v_over_c")},
                 max_energy_error=_max_abs(b["relative_error"]), wall=wall,
                 radiation_work_total=imc.get("radiation_work_total", np.nan),
                 radiation_work_to_matter=imc.get("radiation_work_to_matter", np.nan))
        print(f"  ran     {path.name:32s} wall {wall:7.1f} s  steps {len(b['t']) - 1}  "
              f"max|energy err| {_max_abs(b['relative_error']):.1e}", flush=True)

    if not imc_only:
        run("diffusion")
        if physical:
            run("diffusion_physical", rad_coupling="physical")
    for s in seeds:
        run(f"imc{s}", radiation_solver="imc", imc_seed=s)
    if physical and seeds:
        run(f"imc_physical{seeds[0]}", radiation_solver="imc", imc_seed=seeds[0], rad_coupling="physical")


def _max_abs(a) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float))))


# --------------------------------------------------------------------------- full rad-hydro: analysis
def hydro_analytic(label: str, times_sec, N: int):
    """Semi-analytic AblationSolver: profiles at times_sec, plus front / pressure laws and the inflow.

    The total energy of the self-similar solution is the energy that crossed the surface,
    int_0^t s(0, t') dt' with s = 2 sigma_SB (T_bath^4 - T_s^4) (manuscript Sec. "The energy in
    the system"); for T_bath^4 = T_s^4 (1 + B t^p) it is 2 sigma_SB T_s^4 B t^(p+1) / (p+1)
    (tau = 0; for tau != 0 the exponents add 4 tau).
    """
    from types import SimpleNamespace

    from menahem_new.ablation_solver_og import AblationSolver
    from rad_hydro_sim.simulation.radiation_step import a_Kelvin, c
    from rad_hydro_sim.verification.menahem_comparison import (
        _ablation_kwargs_from_case, run_menahem_piecewise_reference)

    case = _hydro_case(HYDRO_CASES[label])
    ref = run_menahem_piecewise_reference(case, SimpleNamespace(N=N), np.asarray(times_sec))
    solver = AblationSolver(**_ablation_kwargs_from_case(case))
    hs = solver.heat_solver
    p0s = hs.Pf * hs.A**hs.a3 * hs.B**hs.b3
    Tb, tau, B, p = float(hs.Tb), float(hs.tau), float(hs.bath_B), float(hs.bath_power)
    k = 4.0 * tau + p + 1.0
    sigma = 0.25 * a_Kelvin * c

    def laws(t):
        t = np.asarray(t, dtype=float)
        return dict(m_heat=np.array([float(hs.ablated_mass(time=ti)) for ti in t]),
                    m_shock=np.array([float(np.ravel(solver.shock_solver.shocked_mass(time=ti))[0]) for ti in t]),
                    p_abl=p0s * t**hs.c3,
                    energy_in=2.0 * sigma * Tb**4 * B * t**k / k)
    return ref, laws


def _load_hydro(out: Path, label: str, seeds):
    runs = {"diffusion": np.load(out / f"{label}_diffusion.npz")}
    runs["imc"] = [np.load(out / f"{label}_imc{s}.npz") for s in seeds]
    for tag in ("diffusion_physical", f"imc_physical{seeds[0]}"):
        if (out / f"{label}_{tag}.npz").exists():
            runs[tag.rstrip("0123456789")] = np.load(out / f"{label}_{tag}.npz")
    return runs


def _series_at(run, key, t):
    return np.interp(t, run["series_t"], run[key])


def _mean_se(values):
    v = np.asarray(values, dtype=float)
    se = v.std(0, ddof=1) / np.sqrt(v.shape[0]) if v.shape[0] > 1 else np.zeros(v.shape[1:])
    return v.mean(0), se


def hydro_table(out: Path, label: str, seeds) -> str:
    runs = _load_hydro(out, label, seeds)
    d, imc = runs["diffusion"], runs["imc"]
    t = np.array([1.0, 1.5, 2.0]) * 1e-9
    _, laws = hydro_analytic(label, t, 10)
    ana = laws(t)
    lines = ["| quantity | t [ns] | analytic | diffusion | IMC (mean +/- s.e.) | IMC vs diffusion |", "|---|---|---|---|---|---|"]
    names = {"m_heat": "heat front [g/cm^2]", "m_shock": "shock front [g/cm^2]", "p_abl": "ablation pressure [Mbar]"}
    for key, name in names.items():
        scale = 1e-12 if key == "p_abl" else 1.0
        for k, tk in enumerate(t):
            vd = _series_at(d, key, tk) * scale
            vi, se = _mean_se([_series_at(r, key, tk) * scale for r in imc])
            lines.append(f"| {name} | {tk * 1e9:.1f} | {ana[key][k] * scale:.4g} | {vd:.4g} ({vd / (ana[key][k] * scale) - 1:+.1%}) "
                         f"| {vi:.4g} +/- {se:.2g} ({vi / (ana[key][k] * scale) - 1:+.1%}) | {vi / vd - 1:+.1%} |")
    # energy that entered, at 2 ns
    tb = imc[0]["budget_t"]
    e_ana = laws(np.array([d["budget_t"][-1]]))["energy_in"][0]
    e_d = d["budget_rad_in_left"][-1]
    e_i, se = _mean_se([[r["budget_rad_in_left"][-1]] for r in imc])
    lines.append(f"| energy in [erg/cm^2] | {tb[-1] * 1e9:.1f} | {e_ana:.4g} | {e_d:.4g} ({e_d / e_ana - 1:+.1%}) "
                 f"| {e_i[0]:.4g} +/- {se[0]:.2g} ({e_i[0] / e_ana - 1:+.1%}) | {e_i[0] / e_d - 1:+.1%} |")
    lines.append("")
    lines.append(f"max |energy error|: diffusion {float(d['max_energy_error']):.1e}, "
                 f"IMC {max(float(r['max_energy_error']) for r in imc):.1e}")
    for tag, base in (("diffusion_physical", d), ("imc_physical", imc[0])):
        if tag in runs:
            ph = runs[tag]
            diffs = ", ".join(f"{name.split(' [')[0]} {_series_at(ph, key, t[-1]) / _series_at(base, key, t[-1]) - 1:+.2%}"
                              for key, name in names.items())
            extra = ""
            if np.isfinite(float(ph["radiation_work_total"])):
                extra = (f"; radiation work charged to the matter: "
                         f"{float(ph['radiation_work_to_matter']) / max(float(ph['radiation_work_total']), 1e-300):.1%}")
            lines.append(f"{tag} vs matched at 2 ns: {diffs}; max |energy error| "
                         f"{float(ph['max_energy_error']):.1e}{extra}")
    vc = max(_max_abs(r["budget_max_v_over_c"]) for r in [d, *imc])
    tvc = max(_max_abs(r["budget_tau_v_over_c"]) for r in [d, *imc])
    lines.append(f"A9: max v/c = {vc:.1e}, max sum(tau v/c) over the heated cells = {tvc:.1e}")
    return "\n".join(lines)


def plot_hydro_profiles(out: Path, label: str, seeds, path: Path):
    """rho, p, u, T against mass at 1, 1.5, 2 ns: analytic, diffusion and IMC (band = ±2 s.e.)."""
    import matplotlib.pyplot as plt

    runs = _load_hydro(out, label, seeds)
    d, imc = runs["diffusion"], runs["imc"]
    snaps = [k for k, t in enumerate(d["t"]) if t * 1e9 >= 0.99]
    ref, _ = hydro_analytic(label, d["t"][snaps], d["m"].shape[1])
    fields = [("rho", r"$\rho$ [g/cm$^3$]", 1.0, "rho"), ("p", "p [Mbar]", 1e-12, "p"),
              ("u", "u [km/s]", 1e-5, "u"), ("T_material", "T [HeV]", 1.0 / K_PER_HEV, "T")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = ["tab:green", "tab:orange", "tab:purple"]
    for (key, ylab, scale, ref_key), ax in zip(fields, axes.ravel()):
        for col, (j, k) in zip(colors, enumerate(snaps)):
            t_ns = d["t"][k] * 1e9
            r_scale = 1.0 if ref_key == "T" else scale
            ax.plot(ref.m[j], np.asarray(getattr(ref, ref_key)[j]) * r_scale, "--", color="0.45", lw=1.0)
            ax.plot(d["m"][k], d[key][k] * scale, "k-", lw=0.9)
            mean, se = _mean_se([r[key][k] * scale for r in imc])
            ax.fill_between(imc[0]["m"][k], mean - 2 * se, mean + 2 * se, color=col, alpha=0.3)
            ax.plot(imc[0]["m"][k], mean, color=col, lw=1.3, label=f"IMC, t = {t_ns:.1f} ns")
        ax.plot([], [], "k-", lw=0.9, label="diffusion")
        ax.plot([], [], "--", color="0.45", lw=1.0, label="semi-analytic")
        ax.set_xscale("log")
        ax.set_xlim(1e-5, d["m"][snaps[-1]][-1])
        if key in ("rho", "p"):
            ax.set_yscale("log")
        ax.set_xlabel(r"mass coordinate m [g/cm$^2$]")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{HYDRO_CASES[label].title}, full rad-hydro: IMC ({len(imc)} seeds) vs diffusion vs semi-analytic",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hydro_fronts(out: Path, label: str, seeds, path: Path):
    """Heat and shock fronts, ablation pressure and energy in, against the self-similar laws."""
    import matplotlib.pyplot as plt

    runs = _load_hydro(out, label, seeds)
    d, imc = runs["diffusion"], runs["imc"]
    t = np.geomspace(0.05e-9, d["series_t"][-1], 60)
    _, laws = hydro_analytic(label, t[:1], 10)
    ana = laws(t)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [("m_heat", r"heat front $m_f$ [g/cm$^2$]", 1.0), ("m_shock", r"shock front $m_s$ [g/cm$^2$]", 1.0),
              ("p_abl", "ablation pressure [Mbar]", 1e-12), ("energy_in", r"energy in [erg/cm$^2$]", 1.0)]
    for (key, ylab, scale), ax in zip(panels, axes.ravel()):
        ax.plot(t * 1e9, ana[key] * scale, "--", color="0.45", lw=1.2, label="semi-analytic")
        if key == "energy_in":
            ax.plot(d["budget_t"] * 1e9, d["budget_rad_in_left"], "k-", lw=0.9, label="diffusion")
            for r, col in zip(imc, ["tab:blue", "tab:cyan", "tab:purple"]):
                ax.plot(r["budget_t"] * 1e9, r["budget_rad_in_left"], color=col, lw=0.9, alpha=0.8)
        else:
            ax.plot(d["series_t"] * 1e9, d[key] * scale, "k-", lw=0.9, label="diffusion")
            for r, col in zip(imc, ["tab:blue", "tab:cyan", "tab:purple"]):
                ax.plot(r["series_t"] * 1e9, r[key] * scale, color=col, lw=0.9, alpha=0.8)
        for tag, ls in (("diffusion_physical", ":"), ("imc_physical", "-.")):
            if tag in runs and key != "energy_in":
                ax.plot(runs[tag]["series_t"] * 1e9, runs[tag][key] * scale, ls, color="tab:red", lw=1.0,
                        label=f"{tag.replace('_', ', ')} coupling")
        ax.plot([], [], color="tab:blue", lw=0.9, label=f"IMC ({len(imc)} seeds)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.05, d["series_t"][-1] * 1e9)
        ax.set_xlabel("time [ns]")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{HYDRO_CASES[label].title}, full rad-hydro: fronts and energy vs the self-similar laws", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- driver
FIGURE_NAMES = {"t1": "t1_sio2_outgoing_flux.png", "test1": "test1_profiles.png",
                "test2": "test2_profiles.png", "thin100": "thin100_profiles.png",
                "test1_full": ("test1_full_profiles.png", "test1_full_fronts.png"),
                "test2_full": ("test2_full_profiles.png", "test2_full_fronts.png")}


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
    p.add_argument("--hydro-n", type=int, default=None, help="override the full rad-hydro cell count")
    p.add_argument("--hydro-t-end", type=float, default=None, help="override the full rad-hydro end time [s]")
    p.add_argument("--imc-only", action="store_true",
                   help="full rad-hydro: only the IMC runs (to run seeds in parallel processes)")
    p.add_argument("--no-physical", action="store_true", help="full rad-hydro: skip rad_coupling='physical'")
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
        elif case in HYDRO_CASES:
            if not args.no_runs:
                run_hydro(out, case, seeds, args.rerun, args.hydro_n, args.hydro_t_end,
                          imc_only=args.imc_only, physical=not args.no_physical)
            if args.imc_only:
                continue
            plot_hydro_profiles(out, case, seeds, fig_dir / FIGURE_NAMES[case][0])
            plot_hydro_fronts(out, case, seeds, fig_dir / FIGURE_NAMES[case][1])
            report.append(f"## {case}\n\n{hydro_table(out, case, seeds)}\n")
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

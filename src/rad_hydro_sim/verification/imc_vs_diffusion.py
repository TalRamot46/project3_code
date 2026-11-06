"""IMC vs. diffusion verification: runs, tables and figures (plan milestone M10, Phases B and C).

One command regenerates the results of md/phase_b/phase_b_results.md and
md/phase_c/phase_c_results.md:

    uv run python -m rad_hydro_sim.verification.imc_vs_diffusion
    uv run python -m rad_hydro_sim.verification.imc_vs_diffusion --cases t1 test1 test2 thin100 \
        --figures-dir md/phase_b/figures
    uv run python -m rad_hydro_sim.verification.imc_vs_diffusion --cases test1_full test2_full \
        --figures-dir md/phase_c/figures/paper_bath
    (the fronts figures land there too; md/phase_c/figures keeps only full profiles, see its
    README; the per-bath set comes from rad_hydro_sim.verification.bath_calibration --profiles)

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
    x_max_mass: float | None = None   # domain as areal mass [g/cm^2] (omega = 0 only); None keeps the preset's
    store_every: int = 20             # history frames (fronts series); keep ~1000 frames per run
    imc: tuple = ()                   # extra SimulationConfig items for the IMC runs, as (key, value) pairs
    config: tuple = ()                # extra SimulationConfig items for every run, as (key, value) pairs
    log_mass_axis: bool = False       # profiles on a log mass axis (a thin ablated layer on a wide domain)


HYDRO_CASES = {
    "test1_full": HydroCase("PRESET_PAPER_TEST1", r"Test 1 ($\tau=0$)"),
    "test2_full": HydroCase("PRESET_PAPER_TEST2", r"Test 2 ($\tau=158/1285$)"),
    # Phase C 1/2 (plan Addendum B): the manuscript's Tests 3-6.
    # omega = +0.5: rho ~ x^-0.5 puts ~1e6 g/cm^3 in a first cell 1e-10 cm thick; the acoustic
    # CFL gives dt ~ 1e-15 s, ~2e6 steps to 2 ns, so the IMC runs with 1/10 of the packets.
    "test3_full": HydroCase("CONSTANT_TEMPERATURE_OMEGA_0_5_FULL", r"Test 3 (Au, $\omega=+0.5$)", N=25,
                            store_every=2000, log_mass_axis=True,
                            imc=(("imc_n_emission", 2000), ("imc_n_bath", 2000), ("imc_max_census", 20000))),
    # omega = -0.5: rho ~ x^+0.5 vanishes at the surface; those optically thin cells hold the census
    # near its 2e5 cap (0.6 s per step on the 1397-cell mesh), so the census is combed at 3e4.
    # The default graded mesh left 0.06 mg/cm^2 cells across the shock (the shock 13% behind the
    # solver's); a uniform-mass mesh of 1000 cells gives 0.0073 mg/cm^2 everywhere.
    # Test 3 on a finer mesh: the default graded mesh's shocked region has cells of 1-12 mg/cm^2
    # (the shock at 200-290 mg/cm^2); capping cells at 0.3 mg/cm^2 refines it ~30x while keeping
    # the 1e-13 cm surface cells that set dt, so the step count is unchanged.
    "test3_fine": HydroCase("CONSTANT_TEMPERATURE_OMEGA_0_5_FULL", r"Test 3 (Au, $\omega=+0.5$), fine mesh", N=25,
                            store_every=2000, log_mass_axis=True, config=(("max_cell_mass", 3e-4),),
                            imc=(("imc_n_emission", 2000), ("imc_n_bath", 2000), ("imc_max_census", 20000))),
    # The same with the smooth cap (cell mass growing by <= 5% per cell); "test3_fine" was made with
    # an earlier cap that split cells into whole pieces and left density spikes at the mass jumps.
    "test3_smooth": HydroCase("CONSTANT_TEMPERATURE_OMEGA_0_5_FULL", r"Test 3 (Au, $\omega=+0.5$), fine mesh",
                              N=25, store_every=2000, log_mass_axis=True, config=(("max_cell_mass", 3e-4),),
                              imc=(("imc_n_emission", 2000), ("imc_n_bath", 2000), ("imc_max_census", 20000))),
    "test4_full": HydroCase("CONSTANT_TEMPERATURE_OMEGA_MINUS_0_5_FULL", r"Test 4 (Au, $\omega=-0.5$)", N=1000,
                            store_every=40, config=(("mesh", "uniform_mass"),),
                            imc=(("imc_n_emission", 5000), ("imc_n_bath", 5000), ("imc_max_census", 30000))),
    # Cu: the preset's 0.1 g/cm^2 would leave ~18 cells in the ablated layer at N = 1000; the
    # analytic shock is at 2.2e-2 g/cm^2 at 2 ns, so 3e-2 g/cm^2 holds it.
    "test5_full": HydroCase("PRESET_COPPER_CONST_TEMPERATURE", r"Test 5 (Cu)", N=1000, x_max_mass=3e-2,
                            store_every=40),
    "test6_full": HydroCase("PRESET_ALUMINUM_CONST_TEMPERATURE", r"Test 6 (Al)", N=400, store_every=20),
}
# The meshes of the paper's figures: 2.5x the cells of each test's default mesh, the omega sweep's
# fine-mesh rule (md/phase_c/omega_sweep/README.md, "Mesh convergence"); Test 3's is test3_smooth.
# Frames are stored 2.5x as rarely, since the step count grows with the cells.  The IMC packet
# counts per step stay as they are: over the run each cell then receives about the same packets.
FINE_OF = {"test1_full": "test1_fine", "test2_full": "test2_fine", "test3_full": "test3_smooth",
           "test4_full": "test4_fine", "test5_full": "test5_fine", "test6_full": "test6_fine"}
for _coarse, _fine in FINE_OF.items():
    if _fine not in HYDRO_CASES:
        _spec = HYDRO_CASES[_coarse]
        HYDRO_CASES[_fine] = replace(_spec, title=_spec.title + ", fine mesh", N=int(round(2.5 * _spec.N)),
                                     store_every=int(round(2.5 * _spec.store_every)))
COARSE_OF = {fine: coarse for coarse, fine in FINE_OF.items()}
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

    case = replace(P.PRESET_TEST_CASES[getattr(P, spec.preset)], t_sec_end=t_end or spec.t_end)
    if spec.x_max_mass is not None:
        assert case.omega == 0.0, "x_max_mass assumes a uniform initial density"
        case = replace(case, x_max=spec.x_max_mass / case.rho0)
    return case


HEAT_FRONT_LEVEL = 0.3     # the heat front is where T falls below this fraction of the surface T


def _initial_density(m, case):
    """Each cell's initial density for rho0 x^-omega (planar), from its cumulative mass m."""
    w = case.omega
    edges_m = np.concatenate(([0.0], m))
    x = ((1.0 - w) * edges_m / case.rho0) ** (1.0 / (1.0 - w))
    return np.diff(edges_m) / np.diff(x)


def fronts_of_profile(m, T, rho, p, rho_init):
    """(heat front, shock front, ablation pressure) of one profile.

    Heat (ablation) front: the first point from the surface where T falls below
    HEAT_FRONT_LEVEL x T_surface, linear between cell centres.  It sits on the
    front's temperature cliff (from ~0.5 to a few hundredths of T_s within a cell or
    two), whatever the initial density profile.  Shock: beyond the heat front, where
    the compression rho/rho_init - 1 falls through half its peak, searching from the
    cold side.  Ablation pressure: p at the heat front.
    """
    mc = m - 0.5 * np.diff(m, prepend=0.0)
    level = HEAT_FRONT_LEVEL * T[0]
    below = np.flatnonzero(T < level)
    if not below.size or below[0] == 0:
        return np.nan, np.nan, np.nan
    j = below[0]
    m_heat = float(np.interp(level, [T[j], T[j - 1]], [mc[j], mc[j - 1]]))
    excess = rho / rho_init - 1.0
    half = 0.5 * excess[j:].max()
    k = j + int(np.flatnonzero(excess[j:] >= half)[-1])
    m_shock = (float(np.interp(half, [excess[k + 1], excess[k]], [mc[k + 1], mc[k]]))
               if k + 1 < rho.size else float(mc[k]))
    return m_heat, m_shock, float(np.interp(m_heat, mc, p))


def front_values(run, label: str, t: float) -> dict:
    """{m_heat, m_shock, p_abl} at time t: from the stored profile when t is a snapshot
    (fronts_of_profile, the definition every table uses), else from the stored series.

    A snapshot is the stored frame nearest t; with sparse frames (Test 3 stores every 2000th
    step) it lies up to 0.3% from t, so within 1% counts: a front growing like t^0.5 then moves
    by at most 0.5%.
    """
    k = int(np.argmin(np.abs(run["t"] - t)))
    if abs(run["t"][k] - t) <= 1e-2 * t:
        case = _hydro_case(HYDRO_CASES[label])
        m = run["m"][k]
        vals = fronts_of_profile(m, run["T_material"][k], run["rho"][k], run["p"][k], _initial_density(m, case))
        return dict(zip(("m_heat", "m_shock", "p_abl"), vals))
    return {key: float(_series_at(run, key, t)) for key in ("m_heat", "m_shock", "p_abl")}


BATH_CORRECTION_DIR = REPO / "src" / "rad_hydro_sim" / "data" / "bath_correction"


def calibrate_bath(label: str, target: str = "temperature", iterations: int = 6, tol: float = 2e-3,
                   t_min_ns: float = 0.2, N=None) -> Path:
    """Option (a) of md/phase_c/why_the_solver_differs.md: a bath correction c(t) for one case.

    The manuscript's bath T_bath(t) follows from the Marshak relation assuming radiation and
    matter are in equilibrium at the surface.  Where they are not (thin ablated plasma), no bath
    reproduces the semi-analytic solution's surface temperature and its energy flux at once, so
    the calibration matches one of them:

      target="temperature"  the matter's surface temperature T(cell 0) follows T_s(t)
                            (bath_drive="calibrated", table <preset key>.txt);
      target="flux"         the net energy flux into the slab follows the solution's
                            s(0, t) = 2 sigma_SB T_b^4 B t^(4 tau + p), i.e. the energy delivered
                            matches its closed-form total energy (bath_drive="calibrated_flux",
                            table <preset key>_flux.txt).

    Each iteration is one diffusion run (the Phase C configuration) on a log time grid from
    t_min_ns to t_end, updating, with Q the matched quantity in units of T^4 (T_surface^4, or
    flux / 2 sigma_SB),
        T_bath^4 <- T_bath^4 + g (Q_target - Q),
    with g = 1 at first and then the inverse of the measured response dQ / dT_bath^4 (secant,
    clipped), until max |Q / Q_target - 1|^(1/4) < tol over [t_min_ns, t_end].  Earlier the heated
    layer is thinner than the first cell, so the factor is held at its t_min_ns value.  IMC then
    runs with the same table, so both solvers keep the same drive.
    """
    from hydro_sim.problems.simulation_config import SimulationConfig
    from rad_hydro_sim.problems import presets_config as P
    from rad_hydro_sim.simulation.iterator import (
        bath_correction_path, get_dimensionless_boundary_flux, simulate_rad_hydro)
    from rad_hydro_sim.simulation.radiation_step import a_Kelvin, c as light_c

    if target not in ("temperature", "flux"):
        raise ValueError(f"target must be 'temperature' or 'flux', got {target!r}")
    drive = "calibrated" if target == "temperature" else "calibrated_flux"
    spec = HYDRO_CASES[label]
    case = _hydro_case(spec)
    key = getattr(P, spec.preset)
    case = replace(case, bath_correction=str(BATH_CORRECTION_DIR / f"{key}.txt"))
    path = Path(bath_correction_path(case, drive))
    path.parent.mkdir(parents=True, exist_ok=True)
    law, _ = get_dimensionless_boundary_flux(case)
    tg = np.geomspace(t_min_ns * 1e-9, case.t_sec_end, 60)
    law4 = np.asarray(law.Tbath(time=tg), dtype=float) ** 4
    sigma2 = 2.0 * 0.25 * a_Kelvin * light_c            # 2 sigma_SB
    if target == "temperature":
        target4 = (case.T0_Kelvin * (tg * 1e9) ** case.tau) ** 4
    else:   # s(0, t) / (2 sigma_SB) = T_b^4 B t^(4 tau + p), the derivative of hydro_analytic's energy_in
        target4 = float(law.Tb) ** 4 * float(law.bath_B) * tg ** (4.0 * float(law.tau) + float(law.bath_power))
    c = np.ones_like(tg)
    prev = None
    cfg = SimulationConfig(N=N or spec.N, store_every=spec.store_every, show_plot=False, energy_diagnostics=True,
                           dt_ramp_max=spec.dt_max, bath_drive=drive, **HYDRO_MATCHED, **dict(spec.config))
    for it in range(iterations + 1):
        header = (f"bath correction for {key} (N = {cfg.N}, target = {target}), iteration {it}: "
                  "T_bath = c(t) T_bath,law(t)\nt [ns]  c")
        np.savetxt(path, np.column_stack((tg * 1e9, c)), fmt="%.6e", header=header)
        _, _, meta, hist = simulate_rad_hydro(case, cfg)
        if target == "temperature":
            q4 = np.interp(np.log(tg), np.log(np.maximum(hist.t, 1e-30)), hist.T_material[:, 0]) ** 4
        else:   # net flux in, from the cumulative energy over +-5% in t
            bt, e_in = np.asarray(meta["energy_budget"]["t"]), np.asarray(meta["energy_budget"]["rad_in_left"])
            lo, hi = tg / 1.05, np.minimum(tg * 1.05, bt[-1])
            q4 = (np.interp(hi, bt, e_in) - np.interp(lo, bt, e_in)) / (hi - lo) / sigma2
        err = np.max(np.abs((q4 / target4) ** 0.25 - 1.0))
        print(f"  {label} iteration {it} ({target}): max |(Q / Q_target)^(1/4) - 1| = {err:.2e}  "
              f"(c from {c.min():.4f} to {c.max():.4f})", flush=True)
        if err < tol or it == iterations:
            break
        bath4 = law4 * c**4
        gain = np.ones_like(tg)
        if prev is not None:
            d_bath, d_q = bath4 - prev[0], q4 - prev[1]
            ok = np.abs(d_bath) > 1e-6 * bath4
            gain[ok] = 1.0 / np.clip(d_q[ok] / d_bath[ok], 0.3, 3.0)
        prev = (bath4, q4)
        new4 = bath4 + gain * (target4 - q4)
        c_new = np.clip(new4 / law4, 0.7**4, 1.5**4) ** 0.25
        c = np.convolve(np.pad(c_new, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")    # light smoothing in log t
    return path


def run_hydro(out: Path, label: str, seeds, rerun=False, N=None, t_end=None, imc_only=False,
              physical=True, bath="paper"):
    """Diffusion and IMC on one full rad-hydro case; matched coupling, optionally also physical.

    Each run stores the profiles at HYDRO_SNAP_NS, the front / ablation-pressure series of
    every stored frame, the energy budget (with the A9 v/c diagnostic) and the per-cell
    optical depth at the snapshots.
    """
    spec = HYDRO_CASES[label]
    case = _hydro_case(spec, t_end)
    snaps = np.array([s for s in HYDRO_SNAP_NS if s * 1e-9 <= case.t_sec_end * (1 + 1e-9)]) * 1e-9
    common = dict(N=N or spec.N, store_every=spec.store_every, show_plot=False, energy_diagnostics=True,
                  dt_ramp_max=spec.dt_max, bath_drive=bath, **HYDRO_MATCHED, **dict(spec.config))

    def run(tag, **opts):
        path = out / f"{label}_{tag}.npz"
        if _cached(path, rerun):
            return
        _, meta, hist, wall = _simulate(case, **common, **opts)
        idx = [int(np.argmin(np.abs(hist.t - ts))) for ts in snaps]
        fronts = np.array([fronts_of_profile(hist.m[k], hist.T_material[k], hist.rho[k], hist.p[k], hist.rho[0])
                           for k in range(hist.t.size)])
        # the fronts' Eulerian positions: x is stored at cell centres, the fronts at cell-centre mass
        mc = hist.m - 0.5 * np.diff(hist.m, axis=1, prepend=0.0)
        x_front = np.array([[np.interp(fronts[k, j], mc[k], hist.x[k]) for j in (0, 1)]
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
                 x_heat=x_front[:, 0], x_shock=x_front[:, 1],
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
        run(f"imc{s}", radiation_solver="imc", imc_seed=s, **dict(spec.imc))
    if physical and seeds:
        run(f"imc_physical{seeds[0]}", radiation_solver="imc", imc_seed=seeds[0], rad_coupling="physical",
            **dict(spec.imc))


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
    # IMC runs that have finished; a figure or table without any shows diffusion vs the solver.
    runs["imc"] = [np.load(out / f"{label}_imc{s}.npz") for s in seeds if (out / f"{label}_imc{s}.npz").exists()]
    for tag in ("diffusion_physical", *((f"imc_physical{seeds[0]}",) if seeds else ())):
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
            vd = front_values(d, label, tk)[key] * scale
            vi, se = (_mean_se([front_values(r, label, tk)[key] * scale for r in imc]) if imc
                      else (np.nan, np.nan))
            lines.append(f"| {name} | {tk * 1e9:.1f} | {ana[key][k] * scale:.4g} | {vd:.4g} ({vd / (ana[key][k] * scale) - 1:+.1%}) "
                         f"| {vi:.4g} +/- {se:.2g} ({vi / (ana[key][k] * scale) - 1:+.1%}) | {vi / vd - 1:+.1%} |")
    # energy that entered, at 2 ns
    tb = d["budget_t"]
    e_ana = laws(np.array([d["budget_t"][-1]]))["energy_in"][0]
    e_d = d["budget_rad_in_left"][-1]
    e_i, se = _mean_se([[r["budget_rad_in_left"][-1]] for r in imc]) if imc else (np.array([np.nan]), np.array([np.nan]))
    lines.append(f"| energy in [erg/cm^2] | {tb[-1] * 1e9:.1f} | {e_ana:.4g} | {e_d:.4g} ({e_d / e_ana - 1:+.1%}) "
                 f"| {e_i[0]:.4g} +/- {se[0]:.2g} ({e_i[0] / e_ana - 1:+.1%}) | {e_i[0] / e_d - 1:+.1%} |")
    lines.append("")
    lines.append(f"max |energy error|: diffusion {float(d['max_energy_error']):.1e}, "
                 + (f"IMC {max(float(r['max_energy_error']) for r in imc):.1e}" if imc else "IMC still running"))
    for tag, base in (("diffusion_physical", d), ("imc_physical", imc[0] if imc else None)):
        if tag in runs and base is not None:
            ph = runs[tag]
            diffs = ", ".join(f"{name.split(' [')[0]} {front_values(ph, label, t[-1])[key] / front_values(base, label, t[-1])[key] - 1:+.2%}"
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
    lines.append(optical_depth_line(d))
    return "\n".join(lines)


def optical_depth_line(run) -> str:
    """Optical depth of the hot ablated plasma (layer_optical_depth) at each snapshot."""
    parts = []
    for k, t in enumerate(run["t"]):
        tau, thin, T_s = layer_optical_depth(run, k)
        parts.append(f"{t * 1e9:.1f} ns: {tau:.3g} (mass in cells with tau < 1: {thin:.0%}; surface T {T_s:.3f} HeV)")
    return "optical depth of the hot plasma, T > T_s/2 (diffusion run): " + "; ".join(parts)


def label_short_log_axes(ax):
    """On a log axis spanning under ~1.5 decades matplotlib labels no tick; label the 2, 3, 5 ones."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        if axis.get_scale() == "log" and lo > 0 and hi / lo < 30:
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


VIEW_SCALE_TEXT = {"linear": "linear", "semilogx": "semi-log (log mass)", "semilogy": "semi-log (log fields)",
                   "loglog": "log-log"}
# profile figures: ink and muted text, a recessive grid; diffusion per time on an ordered blue ramp
# (light to dark, 1 -> 2 ns), IMC in one contrasting hue
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e4e3dd"
SIM_COLORS = ("#6da7ec", "#256abf", "#0d366b")
IMC_COLOR = "#eb6834"


# the paper's tests in a few words: the material and what sets the test apart
TEST_DESCRIPTIONS = {1: r"Au (const $T_s$)", 2: r"Au (const $F_s$)", 3: r"Au ($\omega = +0.5$)",
                     4: r"Au ($\omega = -0.5$)", 5: "Cu", 6: "Al"}


def short_case_name(label: str) -> str:
    """'Test 4: Au (omega = -0.5)' for the paper's tests; 'Au (omega=-0.25)' for the omega-scan cases."""
    import re

    title = HYDRO_CASES[label].title
    match = re.match(r"Test (\d+)", title)
    if match:
        return f"Test {match.group(1)}: {TEST_DESCRIPTIONS[int(match.group(1))]}"
    return f"Au ({title[4:]})" if title.startswith("Au, ") else title


def heat_front_mass(solver, t: float):
    """m_f(t) of an AblationSolver (or of an object wrapping one as .base); None for anything else."""
    hs = getattr(solver, "heat_solver", None) or getattr(getattr(solver, "base", None), "heat_solver", None)
    return None if hs is None else float(hs.ablated_mass(time=t))


def shock_front_mass(solver, t: float):
    """m_s(t) of an AblationSolver (or of an object wrapping one as .base); None for anything else."""
    ps = getattr(solver, "shock_solver", None) or getattr(getattr(solver, "base", None), "shock_solver", None)
    return None if ps is None else float(np.ravel(ps.shocked_mass(time=t))[0])


def isolate_shocked(m_heat, m_shock) -> bool:
    """True when the shocked-region window can start past the last heat front, 1.05 max(m_heat), and
    lose under 5% of the thinnest shocked layer: an ablated layer thin against the shocked one
    (Test 3: m_f ~ 1.5 against m_s ~ 300 mg/cm^2), not Test 4's (m_f ~ 1.5 against m_s ~ 2-5)."""
    return 1.05 * max(m_heat) - min(m_heat) < 0.05 * (min(m_shock) - min(m_heat))


def front_spike_mask(mass, sol, m_f) -> np.ndarray:
    """The analytic points to draw: all but the heat solution's last sliver before its front where it
    is denser or colder than the shocked matter just past the front.

    That sliver is where the two self-similar solutions meet: the heat solution's rigid front
    (gold: rho -> 313 g/cm^3, T -> 0.02 HeV) against a lighter, hotter shocked state, a spike in
    rho and T that is an artifact of the patch.  It holds 0.001% of m_f (Test 1), 0.2-0.5%
    (Test 4), up to 3.5% (omega = -0.8 at 1 ns); the front values and the tables do not use it.
    ``mass`` ascending; no front (m_f None) or no shock point: everything is kept."""
    keep = np.ones(len(mass), dtype=bool)
    if m_f is None:
        return keep
    rho, T = (np.asarray(sol[k], dtype=float) for k in ("density", "temperature"))
    heat, shock = np.flatnonzero(mass < m_f), np.flatnonzero(mass > m_f)
    if not len(heat) or not len(shock):
        return keep
    j = shock[0]
    spiky = ~np.isfinite(rho[heat]) | ~np.isfinite(T[heat]) | (rho[heat] > rho[j]) | (T[heat] < T[j])
    run = int(np.argmin(spiky[::-1])) if not spiky.all() else 0   # the spiky points ending at the front
    if run:
        keep[heat[-run:]] = False
    return keep


def _box_is_free(ax, pos) -> bool:
    """True if no plotted point of ax falls in the box pos = (x0, y0, width, height), in axes fractions."""
    x0, y0, w, h = pos
    to_axes = ax.transData + ax.transAxes.inverted()
    for line in ax.get_lines():
        xy = np.column_stack((line.get_xdata(), line.get_ydata())).astype(float)
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy):
            a = to_axes.transform(xy)
            if np.any((a[:, 0] >= x0) & (a[:, 0] <= x0 + w) & (a[:, 1] >= y0) & (a[:, 1] <= y0 + h)):
                return False
    return True


def style_axes(ax, labelsize=10.5):
    """Recessive frame: light grid under the data, no top/right spines, muted ticks."""
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(which="both", top=False, right=False, colors=MUTED, labelcolor=INK, labelsize=labelsize)


def plot_hydro_profiles(out: Path, label: str, seeds, path: Path, solver=None, patched=(), log_axis=None,
                        analytic_label="Analytic", method=None, region="full", scale=None):
    """rho, p, u, T against mass at 1, 1.5, 2 ns, in the style of the ICTT figures.

    ``region``: "full" (the domain), "heat" (the ablated layer, up to 1.3 m_heat) or "shock"
    (from 0.85 m_heat to 1.1 m_shock); the y limits come from the simulations inside it.
    ``scale``: "linear"; "semilogx" (log mass, linear fields); "semilogy" (linear mass, log rho, p,
    T); "loglog" (log mass, log rho, p, T).  On a log field axis u, which changes sign, goes on a
    symmetric-log axis.  None keeps the case's log_mass_axis (or ``log_axis``).

    ``solver``: the analytic curve, an AblationSolver to reuse (building one takes ~20 min) or
    any object with solve(mass=, time=), such as bath_calibration.Analytic;
    labelled ``analytic_label``.  ``method``: the bath (e.g. "paper bath"), in the title.
    ``patched``: extra curves, (legend label, object with solve(mass=, time=), plot kwargs); with
    any given, ``solver`` is drawn gray dashed and these on top (the legacy fix-B figures,
    md/phase_c/legacy/patching_poc_fix_B.md).  ``log_axis``: overrides the case's log_mass_axis.

    The layout of results/ictt/const_T/dimensional_verification/const_T_fully_patched_comparison.png
    (src/ictt29/full_fitting.py): a 2x2 grid against mass in mg/cm^2; diffusion per time on an
    ordered light-to-dark blue ramp; the semi-analytic AblationSolver in ink on a dense mass grid;
    IMC (mean of the seeds) solid orange at every time; on the linear full-domain view, zoom
    insets on the ablation layer (velocity) and on the shocked region (temperature).  One title
    line: the test, the region, the bath, the scale; the legend in a row under it, off the data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from menahem_new.ablation_solver_og import AblationSolver
    from rad_hydro_sim.verification.menahem_comparison import _ablation_kwargs_from_case

    runs = _load_hydro(out, label, seeds)
    d, imc = runs["diffusion"], runs["imc"]
    case = _hydro_case(HYDRO_CASES[label])
    if solver is None:
        solver = AblationSolver(**_ablation_kwargs_from_case(case))
    snaps = [k for k, t in enumerate(d["t"]) if t * 1e9 >= 0.99][-3:]
    original = dict(color=INK, lw=1.7, zorder=4) if not patched else dict(color="gray", lw=1.5, linestyle="--")
    sim_colors = SIM_COLORS
    imc_style = dict(color=IMC_COLOR, lw=1.4, linestyle="-", zorder=3)
    scales = {"rho": 1.0, "p": 1e-12, "u": 1e-5, "T_material": 1.0 / K_PER_HEV}
    ref_keys = {"rho": "density", "p": "pressure", "u": "velocity", "T_material": "temperature"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
    panels = {"rho": axes[0, 0], "p": axes[0, 1], "u": axes[1, 0], "T_material": axes[1, 1]}
    m_heat, m_shock, T_shocked = [], [], []
    an_shocked = {key: [] for key in scales}   # the solver's shocked layer, for the shocked-region y limits
    an_fronts = []                              # the solver's (m_heat, m_shock), for the mass windows
    m_layer = 1.5 * front_values(d, label, float(d["t"][snaps[-1]]))["m_heat"]
    for col, k in zip(sim_colors, snaps):
        t = float(d["t"][k])
        m_sim = d["m"][k]
        # dense on both a linear and a log axis (a thin ablated layer on a wide domain), and in the layer
        mass = np.unique(np.concatenate((np.linspace(1e-12, m_sim[-1], 1000),
                                         np.geomspace(1e-4 * m_sim[0], m_sim[-1], 1500),
                                         np.linspace(1e-12, min(m_layer, m_sim[-1]), 800))))
        exact = solver.solve(mass=mass, time=t)
        keep = front_spike_mask(mass, exact, heat_front_mass(solver, t))   # no patch spike at the front
        m_f_an, m_s_an = heat_front_mass(solver, t), shock_front_mass(solver, t)
        if m_f_an is not None and m_s_an is not None:
            an_fronts.append((m_f_an, m_s_an))
            inside = (mass > 1.02 * m_f_an) & (mass < 0.98 * m_s_an)
            for key in scales:
                an_shocked[key].append(np.asarray(exact[ref_keys[key]], dtype=float)[inside] * scales[key])
        mean = {key: np.mean([r[key][k] for r in imc], axis=0) for key in scales} if imc else None
        for key, ax in panels.items():
            ref = np.asarray(exact[ref_keys[key]], dtype=float)
            ax.plot(m_sim * 1e3, d[key][k] * scales[key], "-", color=col, lw=1.7, zorder=2)
            ax.plot(mass[keep] * 1e3, ref[keep] * scales[key], **original)
            for _, obj, style in patched:
                sol = obj.solve(mass=mass, time=t)
                ax.plot(mass * 1e3, np.asarray(sol[ref_keys[key]], dtype=float) * scales[key], **style)
            if mean is not None:
                ax.plot(imc[0]["m"][k] * 1e3, mean[key] * scales[key], **imc_style)
        fv = front_values(d, label, t)
        m_heat.append(fv["m_heat"])
        m_shock.append(fv["m_shock"])
        behind = (m_sim > 0.5 * m_shock[-1]) & (m_sim < m_shock[-1])          # the shocked plateau
        T_shocked.append(float(np.max(d["T_material"][k][behind])) / K_PER_HEV if behind.any() else 0.0)

    if scale is None:
        scale = "semilogx" if (HYDRO_CASES[label].log_mass_axis if log_axis is None else log_axis) else "linear"
    log_x, log_y = scale in ("semilogx", "loglog"), scale in ("semilogy", "loglog")
    # the mass window [mg/cm^2]
    x_max = 1e3 * float(d["m"][snaps[-1]][-1])
    x_first = 0.5e3 * float(np.min(d["m"][snaps[-1]]))
    # the fronts of the simulations and of the solver: a window holds both (Test 4: the solver's shock is
    # 14% ahead of diffusion's at 2 ns)
    all_heat = list(m_heat) + [f for f, _ in an_fronts]
    all_shock = list(m_shock) + [s for _, s in an_fronts]
    lo, hi = {"full": (x_first if log_x else 0.0, x_max),
              "heat": (x_first if log_x else 0.0, min(1.3e3 * max(all_heat), x_max)),
              "shock": (0.85e3 * min(all_heat), min(1.1e3 * max(all_shock), x_max))}[region]
    if region == "shock" and isolate_shocked(all_heat, all_shock):
        lo = 1.05e3 * max(all_heat)     # a heat layer thin against the shocked one (Test 3): left out
    # y limits from the simulations inside the window (keeps the analytic singularities in bounds); in the
    # shocked region from the shocked matter alone, simulations and solver: each snapshot's cells between
    # its heat front and its shock, so the plateaus and their differences fill the panels (from 1.1 m_heat:
    # the simulated heat front's tail reaches ~1.05 m_heat, Test 3: T ~ 0.15 HeV against 0.001 behind it)
    sim = {key: np.concatenate([d[key][k][(d["m"][k] * 1e3 >= lo) & (d["m"][k] * 1e3 <= hi)] * scales[key]
                                for k in snaps]) for key in scales}
    if region == "shock":
        shocked = {key: np.concatenate([d[key][k][(d["m"][k] > 1.1 * m_heat[i]) & (d["m"][k] < 0.98 * m_shock[i])]
                                        * scales[key] for i, k in enumerate(snaps)] + an_shocked[key])
                   for key in scales}
        if all(v.size for v in shocked.values()):
            sim = shocked
    for key, ax in panels.items():
        v = sim[key]
        if log_y and key != "u":
            pos = v[v > 0]
            ax.set_yscale("log")
            ax.set_ylim(0.7 * max(pos.min(), 1e-9 * pos.max()), 1.6 * pos.max())
        elif log_y:                   # u changes sign: symmetric log, linear within 1 km/s
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(1.5 * min(v.min(), -1.0), 1.5 * max(v.max(), 1.0))
        elif region == "full":        # as in the reference figure
            top = {"rho": 1.15, "p": 1.15, "T_material": 1.15}
            if key == "u":
                ax.set_ylim(1.15 * v.min() - 5.0, v.max() + 10.0)
            else:
                ax.set_ylim(-(0.025 if key == "rho" else 0.05) * v.max(), top[key] * v.max())
        else:
            span = v.max() - v.min()
            ax.set_ylim(v.min() - 0.06 * span, v.max() + 0.12 * span)

    titles = {"rho": "Density", "p": "Pressure", "u": "Velocity", "T_material": "Temperature"}
    ylabels = {"rho": r"$\rho$ [g/cm$^3$]", "p": r"$P$ [Mbar]", "u": r"$u$ [km/s]", "T_material": r"$T$ [HeV]"}
    for key, ax in panels.items():
        if log_x:         # a thin ablated layer on a wide domain: both in view at once
            ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        label_short_log_axes(ax)
        style_axes(ax)
        ax.set_title(titles[key], color=INK, fontsize=13.5, loc="left", fontweight="bold", pad=8)
        ax.set_ylabel(ylabels[key], color=INK, fontsize=12.5)
        ax.set_xlabel(r"Lagrangian mass $m$ [mg/cm$^2$]", color=INK, fontsize=12.5)

    # Zoom insets: the ablation layer's velocity, the shocked region's temperature.
    u_layer = np.concatenate([d["u"][k][d["m"][k] < 2.0 * max(m_heat)] for k in snaps]) * scales["u"]
    insets = {
        "u": ((0.48, 0.08, 0.50, 0.50), (0.0, 2.0e3 * max(m_heat)), (1.05 * u_layer.min(), 0.0)),
        "T_material": ((0.40, 0.42, 0.60, 0.50), (0.8e3 * min(m_shock), 1.05e3 * max(m_shock)),
                       (0.0, 3.5 * max(T_shocked) if max(T_shocked) > 0 else 0.15)),
    }
    for key, (pos, xlim, ylim) in (insets if (region == "full" and scale == "linear") else {}).items():
        ax = panels[key]
        # no zoom where it would cover curves (omega = -0.8: a hot shocked layer) or would not zoom
        # (omega = -0.8: twice the heat front's mass is more than the domain)
        if xlim[1] - xlim[0] > 0.6 * (hi - lo) or not _box_is_free(ax, pos):
            continue
        ins = ax.inset_axes(list(pos))
        for line in ax.get_lines():
            ins.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linestyle=line.get_linestyle(),
                     linewidth=0.85 * line.get_linewidth(), zorder=line.get_zorder(),
                     alpha=line.get_alpha() if line.get_alpha() is not None else 1.0)
        ins.set_xlim(*xlim)
        ins.set_ylim(ylim[0], min(ylim[1], ax.get_ylim()[1]))   # the zoom box stays inside its panel
        style_axes(ins, labelsize=9)
        for side in ("top", "right"):    # a framed box, so it reads as a zoom and not as more data
            ins.spines[side].set_visible(True)
            ins.spines[side].set_color(MUTED)
        ins.set_facecolor("white")
        ax.indicate_inset_zoom(ins, edgecolor=MUTED, alpha=0.7)

    # One title line (the test, the region, the bath, the scale), the legend in a row under it:
    # both off the data, aligned with the panels' left edge.
    plt.tight_layout(rect=(0, 0, 1, 0.865 if patched else 0.915), h_pad=2.2, w_pad=2.5)
    x0 = panels["rho"].get_position().x0
    region_text = {"full": "full domain", "heat": "heat region", "shock": "shocked region"}[region]
    title = " · ".join(part for part in (short_case_name(label), region_text, method, VIEW_SCALE_TEXT[scale])
                           if part)
    fig.text(x0, 0.982, title, ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    legend = [Line2D([0], [0], color=c, lw=2.2, label=f"Diffusion, {d['t'][k] * 1e9:.1f} ns")
              for c, k in zip(sim_colors, snaps)]
    legend += [Line2D([0], [0], label="Analytic (original patch)" if patched else analytic_label,
                      **{key: v for key, v in original.items() if key != "zorder"})]
    legend += [Line2D([0], [0], label=name, **style) for name, _, style in patched]
    legend += [Line2D([0], [0], **{key: v for key, v in imc_style.items() if key != "zorder"},
                      label=(f"IMC, mean of {len(imc)} seeds" if len(imc) > 1 else "IMC, 1 seed") if imc
                      else ("IMC (not run)" if patched else "IMC (still running)"))]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(x0, 0.94), borderaxespad=0.0, borderpad=0.0,
               ncol=4 if patched else len(legend), fontsize=11.5, frameon=False, handlelength=2.8,
               columnspacing=2.2, labelcolor=INK)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def T_drive(label: str, t: float) -> float:
    """The surface temperature the semi-analytic solver imposes, T_s(t) = T0 t_ns^tau [HeV]."""
    case = _hydro_case(HYDRO_CASES[label])
    return case.T0_Kelvin * (t * 1e9) ** case.tau / K_PER_HEV


def layer_optical_depth(run, k: int = -1, hot_fraction: float = 0.5):
    """(optical depth of the hot ablated plasma, mass fraction of it in cells with tau < 1, surface T [HeV]).

    The hot plasma is the cells with T > hot_fraction T_surface: the region the drive's radiation
    crosses, where transport and diffusion can part.  tau_cell = rho kappa_R dx.  The whole layer
    behind the heat front is not used: its sum is dominated by the cold, dense cells at the front
    and changes with the mesh (gold Test 1: 43 at N = 400, 13 at N = 1000), while this one does not
    (8.9 and 9.0).
    """
    k = k % len(run["t"])
    T = run["T_material"][k]
    hot = T > hot_fraction * T[0]
    tau = run["tau_cell"][k][hot]
    dm = np.diff(run["m"][k], prepend=0.0)[hot]
    return float(tau.sum()), float(dm[tau < 1.0].sum() / dm.sum()), float(T[0]) / K_PER_HEV


def plot_opacity_summary(outs: dict, seeds, path: Path):
    """One figure across the tests: how opaque each material's ablated layer is, and how far
    IMC is from diffusion and both from the solver there.  outs maps label -> run directory
    (the finest mesh run of each test)."""
    import matplotlib.pyplot as plt

    labels, tau, rows = [], [], []
    for label, out in outs.items():
        avail = [s for s in seeds if (out / f"{label}_imc{s}.npz").exists()]
        if not (out / f"{label}_diffusion.npz").exists() or not avail:
            continue
        runs = _load_hydro(out, label, avail)
        d, imc = runs["diffusion"], runs["imc"]
        t = min(float(d["series_t"][-1]), *(float(r["series_t"][-1]) for r in imc))
        _, laws = hydro_analytic(label, np.array([t]), 10)
        ana = {key: float(v[0]) for key, v in laws(np.array([t])).items()}
        row = {}
        for key in ("m_heat", "m_shock", "p_abl"):
            vd = front_values(d, label, t)[key]
            vi = float(np.mean([front_values(r, label, t)[key] for r in imc]))
            row[key] = (100 * (vi / vd - 1), 100 * (vd / ana[key] - 1))
        e_d = float(np.interp(t, d["budget_t"], d["budget_rad_in_left"]))
        e_i = float(np.mean([np.interp(t, r["budget_t"], r["budget_rad_in_left"]) for r in imc]))
        row["energy"] = (100 * (e_i / e_d - 1), 100 * (e_d / ana["energy_in"] - 1))
        tau_k, _, T_s = layer_optical_depth(d)
        row["T_s"] = 100 * (T_s / T_drive(label, t) - 1)
        labels.append(f"{HYDRO_CASES[label].title}\n{d['m'].shape[1]} cells")
        tau.append(tau_k)
        rows.append(row)
    order = np.argsort(tau)[::-1]                       # most opaque first
    labels = [labels[i] for i in order]
    tau = [tau[i] for i in order]
    rows = [rows[i] for i in order]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].bar(x, tau, color="0.35", width=0.6)
    for xi, v in zip(x, tau):
        axes[0].text(xi, v * 1.08, f"{v:.3g}", ha="center", fontsize=11)
    axes[0].axhline(1.0, color="crimson", lw=1.2, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.3, 4.0 * max(tau))
    axes[0].set_title(r"Optical depth $\mathcal{T}$ of the hot plasma ($T > T_s/2$)", fontsize=14, fontweight="bold")
    keys = [("m_heat", "heat front", "royalblue"), ("m_shock", "shock front", "darkorange"),
            ("p_abl", "ablation pressure", "forestgreen"), ("energy", "energy delivered", "crimson")]
    w = 0.2
    for j, (key, name, col) in enumerate(keys):
        axes[1].bar(x + (j - 1.5) * w, [r[key][0] for r in rows], w, color=col, label=name)
        axes[2].bar(x + (j - 1.5) * w, [r[key][1] for r in rows], w, color=col, label=name)
    axes[2].plot(x, [r["T_s"] for r in rows], "kD", ms=8, label="surface T")
    axes[1].set_title("IMC vs diffusion [%]", fontsize=14, fontweight="bold")
    axes[2].set_title("Diffusion vs semi-analytic solver [%]", fontsize=14, fontweight="bold")
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    for ax in axes[1:]:
        ax.axhline(0.0, color="black", lw=0.8)
        ax.legend(fontsize=10, loc="lower left")
    fig.suptitle("Opacity and agreement at 2 ns: the less opaque the ablated layer, the more transport and diffusion part",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


BATH_DIRS = {"paper": "imc_vs_diffusion", "calibrated": "imc_vs_diffusion_calibrated",
             "calibrated_flux": "imc_vs_diffusion_calibrated_flux"}


def plot_bath_comparison(labels, path: Path, t: float = 2e-9):
    """Diffusion vs the semi-analytic solver at t for each bath (paper, temperature-calibrated,
    flux-calibrated): heat front, shock, ablation pressure and energy delivered, one panel per test.
    IMC follows diffusion within the IMC-vs-diffusion differences of each test's table."""
    import matplotlib.pyplot as plt

    baths = [("paper", "paper bath (option c)", "0.55"), ("calibrated", "surface T matched", "darkorange"),
             ("calibrated_flux", "energy flux matched", "royalblue")]
    keys = [("m_heat", "heat front"), ("m_shock", "shock"), ("p_abl", "ablation p"), ("energy_in", "energy in")]
    fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 4.8), sharey=True)
    for ax, label in zip(np.atleast_1d(axes), labels):
        _, laws = hydro_analytic(label, np.array([t]), 10)
        ana = {key: float(v[0]) for key, v in laws(np.array([t])).items()}
        x = np.arange(len(keys))
        for j, (bath, name, col) in enumerate(baths):
            f = REPO / "results" / BATH_DIRS[bath] / f"{label}_diffusion.npz"
            if not f.exists():
                continue
            d = np.load(f)
            vals = front_values(d, label, t)
            vals["energy_in"] = float(np.interp(t, d["budget_t"], d["budget_rad_in_left"]))
            ax.bar(x + (j - 1) * 0.27, [100 * (vals[k] / ana[k] - 1) for k, _ in keys], 0.27, color=col, label=name)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(x, [n for _, n in keys], fontsize=10)
        ax.set_title(HYDRO_CASES[label].title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel(f"diffusion vs semi-analytic solver at {t * 1e9:g} ns [%]", fontsize=11)
    np.atleast_1d(axes)[0].legend(fontsize=9)
    fig.suptitle("The bath choice: matching the solver's energy flux brings Cu and Al onto it; "
                 "Test 4's gap is the patching", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
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
            for r, col, name in [(d, "black", "diffusion"),
                                 *zip(imc, ["tab:blue", "tab:cyan", "tab:purple"], [None] * len(imc))]:
                # Markers: the snapshots, by fronts_of_profile (the tables' definition).  Line: the
                # stored series, drawn only where it agrees with them; runs made before the heat
                # front moved to the temperature criterion stored a density-based series, which
                # fails for omega = +0.5.
                snap = np.array([front_values(r, label, ts)[key] for ts in r["t"]])
                series_ok = np.all(np.abs(np.interp(r["t"], r["series_t"], r[key]) / snap - 1.0) < 0.2)
                if series_ok:
                    ax.plot(r["series_t"] * 1e9, r[key] * scale, "-", color=col, lw=0.9, alpha=0.8, label=name)
                ax.plot(r["t"] * 1e9, snap * scale, "o", color=col, ms=4,
                        label=None if series_ok else name)
        for tag, ls in (("diffusion_physical", ":"), ("imc_physical", "-.")):
            if tag in runs and key != "energy_in":
                ax.plot(runs[tag]["series_t"] * 1e9, runs[tag][key] * scale, ls, color="tab:red", lw=1.0,
                        label=f"{tag.replace('_', ', ')} coupling")
        ax.plot([], [], color="tab:blue", lw=0.9,
                label=f"IMC ({len(imc)} seeds)" if imc else "IMC (still running)")
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
                "test2_full": ("test2_full_profiles.png", "test2_full_fronts.png"),
                **{f"test{k}_full": (f"test{k}_full_profiles.png", f"test{k}_full_fronts.png") for k in range(3, 7)},
                "test3_fine": ("test3_fine_profiles.png", "test3_fine_fronts.png"),
                "test3_smooth": ("test3_full_profiles.png", "test3_smooth_fronts.png"),
                # the fine meshes draw the paper's figures, under the tests' own names
                **{f"test{k}_fine": (f"test{k}_full_profiles.png", f"test{k}_fine_fronts.png") for k in (1, 2, 4, 5, 6)}}


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
    p.add_argument("--bath", choices=("paper", "calibrated", "calibrated_flux"), default="paper",
                   help="full rad-hydro bath: the manuscript's relation (option c), or that relation times the "
                        "case's calibration table matching the surface temperature (calibrated) or the energy "
                        "flux (calibrated_flux); option (a), runs and report go to <out>_<bath>")
    p.add_argument("--calibrate-target", choices=("temperature", "flux"), default="temperature",
                   help="what --calibrate-bath matches: the surface temperature or the energy flux")
    p.add_argument("--calibrate-bath", action="store_true",
                   help="make the calibration table of each full rad-hydro case in --cases, then stop")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.calibrate_bath:
        for case in args.cases:
            if case in HYDRO_CASES:
                print(f"[{case}] calibrating the bath", flush=True)
                print(f"  wrote {calibrate_bath(case, target=args.calibrate_target, N=args.hydro_n)}")
        return
    out = args.out
    if args.bath != "paper":
        out = out.with_name(out.name + "_" + args.bath)
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
                          imc_only=args.imc_only, physical=not args.no_physical, bath=args.bath)
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

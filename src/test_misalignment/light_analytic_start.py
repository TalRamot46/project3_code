"""Be and Al from the semi-analytic solution at 0.1 ns (md/misalignment/01_surface_equilibrium.md section 3).

In the light materials the hot ablated plasma is only marginally opaque, so at the surface the
radiation is hotter than the matter (01_surface_equilibrium.md section 1).  Question: does a start
from the solver's (one-temperature) profile at 0.1 ns, where the layer is already resolved, bring
the runs to equilibrium and onto the solver?  And for Al: a bath calibrated, from that start, so
that the matter's surface temperature follows T_s(t).

Cases (300 K target, omega = 0, T_s = 1 HeV, tau = 0, 1000 cells):
  al    the manuscript's Test 6 (Al) on its fine mesh
  be    Be with Shussman's Be.m (alpha 4.893, beta 1.0902, lambda 0.6726, mu 0.0701, r 0.5529),
        rho0 = 1.85 g/cm^3, domain 1.15 x the solver's shocked mass at 2 ns
Starts: cold (t = 0) and analytic (the solver's profile at 0.1 ns, shock on a face, smeared).
Baths: paper, calibrated (matter surface temperature, calibrated from the analytic start: its
own table <key>_analytic.txt), calibrated_flux (Al: the existing cold-start table).

    python -m test_misalignment.light_analytic_start --run al_analytic_paper be_cold_paper
    python -m test_misalignment.light_analytic_start --calibrate al_analytic
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
OUT = REPO / "results" / "misalignment" / "light_analytic_start" / "runs"
KELVIN_PER_HEV = 1_160_500.0
T_SEED = 1e-10

BE_KEY = "beryllium_const_temperature"
AL_KEY = "aluminum_const_temperature"


def _register():
    from rad_hydro_sim.problems import presets_config as P
    if BE_KEY not in P.PRESET_TEST_CASES:
        al = P.PRESET_TEST_CASES[AL_KEY]
        H = KELVIN_PER_HEV
        be = replace(al, alpha=4.893, beta_Rosen=1.0902, lambda_=0.6726, mu=0.0701,
                     f_Kelvin=8.8053e13 / H**1.0902, g_Kelvin=1.0 / (402.8102 * H**4.893), r=0.5529,
                     rho0=1.85, title=r"Beryllium constant temperature ($T_0 = 1$ HeV, $\tau = 0$, Be)")
        s = solver_of(be, "be")
        m_s = float(np.ravel(s.shock_solver.shocked_mass(time=be.t_sec_end))[0])
        P.PRESET_TEST_CASES[BE_KEY] = replace(be, x_max=1.15 * m_s / be.rho0)
        P.PRESET_BERYLLIUM_CONST_TEMPERATURE = BE_KEY
    for mat, preset in (("al", "PRESET_ALUMINUM_CONST_TEMPERATURE"), ("be", "PRESET_BERYLLIUM_CONST_TEMPERATURE")):
        for start in ("cold", "analytic"):
            items = (("initial_condition", "analytic_ablation"), ("t_sec_start", T_SEED)) if start == "analytic" else ()
            cfg = (("seed_shock_on_face", True), ("seed_shock_smeared", True)) if start == "analytic" else ()
            ivd.HYDRO_CASES.setdefault(f"{mat}_{start}", ivd.HydroCase(
                preset, f"{mat} {start}", N=1000, store_every=50, case_items=items, config=cfg))


def solver_of(case, label):
    from rad_hydro_sim.verification import bath_calibration as bc
    return bc._solver_for_case(case, label)


def case_of(label: str, bath: str):
    """(case, config dict) of '<mat>_<start>' under a bath."""
    from rad_hydro_sim.problems import presets_config as P
    _register()
    spec = ivd.HYDRO_CASES[label]
    case = ivd._hydro_case(spec)
    key = getattr(P, spec.preset)
    if bath == "calibrated":
        # the analytic start's own temperature table
        case = replace(case, bath_correction=str(ivd.BATH_CORRECTION_DIR / f"{key}_analytic.txt"))
    elif bath == "calibrated_flux":
        case = replace(case, bath_correction=str(ivd.BATH_CORRECTION_DIR / f"{key}.txt"))
    cfg = dict(N=spec.N, store_every=spec.store_every, show_plot=False, energy_diagnostics=True,
               dt_ramp_max=spec.dt_max, bath_drive=bath, **ivd.HYDRO_MATCHED, **dict(spec.config))
    return case, cfg


def simulate(case, cfg):
    from hydro_sim.problems.simulation_config import SimulationConfig
    from rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    wall = time.time()
    _, _, meta, hist = simulate_rad_hydro(case, SimulationConfig(**cfg))
    return meta, hist, time.time() - wall


def save(path, case, meta, hist, wall):
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    b = meta["energy_budget"]
    keep = np.unique(np.linspace(0, len(b["t"]) - 1, 3000).astype(int))
    np.savez(path, t=hist.t, m=hist.m[0], x=f32(hist.x), rho=f32(hist.rho), p=f32(hist.p), u=f32(hist.u),
             T_material=f32(hist.T_material), T_rad=f32(hist.T), wall=wall,
             budget_t=np.asarray(b["t"])[keep], budget_in=np.asarray(b["rad_in_left"])[keep],
             max_energy_error=float(np.max(np.abs(b["relative_error"]))), t_start=hist.t[0])
    print(f"  ran     {path.name}: {hist.t.size} frames, wall {wall:.0f} s", flush=True)


def run(name: str, rerun: bool = False):
    """name = <mat>_<start>_<bath>[_n<N>] (N: cells, default 1000)."""
    base, _, n = name.partition("_n")
    mat, start, bath = base.split("_", 2)
    path = OUT / f"{name}.npz"
    if path.exists() and not rerun:
        print(f"  cached  {path.name}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    case, cfg = case_of(f"{mat}_{start}", bath)
    if n:
        cfg["N"] = int(n)
        cfg["store_every"] = int(round(cfg["store_every"] * int(n) / 1000))
    meta, hist, wall = simulate(case, cfg)
    save(path, case, meta, hist, wall)


def calibrate(label: str, iterations: int = 6, tol: float = 2e-3, t_min_ns: float = 0.2):
    """ivd.calibrate_bath (temperature target) for '<mat>_<start>', writing <key>_analytic.txt;
    each iteration's run is kept as <label>_calibrated_it<k>.npz."""
    from rad_hydro_sim.problems import presets_config as P
    from rad_hydro_sim.simulation.iterator import get_dimensionless_boundary_flux
    _register()
    case, cfg = case_of(label, "calibrated")
    path = case.bath_correction
    law, _ = get_dimensionless_boundary_flux(case)
    tg = np.geomspace(t_min_ns * 1e-9, case.t_sec_end, 60)
    law4 = np.asarray(law.Tbath(time=tg), dtype=float) ** 4
    target4 = (case.T0_Kelvin * (tg * 1e9) ** case.tau) ** 4
    c, prev = np.ones_like(tg), None
    OUT.mkdir(parents=True, exist_ok=True)
    for it in range(iterations + 1):
        np.savetxt(path, np.column_stack((tg * 1e9, c)), fmt="%.6e",
                   header=f"bath correction for {label} (temperature target, analytic start), iteration {it}\nt [ns]  c")
        meta, hist, wall = simulate(case, cfg)
        save(OUT / f"{label}_calibrated_it{it}.npz", case, meta, hist, wall)
        q4 = np.interp(np.log(tg), np.log(np.maximum(hist.t, 1e-30)), hist.T_material[:, 0]) ** 4
        err = np.max(np.abs((q4 / target4) ** 0.25 - 1.0))
        print(f"  {label} iteration {it}: max |T_surface / T_s - 1| = {err:.2e} (c {c.min():.4f}..{c.max():.4f})",
              flush=True)
        if err < tol or it == iterations:
            break
        bath4 = law4 * c**4
        gain = np.ones_like(tg)
        if prev is not None:
            d_bath, d_q = bath4 - prev[0], q4 - prev[1]
            ok = np.abs(d_bath) > 1e-6 * bath4
            gain[ok] = 1.0 / np.clip(d_q[ok] / d_bath[ok], 0.3, 3.0)
        prev = (bath4, q4)
        c_new = np.clip((bath4 + gain * (target4 - q4)) / law4, 0.7**4, 1.5**4) ** 0.25
        c = np.convolve(np.pad(c_new, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    import shutil
    shutil.copy(OUT / f"{label}_calibrated_it{it}.npz", OUT / f"{label}_calibrated.npz")
    shutil.copy(path, str(path).replace(".txt", f"_tmin{t_min_ns:g}.txt"))


def main(argv=None):
    import logging
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+")
    ap.add_argument("--calibrate", nargs="+")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--t-min", type=float, default=0.2, help="calibration grid start [ns]")
    ap.add_argument("--calibrate-flux", nargs="+", help="flux-target calibration (ivd.calibrate_bath) of <mat>_<start>")
    args = ap.parse_args(argv)
    for name in args.run or ():
        run(name, args.rerun)
    for label in args.calibrate or ():
        calibrate(label, t_min_ns=args.t_min)
    for label in args.calibrate_flux or ():
        _register()
        ivd.calibrate_bath(label, target="flux", t_min_ns=args.t_min)


if __name__ == "__main__":
    main()

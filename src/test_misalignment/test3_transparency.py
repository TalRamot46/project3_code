"""Test 3 at short times: transparency, non-equilibrium and their imprint (md/misalignment/01_surface_equilibrium.md section 2, 03_startup.md section 3, 05_shocked_region.md section 2).

Runs
- ``dense_<name>``: a run of test3_on_the_solver (cold_2T_30K, cold_1T_30K, ...) again, storing every
  ``store_every`` steps (log-spaced in t, since dt ~ t) the whole profile including the radiation
  temperature, from ~1e-8 ns to 2 ns.  Used for the optical depth tau(m, t) and the front series.

    python -m test_misalignment.test3_transparency --dense cold_2T_30K
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from test_misalignment import test3_on_the_solver as tos

REPO = ivd.REPO
OUT = REPO / "results" / "misalignment" / "test3_transparency" / "runs"                                # run data (untracked)
FIGURES = REPO / "results" / "misalignment" / "test3_transparency" / "figures"    # figures (tracked)


def run_dense(name: str, store_every: int = 1000, t_end: float = 2e-9, rerun: bool = False):
    from hydro_sim.problems.simulation_config import SimulationConfig
    from rad_hydro_sim.simulation.iterator import simulate_rad_hydro

    path = OUT / f"dense_{name}.npz"
    if path.exists() and not rerun:
        print(f"  cached  {path.name}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    case, cfg = tos.case_and_config(name, t_end)
    cfg["store_every"] = store_every
    wall = time.time()
    _, _, meta, hist = simulate_rad_hydro(case, SimulationConfig(**cfg))
    wall = time.time() - wall
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    np.savez(path, t=hist.t, m=hist.m[0], x=f32(hist.x), rho=f32(hist.rho), p=f32(hist.p), u=f32(hist.u),
             e=f32(hist.e), T_material=f32(hist.T_material), T_rad=f32(hist.T), wall=wall,
             chi=case.chi, T_initial=case.T_initial_Kelvin)
    print(f"  ran     {path.name}: {hist.t.size} frames, wall {wall:.0f} s", flush=True)


# --------------------------------------------------------------------------- task 2: p_f(t)-driven runs
# name -> (start, physics, drive).  test3_on_the_solver's mesh of the seeded runs, 30 K target.
#   physics "hydro": hydro only;  "rad": the 2T diffusion (chi = 1) with zero radiation flux at m = 0.
#   drive "p_solver": p_f(t) = p0 t^c3;  "p_2T": the pressure on the dense side of the front of the 2T
#   rad-hydro run with the same start (cold_2T_30K / seeded_2T_30K_smeared), smoothed.
PDRIVE = {f"{start}_{phys}_{drive}": (start, phys, drive)
          for start in ("cold", "seeded") for drive in ("p_solver", "p_2T") for phys in ("hydro", "rad")}
P_SOURCE = {"cold": "cold_2T_30K", "seeded": "seeded_2T_30K_smeared"}


def p2T_series(start: str):
    """(t, p) on the dense side of the 2T run's front, a running median over 9 stored points (the
    stored value is the largest pressure of 30 cells beyond the front, biased by the ringing)."""
    run = np.load(tos.OUT / f"{P_SOURCE[start]}.npz")
    t, p = run["series_t"], run["p_dense"]
    ok = np.isfinite(p) & (p > 0) & (t > 0)
    t, lp = t[ok], np.log(p[ok])
    k = 4
    pad = np.pad(lp, k, mode="edge")
    med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(lp.size)])
    return t, np.exp(med)


def pdrive_case_config(name: str, t_end: float = 2e-9):
    from dataclasses import replace
    start, phys, _ = PDRIVE[name]
    base = "cold_2T_30K" if start == "cold" else "seeded_2T_30K_smeared"
    case, cfg = tos.case_and_config(base, t_end)
    case = replace(case, initial_condition="temperature, density", t_sec_start=0.0,
                   scenario="full_rad_hydro" if phys == "rad" else "hydro_only", bc_type="reflective")
    for key in ("seed_shock_on_face", "seed_shock_smeared"):
        cfg.pop(key, None)
    # equal 0.0216 mg/cm^2 surface cells for both starts: a driven surface cell is never ablated,
    # and on the graded mesh its 1e-15 cm would set the step for the whole run
    cfg.update(dict(tos.MESH_SEEDED))
    cfg.pop("dt_first_frac", None)
    return case, cfg


def run_pdrive(name: str, t_end: float = 2e-9, rerun: bool = False, store_every: int = 2000):
    from dataclasses import replace
    from hydro_sim.problems.simulation_config import SimulationConfig
    from rad_hydro_sim.simulation.integrator import step_rad_hydro
    from rad_hydro_sim.simulation.iterator import compute_adaptive_dt, initialize_problem
    from rad_hydro_sim.simulation.radiation_step import a_Kelvin
    from test_misalignment import test3_hydro_only as h

    path = OUT / f"pdrive_{name}.npz"
    if path.exists() and not rerun:
        print(f"  cached  {path.name}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    start, phys, drive = PDRIVE[name]
    s = h.solver()
    p0, c3 = h.solver_pressure_law(s)
    if drive == "p_solver":
        p_of_t = lambda t: p0 * t**c3
    else:
        p_of_t = h.tabulated_pressure(*p2T_series(start), c3)
    case, cfg = pdrive_case_config(name, t_end)
    config = SimulationConfig(**cfg)
    state = initialize_problem(case, config)
    if start == "seeded":
        state = h.seed_piston_shock(case, state, s, tos.T_SEED)
    state.T_rad = state.T_material.copy()
    state.E_rad = a_Kelvin * state.T_material**4
    m = np.cumsum(state.m_cells)
    rho_init = ivd._initial_density(m, case)
    snaps = [x * 1e-9 for x in tos.SNAP_NS if state.t < x * 1e-9 <= t_end * (1 + 1e-12)]
    frames, series = [], []
    wall0, step, dt_prev = time.time(), 0, np.inf
    prev_E, prev_T = state.E_rad, state.T_rad

    def record():
        excess = state.rho / rho_init - 1.0
        k = int(np.flatnonzero(excess >= 0.5 * excess.max())[-1])
        series.append((state.t, m[k], state.p[0], state.T_material[0], state.T_rad[0]))

    def frame():
        frames.append(dict(t=state.t, x=0.5 * (state.x[:-1] + state.x[1:]), rho=state.rho.copy(),
                           p=state.p.copy(), u=0.5 * (state.u[:-1] + state.u[1:]),
                           T_material=state.T_material.copy(), T_rad=np.asarray(state.T_rad).copy()))

    record()
    for t_snap in snaps:
        while state.t < t_snap * (1 - 1e-12):
            dt = compute_adaptive_dt(step=step, t=state.t, t_end=t_end, dt_prev=dt_prev, state=state,
                                     scenario="full_rad_hydro", CFL=config.CFL, gamma=case.r + 1.0,
                                     E_rad_last=prev_E, T_rad_last=prev_T, dt_ramp_max=config.dt_ramp_max,
                                     dt_first_frac=config.dt_first_frac)
            dt = min(dt, t_snap - state.t)
            p_mid = float(p_of_t(state.t + 0.5 * dt))
            step_case = replace(case, P0_Barye=p_mid, tau=0.0)
            prev_E, prev_T = state.E_rad, state.T_rad
            new, _ = step_rad_hydro(state, dt, step_case, config, None)
            if phys == "hydro":
                new.T_rad = new.T_material
                new.E_rad = np.zeros_like(new.rho)
            state, dt_prev, step = new, dt, step + 1
            if step % store_every == 0:
                record()
        frame()
        record()
        print(f"  {name}: t = {state.t * 1e9:.2f} ns, step {step}, wall {time.time() - wall0:.0f} s", flush=True)
    series = np.array(series)
    np.savez(path, t=np.array([f["t"] for f in frames]), m=m,
             **{k: np.array([f[k] for f in frames]) for k in ("x", "rho", "p", "u", "T_material", "T_rad")},
             series_t=series[:, 0], m_shock=series[:, 1], p_first=series[:, 2], T_first=series[:, 3],
             Trad_first=series[:, 4], steps=step, wall=time.time() - wall0)
    print(f"  ran     {path.name}: {step} steps, wall {time.time() - wall0:.0f} s", flush=True)


def main(argv=None):
    import logging
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", nargs="+", choices=list(tos.RUNS))
    ap.add_argument("--store-every", type=int, default=1000)
    ap.add_argument("--t-end", type=float, default=2e-9)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--pdrive", nargs="+", choices=list(PDRIVE))
    args = ap.parse_args(argv)
    for name in args.pdrive or ():
        run_pdrive(name, args.t_end, args.rerun)
    for name in args.dense or ():
        run_dense(name, args.store_every, args.t_end, args.rerun)


if __name__ == "__main__":
    main()

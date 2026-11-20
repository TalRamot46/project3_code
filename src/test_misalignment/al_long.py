"""Al (Test 6) to 50 ns: cold start and the solver's profile seeded at 2 ns, against the solver.

Why 2 ns (md/misalignment/01_surface_equilibrium.md section 3): the seeded one-temperature state survives only where the
solver's own profile has a large matter-radiation coupling t/t_eq (01_surface_equilibrium.md section 2
section 4.6).  Gold held from 0.1 ns, where t/t_eq = 6.7 (surface 3.5); Al reaches that at 2 ns
(7.5, surface 6.7), at 30 ns ~74 (gold's value at 2 ns) and at 50 ns ~113.

Runs (30 K target, paper bath, to T_END):
  <start>_<physics>   start: cold (t = 0) | seeded (solver at 2 ns, shock on a face, smeared)
                      physics: 2T (chi = 1) | chi1000 (2T, coupling x1000) | imc1..imc3 (IMC, seeds 1-3, 5000 + 5000 packets per step)

    python -m test_misalignment.al_long --run cold_2T seeded_2T
    python -m test_misalignment.al_long --t-end 1e-9 --run cold_imc1 --tag _probe
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
OUT = REPO / "results" / "misalignment" / "al_long" / "runs"                 # run data (untracked)
FIGURES = REPO / "results" / "misalignment" / "al_long" / "figures"         # figures and table (tracked)
T_END = 50e-9
T_SEED = 2e-9
T_TARGET = 30.0
DOMAIN_MASS = 0.19           # g/cm^2: 1.15 x the solver's shock mass at 50 ns (0.163)

# graded mass mesh [g/cm^2]: fine and equal across the ablated layer to 20 mg/cm^2 (m_f(50 ns) = 13.3),
# then 0.3% growth to 0.2 mg/cm^2 cells in the shocked region
MESH_COLD = (("mesh", "log_mass"), ("log_mass_first", 1e-7),
             ("log_mass_segments", ((1.05, 2e-5, 2e-2), (1.003, 2e-4, float("inf")))))
MESH_SEEDED = (("mesh", "log_mass"), ("log_mass_first", 2e-5),
               ("log_mass_segments", ((1.0, 2e-5, 2e-2), (1.003, 2e-4, float("inf")))))
IMC = dict(radiation_solver="imc", imc_seed=1, imc_n_emission=5000, imc_n_bath=5000, imc_max_census=50000)

RUNS = [f"{s}_{p}" for s in ("cold", "seeded") for p in ("2T", "chi1000", "imc1", "imc2", "imc3")]
TEXT = {"2T": r"2T diffusion, $\chi$ = 1", "chi1000": r"2T diffusion, $\chi$ = 1000", "imc": "IMC (mean of 3 seeds)"}


def case_and_config(name: str, t_end: float = T_END):
    start, physics = name.split("_")
    spec = ivd.HYDRO_CASES["test6_full"]
    base = ivd._hydro_case(spec, t_end)
    case = replace(base, x_max=DOMAIN_MASS / base.rho0, T_initial_Kelvin=T_TARGET, t_sec_end=t_end)
    cfg = dict(N=1000, store_every=50, show_plot=False, energy_diagnostics=True, dt_ramp_max=spec.dt_max,
               bath_drive="paper", **ivd.HYDRO_MATCHED)
    cfg.update(dict(MESH_SEEDED if start == "seeded" else MESH_COLD))
    if start == "seeded":
        case = replace(case, initial_condition="analytic_ablation", t_sec_start=T_SEED)
        cfg.update(seed_shock_on_face=True, seed_shock_smeared=True)
    if physics == "chi1000":
        case = replace(case, chi=1000.0)
    if physics.startswith("imc"):
        cfg.update(IMC, imc_seed=int(physics[3:] or 1))
    return case, cfg


def fronts(m, T, rho, p, rho_init):
    mh, ms, _ = ivd.fronts_of_profile(m, T, rho, p, rho_init)
    if not np.isfinite(mh):
        return mh, ms, np.nan
    mc = m - 0.5 * np.diff(m, prepend=0.0)
    j = int(np.searchsorted(mc, mh))
    return mh, ms, float(p[j:j + 15].max()) if j < m.size else np.nan


def run(name: str, t_end: float = T_END, tag: str = "", store_every: int | None = None, n_frames: int = 300):
    from hydro_sim.problems.simulation_config import SimulationConfig
    from rad_hydro_sim.simulation.iterator import simulate_rad_hydro

    OUT.mkdir(parents=True, exist_ok=True)
    case, cfg = case_and_config(name, t_end)
    if store_every:
        cfg["store_every"] = store_every
    wall = time.time()
    _, _, meta, hist = simulate_rad_hydro(case, SimulationConfig(**cfg))
    wall = time.time() - wall
    m = hist.m[0]
    rho_init = ivd._initial_density(m, case)
    fr = np.array([fronts(m, hist.T_material[k], hist.rho[k], hist.p[k], rho_init) for k in range(hist.t.size)])
    t0 = max(float(hist.t[0]), 1e-12)
    targets = np.geomspace(t0 * 1.0001, hist.t[-1], n_frames)
    keep = np.unique(np.concatenate(([int(np.argmin(np.abs(hist.t - x))) for x in targets], [hist.t.size - 1])))
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    b = meta["energy_budget"]
    err = float(np.max(np.abs(b["relative_error"])))
    path = OUT / f"{name}{tag}.npz"
    np.savez(path, t=hist.t[keep], m=m, x=f32(hist.x[keep]), rho=f32(hist.rho[keep]), p=f32(hist.p[keep]),
             u=f32(hist.u[keep]), T_material=f32(hist.T_material[keep]), T_rad=f32(hist.T[keep]),
             series_t=hist.t, m_heat=fr[:, 0], m_shock=fr[:, 1], p_dense=fr[:, 2],
             T_surface=hist.T_material[:, 0], Trad_surface=hist.T[:, 0],
             steps=len(b["t"]) - 1, wall=wall, max_energy_error=err, t_start=hist.t[0], chi=case.chi)
    print(f"  ran     {path.name}: {m.size} cells, {len(b['t']) - 1} steps, wall {wall:.0f} s, "
          f"energy error {err:.1e}", flush=True)


def main(argv=None):
    import logging
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", choices=RUNS)
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--tag", default="")
    ap.add_argument("--store-every", type=int)
    args = ap.parse_args(argv)
    for name in args.run or ():
        run(name, args.t_end, args.tag, args.store_every)


if __name__ == "__main__":
    main()

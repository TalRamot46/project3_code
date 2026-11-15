"""Test 3's cold start against the density exponent omega (md/phase_c/test3_omega_scan.md).

Gold, T_s = 1 HeV, 30 K target, cold start from t = 0 on the graded surface mesh, with
omega = 0.5 (Test 3; the runs of test3_on_the_solver.py), 0.4, 0.3, 0.2, 0.1 and 0.05, by IMC
(three seeds each) and by the 2T diffusion code at chi = 1.  Each run is compared with the semi-analytic solution of its own
omega.  The question: does the two-temperature deficit of the cold start depend on omega?

    python -m rad_hydro_sim.verification.test3_omega_scan --run imc 0.3 --seed 2
    python -m rad_hydro_sim.verification.test3_omega_scan --run 2T 0.05
    python -m rad_hydro_sim.verification.test3_omega_scan --run-all --jobs 2
    python -m rad_hydro_sim.verification.test3_omega_scan --figures
"""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import replace

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from rad_hydro_sim.verification import imc_vs_diffusion as ivd
from rad_hydro_sim.verification import test3_on_the_solver as t3

OUT = ivd.REPO / "results" / "test3_omega_scan"                               # run data (untracked)
FIGURES = ivd.REPO / "results" / "phase_c" / "figures" / "test3_omega_scan"   # figures (tracked)
OMEGAS = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
SEEDS = (1, 2, 3)
T_END = 1e-9                     # the new omegas run to 1 ns; everything is compared there
GAMMA = 1.25
# omega = 0.5 is Test 3: its runs are test3_on_the_solver's (to 2 ns, with a 1 ns snapshot)
TEST3 = {"imc": ["cold_IMC_30K_seed1", "cold_IMC_30K_seed2", "cold_IMC_30K_seed3"], "2T": ["cold_2T_30K"]}


def tag(omega: float) -> str:
    return f"{omega:.2f}".replace(".", "p")


def base_case(omega: float, t_end: float = T_END):
    case = ivd._hydro_case(ivd.HYDRO_CASES[t3.BASE], t_end)
    return replace(case, omega=float(omega), T_initial_Kelvin=30.0,
                   title=rf"Au, $\omega={omega:+.2f}$, 30 K")


def solver(omega: float):
    """The AblationSolver of this omega (built in ~10 s, then cached)."""
    from menahem_new.ablation_solver_og import AblationSolver
    from rad_hydro_sim.verification.menahem_comparison import _ablation_kwargs_from_case

    if abs(omega - 0.5) < 1e-12:
        from rad_hydro_sim.verification import test3_hydro_only as h
        return h.solver()
    path = OUT / f"solver_omega{tag(omega)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    s = AblationSolver(**_ablation_kwargs_from_case(base_case(omega)))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(s, f)
    return s


def case_and_config(omega: float, physics: str, t_end: float = T_END, seed: int = 1):
    """Test 3's cold start with another omega: the domain holds 1.1 x the shocked mass at t_end;
    the surface mesh is test3_on_the_solver's (the ablated mass does not depend on omega); the
    shocked region's cells grow 0.3% per cell up to 1/1060 of the domain."""
    s = solver(omega)
    m_end = 1.1 * float(np.ravel(s.shock_solver.shocked_mass(time=t_end))[0])
    case = replace(base_case(omega, t_end), x_max=float(s.shock_solver.initial_position(mass=m_end)))
    spec = ivd.HYDRO_CASES[t3.BASE]
    cfg = dict(N=spec.N, store_every=100, show_plot=False, energy_diagnostics=True,
               dt_ramp_max=spec.dt_max, bath_drive="paper", **ivd.HYDRO_MATCHED)
    cfg.update(mesh="log_mass", log_mass_first=4e-7,
               log_mass_segments=((1.05, 2.16e-5, 1.5e-3), (1.003, max(2.16e-5, m_end / 1060.0), float("inf"))),
               dt_first_frac=5e-24 / t_end)
    if physics == "imc":
        cfg.update(dict(spec.imc), radiation_solver="imc", imc_seed=seed)
    elif physics != "2T":
        raise ValueError(physics)
    return case, cfg


def run_path(omega: float, physics: str, seed: int = 1):
    return OUT / (f"cold_{physics}_30K_omega{tag(omega)}" + (f"_seed{seed}" if physics == "imc" else "") + ".npz")


def run(physics: str, omega: float, seed: int = 1, rerun: bool = False):
    path = run_path(omega, physics, seed)
    if path.exists() and not rerun:
        print(f"  cached  {path.name}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    case, cfg = case_and_config(omega, physics, seed=seed)
    t3.simulate_and_save(path, case, cfg)


def load(omega: float, physics: str):
    """The runs of (omega, physics): one, or Test 3's seeds."""
    if abs(omega - 0.5) < 1e-12:
        return [np.load(t3.OUT / f"{n}.npz") for n in TEST3[physics] if (t3.OUT / f"{n}.npz").exists()]
    paths = [run_path(omega, physics, k) for k in (SEEDS if physics == "imc" else (1,))]
    return [np.load(p) for p in paths if p.exists()]


def run_all(jobs: int = 2):
    """Every run of the scan that is not cached, cheapest first (the cost grows with omega)."""
    from concurrent.futures import ProcessPoolExecutor
    for w in OMEGAS[:0:-1]:
        solver(w)
    todo = [(ph, w, k) for w in OMEGAS[:0:-1] for ph, seeds in (("2T", (1,)), ("imc", SEEDS)) for k in seeds]
    with ProcessPoolExecutor(jobs) as pool:
        list(pool.map(run, *zip(*todo)))


def main(argv=None):
    import logging
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=2, metavar=("PHYSICS", "OMEGA"), help="imc or 2T, and omega")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    if args.run:
        run(args.run[0], float(args.run[1]), seed=args.seed, rerun=args.rerun)
    if args.run_all:
        run_all(args.jobs)
    if args.figures:
        from rad_hydro_sim.verification import test3_omega_scan_figures as figs
        figs.all_figures()


if __name__ == "__main__":
    main()

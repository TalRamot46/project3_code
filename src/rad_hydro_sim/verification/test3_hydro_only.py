"""Test 3's shocked region as a hydro-only problem (md/phase_c/test3_hydro_only.md).

The semi-analytic solution patches a self-similar heat wave to a self-similar shock: the heat
flux is zero beyond the ablation front, so the matter there is pure hydro, driven by the
pressure at the front, p_f(t) = p0 t^c3 (the PistonShock solver, pressure applied at m = 0).
This module tests that picture with the rad-hydro code's own hydro (same mesh, artificial
viscosity, compatible energy update and 300 K target), with the radiation switched off and the
left boundary driven by

- ``p_solver``:      the solver's p_f(t);
- ``p_diffusion``:   the pressure on the dense side of the heat front of the diffusion rad-hydro
                     run: the stored p_abl(t) times a constant (front_pressure_factor);
- ``p_imc``:         the same from the IMC runs (mean of the seeds);
- ``p_diffusion_raw``: the stored p_abl(t) as it is.  It is sampled where T = 0.3 T_s, two cells
                     on the hot side of the density jump, where the pressure is still 4.5%
                     below the value the shocked matter feels;
- ``piston``:        a piston moving with the solver's piston velocity, u_p(t).

Each drive runs from the cold target at t = 0 (``cold``) and from the PistonShock profile
seeded at 0.1 ns (``seeded``, the hydro-only twin of initial_condition="analytic_ablation").
The boundary is the material surface m = 0.  In the full problem the pressure acts at the
ablation front m_f(t), which moves through 0.5% of the shocked mass; the pressure difference
across that layer, m_f x acceleration, is below 0.1% of p_f.

    python -m rad_hydro_sim.verification.test3_hydro_only --run cold_p_solver   # one run
    python -m rad_hydro_sim.verification.test3_hydro_only --run-all --jobs 2
    python -m rad_hydro_sim.verification.test3_hydro_only --figures --table
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from hydro_sim.core.geometry import planar
from hydro_sim.core.integrator import compute_acceleration_nodes
from hydro_sim.core.eos import pressure_ideal_gas
from hydro_sim.core.state import RadHydroState
from hydro_sim.core.timestep import compute_dt_cfl
from hydro_sim.problems.simulation_config import SimulationConfig
from rad_hydro_sim.simulation.hydro_steps import (
    boundary_work, get_e_star_from_hydro, kinetic_energy, update_nodes_compatible)
from rad_hydro_sim.simulation.iterator import initialize_problem
from rad_hydro_sim.simulation.radiation_step import calculate_temperature_from_specific_energy
from rad_hydro_sim.verification import imc_vs_diffusion as ivd

REPO = ivd.REPO
OUT = REPO / "results" / "test3_hydro_only"                              # run data (untracked)
FIGURES = REPO / "results" / "phase_c" / "figures" / "test3_hydro_only"  # figures and table (tracked)
LABEL = "test3_smooth"            # Test 3 on its fine mesh (1237 cells)
SEED_LABEL = "test3_smooth_analytic"
T_SEED = 1e-10
SNAP_NS = (0.5, 1.0, 1.5, 2.0)
FIGURE_NS = (1.0, 1.5, 2.0)
GAMMA = 1.25

# name -> (start, drive)
RUNS = {
    "cold_p_solver": ("cold", "p_solver"),
    "cold_p_diffusion": ("cold", "p_diffusion"),
    "cold_p_diffusion_raw": ("cold", "p_diffusion_raw"),
    "cold_p_imc": ("cold", "p_imc"),
    "cold_piston": ("cold", "piston"),
    "seeded_p_solver": ("seeded", "p_solver"),
    "seeded_p_diffusion": ("seeded", "p_diffusion"),
    "seeded_piston": ("seeded", "piston"),
    # the target at 30 K instead of 300 K: 1/40 of the initial pressure (e ~ T^1.6)
    "seeded_p_solver_30K": ("seeded", "p_solver"),
}
T_INITIAL = {"seeded_p_solver_30K": 30.0}
RUN_TEXT = {
    "cold_p_solver": r"hydro only, $p_f(t)$ of the solver",
    "cold_p_diffusion": r"hydro only, $p_f(t)$ of the diffusion run",
    "cold_p_diffusion_raw": r"hydro only, the diffusion run's $p$ at $T=0.3\,T_s$",
    "cold_p_imc": r"hydro only, $p_f(t)$ of the IMC runs",
    "cold_piston": r"hydro only, the solver's piston $u_p(t)$",
    "seeded_p_solver": r"hydro only, $p_f(t)$ of the solver",
    "seeded_p_diffusion": r"hydro only, $p_f(t)$ of the diffusion run",
    "seeded_piston": r"hydro only, the solver's piston $u_p(t)$",
    "seeded_p_solver_30K": r"hydro only, $p_f(t)$ of the solver, 30 K target",
}


# --------------------------------------------------------------------------- the drives
def solver():
    from rad_hydro_sim.verification import bath_calibration as bc
    return bc.solver_for(LABEL)


def solver_pressure_law(s):
    hs = s.heat_solver
    return float(hs.Pf * hs.A**hs.a3 * hs.B**hs.b3), float(hs.c3)


def front_pressure_series(label: str, tags) -> tuple[np.ndarray, np.ndarray]:
    """(t, p at the heat front) of the stored rad-hydro runs of a case; several tags are averaged
    on the first one's times (log-log)."""
    series = []
    for tag in tags:
        run = np.load(ivd.DEFAULT_OUT / f"{label}_{tag}.npz")
        t, p = run["series_t"], run["p_abl"]
        ok = np.isfinite(p) & (t > 0) & (p > 0)
        series.append((t[ok], p[ok]))
    t0 = series[0][0]
    logs = [np.interp(np.log(t0), np.log(t), np.log(p)) for t, p in series]
    return t0, np.exp(np.mean(logs, axis=0))


def front_pressure_factor(label: str, tags) -> float:
    """(pressure of the shocked matter next to the front) / (stored p_abl), from the snapshots.

    p_abl is the pressure where T = 0.3 T_s (ivd.fronts_of_profile).  That point lies on the hot
    side of the density jump, where the pressure still rises toward the dense matter.  The
    pressure the shocked matter feels is the plateau just beyond the front, taken here as the
    mean over 2-10 heat-front masses.  The ratio is the same at the four snapshots to 0.3%
    (diffusion); it is applied to the whole series, including the start, where no snapshot
    checks it.
    """
    ratios = []
    for tag in tags:
        run = np.load(ivd.DEFAULT_OUT / f"{label}_{tag}.npz")
        ok = np.isfinite(run["p_abl"])
        for k, t in enumerate(run["t"]):
            m_f = np.interp(t, run["series_t"][ok], run["m_heat"][ok])
            p_abl = np.interp(t, run["series_t"][ok], run["p_abl"][ok])
            mc = run["m"][k] - 0.5 * np.diff(run["m"][k], prepend=0.0)
            ratios.append(run["p"][k][(mc > 2 * m_f) & (mc < 10 * m_f)].mean() / p_abl)
    return float(np.mean(ratios))


def tabulated_pressure(t_tab, p_tab, c3):
    """p(t): log-log interpolation of a table; before its first time, the solver's exponent
    through the first point (the stored series starts at ~0.03 ps)."""
    lt, lp = np.log(t_tab), np.log(p_tab)

    def p(t):
        if t <= t_tab[0]:
            return float(p_tab[0] * (t / t_tab[0])**c3)
        return float(np.exp(np.interp(np.log(t), lt, lp)))
    return p


def make_drive(start: str, drive: str, s):
    """('pressure', p(t)) or ('piston', u(t), du/dt(t))."""
    p0, c3 = solver_pressure_law(s)
    if drive == "p_solver":
        return ("pressure", lambda t: p0 * t**c3)
    if drive in ("p_diffusion", "p_imc", "p_diffusion_raw"):
        label = LABEL if start == "cold" else SEED_LABEL
        tags = [f"imc{k}" for k in (1, 2, 3)] if drive == "p_imc" else ["diffusion"]
        t_tab, p_tab = front_pressure_series(label, tags)
        if drive != "p_diffusion_raw":
            p_tab = p_tab * front_pressure_factor(label, tags)
        return ("pressure", tabulated_pressure(t_tab, p_tab, c3))
    if drive == "piston":
        ps = s.shock_solver
        u1 = float(ps.piston_velocity(time=1.0))
        n = (ps.omega + ps.tau) / (2.0 - ps.omega)            # u_p = u1 t^n
        return ("piston", lambda t: u1 * t**n, lambda t: n * u1 * t**(n - 1.0))
    raise ValueError(drive)


# --------------------------------------------------------------------------- the initial state
def cold_state(T_initial=None):
    """Test 3's cold 300 K target on the fine mesh, exactly as the rad-hydro runs start."""
    spec = ivd.HYDRO_CASES[LABEL]
    case = ivd._hydro_case(spec)
    if T_initial is not None:
        case = replace(case, T_initial_Kelvin=float(T_initial))
    config = SimulationConfig(N=spec.N, store_every=10**9, show_plot=False, **ivd.HYDRO_MATCHED,
                              **dict(spec.config))
    state = initialize_problem(case, config)
    return case, config, state


def seed_piston_shock(case, state, s, t0: float, n_sub: int = 8):
    """Overwrite the cold state behind the shock with the PistonShock solution at t0.

    As iterator._seed_ablation_profile, with the shock solution alone down to m = 0 (no ablated
    layer): the nodes behind the shock take the solution's positions and velocities, so the
    cells' densities are the solution's; the specific energy is the solution's mass average over
    each cell (cold target for the sub-points ahead of the shock).
    """
    ps = s.shock_solver
    w = case.omega
    m_nodes = case.rho0 * state.x**(1.0 - w) / (1.0 - w)
    dm = np.diff(m_nodes)
    m_shock = float(np.ravel(ps.shocked_mass(time=t0))[0])
    k = int(np.searchsorted(m_nodes, m_shock))               # nodes 0..k-1 are behind the shock
    sol = ps.solve(mass=m_nodes[1:k], time=t0)
    x, u = state.x.copy(), state.u.copy()
    x[1:k], u[1:k] = sol["position"], sol["velocity"]
    x[0], u[0] = sol["piston_position"], sol["piston_velocity"]
    rho = dm / np.diff(x)
    m_sub = m_nodes[:-1, None] + dm[:, None] * ((np.arange(n_sub) + 0.5) / n_sub)[None, :]
    behind = m_sub < m_shock
    x_sub = ((1.0 - w) * m_sub / case.rho0)**(1.0 / (1.0 - w))
    e_sub = case.f_Kelvin * case.T_initial_Kelvin**case.beta_Rosen * (case.rho0 * x_sub**(-w))**(-case.mu)
    sub = ps.solve(mass=m_sub[behind], time=t0)
    e_sub[behind] = sub["pressure"] / (case.r * sub["density"])
    e = state.e_material.copy()
    seeded = behind.any(axis=1)
    e[seeded] = e_sub[seeded].mean(axis=1)
    p = case.r * rho * e
    T = calculate_temperature_from_specific_energy(e, rho, case.f_Kelvin, case.beta_Rosen, case.mu)
    return RadHydroState(t=t0, x=x, u=u, a=np.zeros_like(x), V=np.diff(x), rho=rho, e_material=e, p=p,
                         q=np.zeros_like(rho), m_cells=state.m_cells, T_material=T, T_rad=T.copy(),
                         E_rad=np.zeros_like(rho))


# --------------------------------------------------------------------------- the step
def close_with_piston(state_old, star, case, dt, u_left, a_left, tol=1e-13, max_iter=100):
    """hydro_steps.update_nodes_compatible with the left node's velocity prescribed."""
    m = star.m_cells
    P_old = state_old.p + state_old.q
    e_new = star.e_material.copy()
    for _ in range(max_iter):
        p_new = pressure_ideal_gas(star.rho, e_new, gamma=case.r + 1)
        a_new = compute_acceleration_nodes(star.x, p_new, star.q, m, planar(), p_left=None, p_right=None)
        u_new = star.u + 0.5 * dt * a_new
        u_new[0] = u_left
        u_bar = 0.5 * (state_old.u + u_new)
        P_bar = 0.5 * (P_old + p_new + star.q)
        e_next = state_old.e_material - dt * P_bar * (u_bar[1:] - u_bar[:-1]) / m
        change = np.max(np.abs(e_next - e_new)) / max(np.max(np.abs(e_next)), 1e-300)
        e_new = e_next
        if change <= tol:
            break
    else:
        raise RuntimeError("piston close did not converge; reduce dt")
    p_new = pressure_ideal_gas(star.rho, e_new, gamma=case.r + 1)
    a_new = compute_acceleration_nodes(star.x, p_new, star.q, m, planar(), p_left=None, p_right=None)
    u_new = star.u + 0.5 * dt * a_new
    u_new[0], a_new[0] = u_left, a_left
    return RadHydroState(t=star.t, x=star.x, u=u_new, a=a_new, V=star.V, rho=star.rho, e_material=e_new,
                         p=p_new, q=star.q, m_cells=m, T_material=star.T_material, T_rad=star.T_rad,
                         E_rad=star.E_rad)


def run(name: str, t_end: float = 2e-9, rerun: bool = False, series_every: int = 500):
    start, drive_name = RUNS[name]
    path = OUT / f"{name}.npz"
    if path.exists() and not rerun:
        print(f"  cached  {path.name}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    s = solver()
    drive = make_drive(start, drive_name, s)
    case, config, state = cold_state(T_INITIAL.get(name))
    case = replace(case, scenario="hydro_only")
    if start == "seeded":
        state = seed_piston_shock(case, state, s, T_SEED)
    if drive[0] == "piston" and start == "cold":
        state.u[0] = 0.0
    rho_init = ivd._initial_density(np.cumsum(state.m_cells), case)
    m = np.cumsum(state.m_cells)
    snaps = [x * 1e-9 for x in SNAP_NS if state.t < x * 1e-9 <= t_end * (1 + 1e-12)]
    if not snaps or snaps[-1] < t_end * (1 - 1e-12):
        snaps.append(t_end)
    frames, series = [], []
    E0 = float(np.sum(state.m_cells * state.e_material)) + kinetic_energy(state.u, state.m_cells)
    work, step, dt_prev, wall0 = 0.0, 0, np.inf, time.time()
    t_start = state.t

    def record():
        excess = state.rho / rho_init - 1.0
        k = int(np.flatnonzero(excess >= 0.5 * excess.max())[-1])
        E = float(np.sum(state.m_cells * state.e_material)) + kinetic_energy(state.u, state.m_cells)
        p_left = 0.5 * (state.p[0] + state.q[0]) if drive[0] == "piston" else drive[1](max(state.t, 1e-300))
        series.append((state.t, m[k], state.u[0], state.p[0] + state.q[0], p_left, E, work))

    record()
    for t_snap in snaps:
        while state.t < t_snap:
            if step <= 2:
                dt = 1e-10 * t_end if state.t == 0.0 else min(1e-10 * t_end, 1e-3 * state.t)
            else:
                dt = min(compute_dt_cfl(state, GAMMA, config.CFL), 1.1 * dt_prev)
            dt = min(dt, t_snap - state.t)
            star = get_e_star_from_hydro(state, case.geom, case.r, config.sigma_visc, dt)
            t_new = state.t + dt
            if drive[0] == "pressure":
                new = update_nodes_compatible(state, star, case, np.zeros_like(star.rho), dt,
                                              float(drive[1](t_new)), "outflow", t_old=state.t)
            else:
                new = close_with_piston(state, star, case, dt, drive[1](t_new), drive[2](t_new))
            work += boundary_work(state, new, dt)
            new.t = t_snap if t_snap - t_new < 1e-9 * dt else t_new
            state, dt_prev, step = new, dt, step + 1
            if step % series_every == 0:
                record()
        T = calculate_temperature_from_specific_energy(state.e_material, state.rho, case.f_Kelvin,
                                                       case.beta_Rosen, case.mu)
        frames.append(dict(t=state.t, x=0.5 * (state.x[:-1] + state.x[1:]), rho=state.rho.copy(),
                           p=state.p.copy(), u=0.5 * (state.u[:-1] + state.u[1:]), T_material=T,
                           q=state.q.copy()))
        record()
        print(f"  {name}: t = {state.t * 1e9:.2f} ns, step {step}, wall {time.time() - wall0:.0f} s", flush=True)
    series = np.array(series)
    E_err = np.abs(series[:, 5] - E0 - series[:, 6]) / np.maximum(series[:, 5], 1e-300)
    np.savez(path, t=np.array([f["t"] for f in frames]), m=np.tile(m, (len(frames), 1)),
             **{k: np.array([f[k] for f in frames]) for k in ("x", "rho", "p", "u", "T_material", "q")},
             series_t=series[:, 0], m_shock=series[:, 1], u_left=series[:, 2], p_first_cell=series[:, 3],
             p_left=series[:, 4], energy=series[:, 5], work=series[:, 6],
             max_energy_error=float(E_err[1:].max()) if drive[0] == "pressure" else np.nan,
             steps=step, wall=time.time() - wall0, t_start=t_start)
    print(f"  ran     {path.name}: {step} steps, wall {time.time() - wall0:.0f} s, "
          f"max|energy err| {E_err[1:].max():.1e}", flush=True)


def run_all(jobs: int, names=None):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    OUT.mkdir(parents=True, exist_ok=True)

    def launch(name):
        with open(OUT / f"{name}.log", "w", encoding="utf-8") as f:
            return subprocess.run([sys.executable, "-m", "rad_hydro_sim.verification.test3_hydro_only",
                                   "--run", name], stdout=f, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        codes = list(pool.map(launch, names or list(RUNS)))
    print({n: c for n, c in zip(names or list(RUNS), codes)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", choices=list(RUNS))
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--t-end", type=float, default=2e-9)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args(argv)
    for name in args.run or ():
        run(name, t_end=args.t_end, rerun=args.rerun)
    if args.run_all:
        run_all(args.jobs)
    if args.figures or args.table:
        from rad_hydro_sim.verification import test3_hydro_only_figures as figs
        if args.figures:
            figs.all_figures()
        if args.table:
            figs.write_table()


if __name__ == "__main__":
    main()

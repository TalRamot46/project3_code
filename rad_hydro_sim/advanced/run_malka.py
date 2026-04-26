"""
Sweep PRESET_MALKA_HEIZLER and record rho_max/rho0 at t_end = 1 ns.

Set ``INTERMEDIATE_SWEEP`` below to choose the sweep:
  "rho0" — log-spaced initial density
  "t0"   — fixed rho0 = 19.32 g/cm^3, log-spaced boundary T0 in eV (50–200 eV)

Each sweep saves per-run density profiles (``density_*`` arrays and ``shock_cell_index``)
in the sweep ``.npz``, and writes a multi-panel PNG of every profile with the detected
shock marked (``rho0_sweep_density_profiles.png`` or ``t0_sweep_density_profiles.png``).

Simulations for distinct ``rho_0`` or ``T_0`` run in parallel (see ``INTERMEDIATE_MAX_WORKERS``),
using up to 10 worker processes and at most ``os.cpu_count()``. Progress is a **single** tqdm
line in the parent: mean ``t / t_end`` over all runs (workers report via shared memory).

Run from repo root:
  python project3_code/rad_hydro_sim/run_intermediate.py

Or call ``main_t0_sweep()`` from another module without editing this file.
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from tqdm import tqdm

# Before NumPy/BLAS: one thread per process when using multiprocessing (avoids oversubscription).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

_REPO_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from project3_code.hydro_sim.problems.simulation_config import get_results_dir
from project3_code.rad_hydro_sim.output_paths import get_rad_hydro_figures_dir
from project3_code.rad_hydro_sim.problems.presets_config import PRESET_MALKA_HEIZLER
from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro

# Select sweep when running this file as a script: "rho0" or "t0"
INTERMEDIATE_SWEEP: Literal["rho0", "t0"] = "rho0"

# Areal mass [g/cm^2] fixed to match preset (15 mg/cm^2)
AREAL_MASS_G_PER_CM2 = 3e-3

# Malka–Heizler reference density [g/cm^3]
RHO0_MALKA_G_CC = 19.32

# Log grid rho0 [g/cm^3]
RHO0_MIN = 0.1
RHO0_MAX = 19.32
NUM_RHO0_POINTS = 6

# Log grid drive temperature [eV] (code uses HeV = 100 eV per radiation_step)
T0_EV_MIN = 0.001
T0_EV_MAX = 100.0
NUM_T0_POINTS = 16

# Parallel simulations (i7-class: cap at 10 physical cores; never exceed os.cpu_count())
INTERMEDIATE_MAX_WORKERS = 10

# Filled by pool initializer on worker processes (Windows spawn cannot pickle ``Array`` into jobs).
_INTERMEDIATE_SHM: shared_memory.SharedMemory | None = None
_INTERMEDIATE_PROGRESS: np.ndarray | None = None


def _intermediate_pool_init(shm_name: str, n_jobs: int) -> None:
    """Attach worker to the parent-created shared-memory block (picklable ``shm_name`` only)."""
    global _INTERMEDIATE_SHM, _INTERMEDIATE_PROGRESS
    _INTERMEDIATE_SHM = shared_memory.SharedMemory(name=shm_name)
    _INTERMEDIATE_PROGRESS = np.ndarray(
        (n_jobs,), dtype=np.float64, buffer=_INTERMEDIATE_SHM.buf
    )


_T = TypeVar("_T")


def _parallel_sweep_with_shared_progress(
    worker_fn: Callable[[Any], _T],
    jobs: list[Any],
    n_jobs: int,
    max_workers: int,
    desc: str,
) -> list[_T]:
    """Run ``ProcessPoolExecutor`` with one ``SharedMemory`` buffer for per-job ``t/t_end`` fractions."""
    size = int(n_jobs * np.dtype(np.float64).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=size)
    try:
        progress = np.ndarray((n_jobs,), dtype=np.float64, buffer=shm.buf)
        progress[:] = 0.0
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_intermediate_pool_init,
            initargs=(shm.name, n_jobs),
        ) as ex:
            futures = [ex.submit(worker_fn, j) for j in jobs]
            _wait_pool_mean_progress(futures, progress, n_jobs, desc=desc)
            return [f.result() for f in futures]
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def _parallel_worker_count() -> int:
    return max(1, min(INTERMEDIATE_MAX_WORKERS, os.cpu_count() or 1))


def _wait_pool_mean_progress(
    futures: list[Future[object]],
    progress_arr: Any,
    n_jobs: int,
    desc: str,
    poll_s: float = 0.25,
) -> None:
    """One tqdm bar: mean completion fraction across ``progress_arr[0:n_jobs]``."""
    with tqdm(
        total=100,
        unit="%",
        desc=desc,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}% [{elapsed}<{remaining}]",
    ) as pbar:
        while True:
            s = 0.0
            for i in range(n_jobs):
                s += float(progress_arr[i])
            mean_pct = 100.0 * s / max(1, n_jobs)
            target_n = int(min(100, mean_pct))
            if target_n > pbar.n:
                pbar.update(target_n - pbar.n)
            for f in futures:
                if f.done():
                    ex = f.exception()
                    if ex is not None:
                        raise ex
            if all(f.done() for f in futures):
                if pbar.n < 100:
                    pbar.update(100 - pbar.n)
                break
            time.sleep(poll_s)


def t0_eV_to_kelvin(T_eV: float) -> float:
    """Boundary T0 [K] from T0 [eV]; KELVIN_PER_HEV converts one HeV (100 eV) to Kelvin."""
    return float(T_eV) / 100.0 * float(KELVIN_PER_HEV)


def find_shock_front(
    rho: np.ndarray,
    m_coordinate: np.ndarray,
) -> tuple[int, float]:
    """
    Rightmost index where d(rho)/dm changes sign (zero-crossing of gradient along m).
    Returns ``(-1, nan)`` if none found.
    """
    drho_dm = np.gradient(rho, m_coordinate)
    # start running from the right and find the first time of drho_dm below threshold=-0.1
    flag = False
    for i in range(len(drho_dm) - 1, -1, -1):
        if not flag and drho_dm[i] < -10:
            flag = True
        elif flag and drho_dm[i] > 0:
            return i, rho[i]
    
    # if we didn't find a subsonic shock front, proceeding to find a supersonic shock front
    # by finding the first zero crossing of drho_dm
    for i in range(len(drho_dm)):
        if drho_dm[i] < 0.001:
            return i, rho[i]
    return -1, float("nan")


def _run_single_rho0_job(
    job_pack: tuple[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Picklable worker: ``(sweep index, rho0)``; progress buffer from pool initializer."""
    job_index, rho0_new = int(job_pack[0]), float(job_pack[1])
    prog = _INTERMEDIATE_PROGRESS
    if prog is None:
        raise RuntimeError("intermediate sweep pool initializer did not set progress buffer")
    base_case, base_config = get_preset(PRESET_MALKA_HEIZLER)
    config_batch = replace(base_config, show_slider=False, show_plot=False)
    case_i = replace(
        base_case,
        rho0=rho0_new,
        x_max=AREAL_MASS_G_PER_CM2,
    )
    x_cells, state, _, _ = simulate_rad_hydro(
        case_i,
        config_batch,
        mp_progress=prog,
        mp_progress_index=job_index,
    )
    m_coordinate = np.cumsum(state.m_cells)
    rho = np.asarray(state.rho, dtype=float)
    # decrease noise near shock front by using window rolling mean
    import pandas as pd
    rho_rolling_mean = np.array(pd.Series(rho).rolling(window=5, center=True).mean().values)
    shock_idx, shock_density = find_shock_front(rho_rolling_mean, m_coordinate)
    return (
        rho_rolling_mean,
        m_coordinate,
        np.asarray(x_cells, dtype=float),
        int(shock_idx),
        float(shock_density),
    )

def _interpolate_x_max(T_eV: float) -> float:
    data = {
        0.001: 0.01 / RHO0_MAX / 1000,
        0.01: 0.012 / RHO0_MAX / 1000,
        0.02: 0.015 / RHO0_MAX / 1000,
        0.1: 0.013 / RHO0_MAX / 1000,
        1: 0.1 / RHO0_MAX / 1000,
        10: 0.2 / RHO0_MAX / 1000,
        20: 0.5 / RHO0_MAX / 1000,
        100: 3.0 / RHO0_MAX / 1000,
    }
    return np.interp(T_eV, list(data.keys()), list(data.values()))

def _run_single_t0_job(
    job_pack: tuple[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Picklable worker: ``(sweep index, T0_eV)`` at fixed rho0."""
    job_index, T_eV = int(job_pack[0]), float(job_pack[1])
    prog = _INTERMEDIATE_PROGRESS
    if prog is None:
        raise RuntimeError("intermediate sweep pool initializer did not set progress buffer")
    base_case, base_config = get_preset(PRESET_MALKA_HEIZLER)
    config_batch = replace(base_config, show_slider=False, show_plot=False)
    rho0_fixed = float(RHO0_MALKA_G_CC)
    x_max_fixed = _interpolate_x_max(T_eV)
    print(f"x_max_fixed: {x_max_fixed}")
    T0_K = t0_eV_to_kelvin(T_eV)
    case_i = replace(
        base_case,
        rho0=rho0_fixed,
        x_max=x_max_fixed,
        T0_Kelvin=T0_K,
    )
    x_cells, state, _, _ = simulate_rad_hydro(
        case_i,
        config_batch,
        mp_progress=prog,
        mp_progress_index=job_index,
    )
    m_coordinate = np.cumsum(state.m_cells)
    rho = np.asarray(state.rho, dtype=float)
    # decrease noise near shock front by using window rolling mean
    import pandas as pd
    rho_rolling_mean = np.array(pd.Series(rho).rolling(window=5, center=True).mean().values)
    shock_idx, shock_density = find_shock_front(rho_rolling_mean, m_coordinate)
    return (
        rho_rolling_mean,
        m_coordinate,
        np.asarray(x_cells, dtype=float),
        int(shock_idx),
        float(shock_density),
    )


def _plot_density_profiles_grid(
    m_profiles: np.ndarray,
    rho_profiles: np.ndarray,
    shock_fronts: np.ndarray,
    row_labels: list[str],
    suptitle: str,
    png_path: Path,
    rho0_reference: np.ndarray,
) -> None:
    """One subplot per profile: rho vs mass coordinate, shock front marked and labeled."""
    k, n = rho_profiles.shape
    ncols = min(4, k)
    nrows = int(math.ceil(k / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False, sharey=False
    )
    # create a color map for the profiles
    color_map = plt.get_cmap("viridis")
    for i in range(k):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        m = m_profiles[i]
        m = m * 1e3 # convert to mg/cm^2
        rho = rho_profiles[i]
        ax.plot(m, rho, color=color_map(i / k), lw=1.2)
        ax.axhline(
            float(rho0_reference[i]),
            color="0.45",
            ls=":",
            lw=0.9,
            alpha=0.85,
            label=r"$\rho_0$",
        )
        shock_cell_index = int(shock_fronts[i])
        if 0 <= shock_cell_index < n:
            m_s, r_s = float(m[shock_cell_index]), float(rho[shock_cell_index])
            ax.axvline(m_s, color="crimson", ls="--", lw=1.0)
            ax.scatter([m_s], [r_s], color="crimson", s=36, zorder=5, marker="o")
            ax.annotate(
                f"shock\n$\\rho$ = {r_s:.4g}",
                xy=(m_s, 0.8 * r_s),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=7,
                color="crimson",
                verticalalignment="bottom",
            )
        # also plot the gradient of rho vs m
        ax.set_title(row_labels[i], fontsize=9)
        ax.set_xlabel(r"$m$ [mg/cm$^2$]")
        ax.set_ylabel(r"$\rho$ [g/cm$^3$]")
        ax.grid(True, alpha=0.3)
    for j in range(k, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# decrease noise near shock front by using r-lowess
def r_lowess(x, y, frac=0.3):
    from scipy.stats import lowess
    return lowess(y, x, frac=frac)

def main_rho0_sweep() -> None:
    case_i, base_config = get_preset(PRESET_MALKA_HEIZLER)
    t_end_sec = case_i.t_sec_end * 1e9
    n_cells = int(base_config.N)
    workers = _parallel_worker_count()

    rho0_grid = np.logspace(
        np.log10(RHO0_MIN), np.log10(RHO0_MAX), NUM_RHO0_POINTS-5
    )
    rho0_grid = np.concatenate([rho0_grid, 
    np.logspace(np.log10(0.5), np.log10(1.1), 5)])
    k_run = len(rho0_grid)
    ratios = np.empty_like(rho0_grid)
    rho_max_arr = np.empty_like(rho0_grid)
    rho_profiles = np.empty((k_run, n_cells))
    m_profiles = np.empty((k_run, n_cells))
    x_profiles = np.empty((k_run, n_cells))
    shock_fronts = np.empty(k_run, dtype=np.int64)

    print(
        f"rho0 sweep: {k_run} simulations in parallel "
        f"({workers} worker process{'es' if workers != 1 else ''})"
    )
    jobs = [(i, float(rho0_grid[i])) for i in range(k_run)]
    results = _parallel_sweep_with_shared_progress(
        _run_single_rho0_job,
        jobs,
        k_run,
        workers,
        "rho0 sweep (mean)",
    )

    for k, rho0_new in enumerate(rho0_grid):
        rho0_new = float(rho0_new)
        rho, m_c, x_c, sidx, rmax = results[k]
        rho_profiles[k] = rho
        m_profiles[k] = m_c
        x_profiles[k] = x_c
        shock_fronts[k] = sidx
        rho_max_arr[k] = rmax
        ratios[k] = rmax / rho0_new if np.isfinite(rmax) else float("nan")

    out_dir = get_results_dir() / "rad_hydro_sim" / "intermediate_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "rho0_sweep_shock.npz"
    np.savez(
        npz_path,
        rho0_g_cc=rho0_grid,
        rho_max_g_cc=rho_max_arr,
        rho_max_over_rho0=ratios,
        density_rho_g_cc=rho_profiles,
        density_m_g_cm2=m_profiles,
        density_x_cm=x_profiles,
        shock_cell_index=shock_fronts,
        areal_mass_g_cm2=AREAL_MASS_G_PER_CM2,
    )
    print(f"Saved {npz_path}")

    csv_path = out_dir / "rho0_sweep_shock.csv"
    np.savetxt(
        csv_path,
        np.column_stack([rho0_grid, rho_max_arr, ratios]),
        delimiter=",",
        header="rho0_g_cc,rho_max_g_cc,rho_max_over_rho0",
        comments="",
    )
    print(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rho0_grid, ratios, "o-", color="C0", markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho_0$ [g/cm$^3$]")
    ax.set_ylabel(r"$\rho_{\mathrm{max}} / \rho_0$")
    ax.set_title(
        "Shock density ratio vs initial density "
        f"$t_\\mathrm{{end}}$={t_end_sec:.2e} s)"
    )
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()
    png_dir = get_rad_hydro_figures_dir() / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / "rho0_sweep_shock_compression.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")

    row_lbl = [rf"$\rho_0$ = {float(v):.4g} g/cm$^3$" for v in rho0_grid]
    prof_png = png_dir / "rho0_sweep_density_profiles.png"
    _plot_density_profiles_grid(
        m_profiles,
        rho_profiles,
        shock_fronts,
        row_lbl,
        rf"Density profiles ($\rho_0$ sweep, $t_{{\mathrm{{end}}}}$ = {t_end_sec:.2e} s)",
        prof_png,
        rho0_reference=rho0_grid,
    )
    print(f"Saved {prof_png}")


def main_T0_sweep() -> None:
    """Fixed rho0 = 19.32 g/cm^3; sweep T0_Kelvin via log-spaced T0 in eV (50–200 eV)."""
    case_i, base_config = get_preset(PRESET_MALKA_HEIZLER)
    t_end_sec = case_i.t_sec_end * 1e9
    rho0_fixed = RHO0_MALKA_G_CC
    workers = _parallel_worker_count()

    T0_eV_grid = np.logspace(
        np.log10(T0_EV_MIN), np.log10(T0_EV_MAX), NUM_T0_POINTS
    )
    T0_K_grid = np.array([t0_eV_to_kelvin(t) for t in T0_eV_grid])
    k_run = len(T0_eV_grid)
    n_cells = int(base_config.N)
    ratios = np.empty_like(T0_eV_grid)
    rho_max_arr = np.empty_like(T0_eV_grid)
    rho_profiles = np.empty((k_run, n_cells))
    m_profiles = np.empty((k_run, n_cells))
    x_profiles = np.empty((k_run, n_cells))
    shock_fronts = np.empty(k_run, dtype=np.int64)

    print(
        f"T0 sweep: {k_run} simulations in parallel "
        f"({workers} worker process{'es' if workers != 1 else ''})"
    )
    jobs = [(i, float(T0_eV_grid[i])) for i in range(k_run)]
    results = _parallel_sweep_with_shared_progress(
        _run_single_t0_job,
        jobs,
        k_run,
        workers,
        "T0 sweep (mean)",
    )

    for k in range(k_run):
        rho, m_c, x_c, sidx, rmax = results[k]
        rho_profiles[k] = rho
        m_profiles[k] = m_c
        x_profiles[k] = x_c
        shock_fronts[k] = sidx
        rho_max_arr[k] = rmax
        ratios[k] = rmax / rho0_fixed if np.isfinite(rmax) else float("nan")

    out_dir = get_results_dir() / "rad_hydro_sim" / "intermediate_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "t0_sweep_shock.npz"
    np.savez(
        npz_path,
        T0_eV=T0_eV_grid,
        T0_Kelvin=T0_K_grid,
        rho0_g_cc=np.full_like(T0_eV_grid, rho0_fixed),
        rho_max_g_cc=rho_max_arr,
        rho_max_over_rho0=ratios,
        density_rho_g_cc=rho_profiles,
        density_m_g_cm2=m_profiles,
        density_x_cm=x_profiles,
        shock_cell_index=shock_fronts,
        areal_mass_g_cm2=AREAL_MASS_G_PER_CM2,
    )
    print(f"Saved {npz_path}")

    csv_path = out_dir / "t0_sweep_shock.csv"
    np.savetxt(
        csv_path,
        np.column_stack([T0_eV_grid, T0_K_grid, rho_max_arr, ratios]),
        delimiter=",",
        header="T0_eV,T0_Kelvin,rho_max_g_cc,rho_max_over_rho0",
        comments="",
    )
    print(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T0_eV_grid, ratios, "o-", color="C1", markersize=6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$T_0$ [eV]")
    ax.set_ylabel(r"$\rho_{\mathrm{max}} / \rho_0$")
    ax.set_title(
        rf"Shock density ratio vs $T_0$ ($\rho_0$ = {rho0_fixed} g/cm$^3$, "
        f"$t_\\mathrm{{end}}$={t_end_sec:.2e} s)"
    )
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()
    png_dir = get_rad_hydro_figures_dir() / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / "t0_sweep_shock_compression.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")

    row_lbl = [rf"$T_0$ = {float(v):.4g} eV" for v in T0_eV_grid]
    prof_png = png_dir / "t0_sweep_density_profiles.png"
    rho0_ref = np.full(k_run, rho0_fixed)
    _plot_density_profiles_grid(
        m_profiles,
        rho_profiles,
        shock_fronts,
        row_lbl,
        rf"Density profiles ($T_0$ sweep, $\rho_0$ = {rho0_fixed} g/cm$^3$, $t_{{\mathrm{{end}}}}$ = {t_end_sec:.2e} s)",
        prof_png,
        rho0_reference=rho0_ref,
    )
    print(f"Saved {prof_png}")


if __name__ == "__main__":
    if INTERMEDIATE_SWEEP == "t0":
        main_T0_sweep()
    else:
        main_rho0_sweep()


"""
Reproduce Menahem-vs-rad_hydro x(t) comparison plots.

The script overlays:
- Eulerian trajectories x(t) for each Lagrangian mass cell (Menahem and simulation),
- piston trajectory x_piston(t),
- shock trajectory x_shock(t).

Default run reproduces tau=0 and can sweep extra taus.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
_MENAHEM_SOLVERS_DIR = Path(__file__).resolve().parents[1] / "menahem_solvers"
if str(_MENAHEM_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_SOLVERS_DIR))

from project3_code.menahem_solvers.piston_shock import PistonShock
from project3_code.menahem_solvers.ablation_solver import AblationSolver

from project3_code.rad_hydro_sim.output_paths import get_menahem_reproduction_figures_dir
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_CONSTANT_PRESSURE,
    PRESET_MALKA_HEIZLER,
)
from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro

S_PER_NS = 1e-9

def _ns_amplitude_rescale(amp: float, tau: float) -> float:
    """Convert drive amplitude defined with t[ns] into amplitude for t[s]."""
    return float(amp) * (1.0e9 ** float(tau))


def _pressure_drive_sim_units(t_sec: float, p0_barye: float, tau: float) -> float:
    """Boundary pressure used by rad_hydro integrator: P0 * (t/ns)^tau."""
    if t_sec <= 0.0:
        return 0.0
    return float(p0_barye) * ((float(t_sec) / S_PER_NS) ** float(tau))


def find_shock_front(
    rho: np.ndarray,
    m_coordinate: np.ndarray,
    *,
    rho_unshocked: float,
    gamma: float,
    Hugoniot_threshold: float = 0.5,
) -> tuple[int, float]:
    """
    Detect the shock as the right edge of the compressed region.

    Primary detector:
      rightmost index where rho exceeds the unshocked density by a small factor.
    Fallback detector:
      location of steepest negative density gradient.
    """
    rho_arr = np.asarray(rho, dtype=float)
    m_arr = np.asarray(m_coordinate, dtype=float)
    n = rho_arr.size
    if n < 3:
        return -1, float("nan")

    rho_thresh = float(rho_unshocked) * Hugoniot_threshold * (gamma + 1.0) / (gamma - 1.0)
    compressed = rho_arr > rho_thresh
    compressed_idx = np.flatnonzero(compressed)
    if compressed_idx.size > 0:
        i = int(compressed_idx[-1])
        return i, float(rho_arr[i])

    drho_dm = np.gradient(rho_arr, m_arr)
    i_steep = int(np.argmin(drho_dm))
    if np.isfinite(drho_dm[i_steep]):
        return i_steep, float(rho_arr[i_steep])
    return -1, float("nan")


def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(a, dtype=float)
    w = int(max(1, window))
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(np.asarray(a, dtype=float), kernel, mode="same")


def _parse_taus(raw: str) -> list[float]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        return [0.0]
    return [float(v) for v in vals]


def _sim_piston_position(
    x_cells: np.ndarray,
    p_cells: np.ndarray,
    p_drive: float,
    rel_tol: float,
) -> tuple[int, float]:
    """First (leftmost) cell close enough to drive pressure; fallback to nearest."""
    denom = max(abs(float(p_drive)), 1e-30)
    rel_err = np.abs(np.asarray(p_cells, dtype=float) - float(p_drive)) / denom
    close_idx = np.flatnonzero(rel_err <= rel_tol)
    if close_idx.size > 0:
        i = int(close_idx[0])
        return i, float(x_cells[i])
    i = int(np.argmin(rel_err))
    return i, float(x_cells[i])


def _ablation_kwargs_from_case(case) -> dict:
    tau = float(case.tau or 0.0)
    Tb = _ns_amplitude_rescale(float(case.T0_Kelvin), tau)
    heat = dict(
        Tb=Tb,
        tau=tau,
        g=float(case.g_Kelvin),
        alpha=float(case.alpha),
        lambdap=float(case.lambda_),
        f=float(case.f_Kelvin),
        beta=float(case.beta_Rosen),
        mu=float(case.mu),
        gamma=float(case.r) + 1.0,
    )
    return dict(
        Tb=heat["Tb"],
        tau=heat["tau"],
        g=heat["g"],
        alpha=heat["alpha"],
        lambdap=heat["lambdap"],
        f_heat=heat["f"],
        beta_heat=heat["beta"],
        mu_heat=heat["mu"],
        gamma_heat=heat["gamma"],
        rho0=float(case.rho0),
        omega=0.0,
        f_shock=heat["f"],
        beta_shock=heat["beta"],
        mu_shock=heat["mu"],
        gamma_shock=heat["gamma"],
    )


def _build_mass_grid_uniform(case, num_cells: int) -> np.ndarray:
    coordinate = np.linspace(0.0, float(case.x_max), int(num_cells) + 1)
    dx = np.diff(coordinate)
    mass_cells = np.full_like(dx, float(case.rho0)) * dx
    mass = np.cumsum(mass_cells)
    tiny_m = 1e-10 * mass[0]
    return np.concatenate([[tiny_m * 1e-5, tiny_m], mass])


def grid_comparison_hydro(
    *,
    tau: float,
    n_cells: int,
    omega: float,
    gamma: float,
    rho0: float,
    p0_barye: float,
    length_cm: float,
    t_end_sec: float | None,
    store_every: int,
    shock_smoothing_window: int,
    piston_rel_tol: float,
) -> Path:
    # Build Menahem solver (time in seconds, amplitude scaled from ns convention).
    p0_menahem = _ns_amplitude_rescale(p0_barye, tau)
    solver = PistonShock(
        rho0=float(rho0),
        omega=float(omega),
        p0=float(p0_menahem),
        tau=float(tau),
        gamma=float(gamma),
    )

    # If no end time is given, use shock reach time at x=L.
    t_end = float(t_end_sec) if t_end_sec is not None else float(
        solver.shock_time(shock_position=float(length_cm))
    )

    # rad_hydro case: pressure-driven hydro-only with matching units/params.
    case_base, config_base = get_preset(PRESET_CONSTANT_PRESSURE)
    case = replace(
        case_base,
        rho0=float(rho0),
        P0_Barye=float(p0_barye),
        tau=float(tau),
        r=float(gamma - 1.0),
        x_min=0.0,
        x_max=float(length_cm),
        t_sec_end=float(t_end),
        title=f"Menahem reproduction (tau={tau:g})",
    )
    config = replace(
        config_base,
        N=int(n_cells),
        store_every=max(1, int(store_every)),
        show_plot=False,
        show_slider=False,
    )

    _, _, _, history = simulate_rad_hydro(case, config)

    times = np.asarray(history.t, dtype=float)
    m_grid = np.asarray(history.m[0], dtype=float)

    # Menahem trajectories sampled on simulation times and mass grid.
    x_menahem = np.zeros((times.size, m_grid.size), dtype=float)
    x_piston_menahem = np.zeros(times.size, dtype=float)
    x_shock_menahem = np.zeros(times.size, dtype=float)
    for k, t in enumerate(times):
        if t <= 0.0:
            t_eval = 1e-18
        else:
            t_eval = float(t)
        sol = solver.solve(mass=m_grid, time=t_eval)
        x_menahem[k, :] = np.asarray(sol["position"], dtype=float)
        x_piston_menahem[k] = float(solver.piston_position(time=t_eval))
        x_shock_menahem[k] = float(solver.shock_position(time=t_eval))

    # Simulation piston/shock trajectories.
    x_piston_sim = np.full(times.size, np.nan, dtype=float)
    x_shock_sim = np.full(times.size, np.nan, dtype=float)
    for k, t in enumerate(times):
        xk = np.asarray(history.x[k], dtype=float)
        pk = np.asarray(history.p[k], dtype=float)
        rhok = np.asarray(history.rho[k], dtype=float)
        mk = np.asarray(history.m[k], dtype=float)

        p_drive = _pressure_drive_sim_units(t, p0_barye, tau)
        _, x_p = _sim_piston_position(xk, pk, p_drive, rel_tol=piston_rel_tol)
        x_piston_sim[k] = x_p

        rhok_smooth = _rolling_mean(rhok, shock_smoothing_window)
        ishock, _ = find_shock_front(
            rhok_smooth,
            mk,
            rho_unshocked=rho0,
            gamma=gamma,
            Hugoniot_threshold=0.5,
        )
        # if t > 0.005 * np.max(times):
        #     # show plot of rho smooth vs mk + designated shock front
        #     plt.title("t = " + str(t) + "ishock = " + str(ishock))
        #     plt.plot(mk, rhok)
        #     plt.plot(mk[ishock], rhok[ishock], "ro")
        #     plt.show()

        #     # plot also drho_dm vs mk
        #     drho_dm = np.gradient(rhok, mk)
        #     i_steep = int(np.argmin(drho_dm))
        #     if np.isfinite(drho_dm[i_steep]):
        #         print("i_steep = " + str(i_steep))
        #     else:
        #         print(f"i_steep = {i_steep} is not finite")
        #     plt.plot(mk, drho_dm)
        #     plt.plot(mk[i_steep], drho_dm[i_steep], "ro")
        #     plt.show()


        if ishock >= 1:
            x_shock_sim[k] = float(xk[ishock])

    # Plot x(t): each mass-cell trajectory + boundaries.
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    x_sim = np.asarray(history.x, dtype=float)
    matched_sim_coordinates = np.zeros(x_sim.shape, dtype=float)
    matched_sim_coordinates[:,1:] = 0.5*(x_sim[:,1:] + x_sim[:,:-1]) 

    chosen_cell_indices = np.linspace(0, x_sim.shape[1]-2, num=10, dtype=int)
    for j in chosen_cell_indices:
        ax.plot(times, matched_sim_coordinates[:, j+1], color="blue", lw=0.4, alpha=0.7)
        ax.plot(times, x_menahem[:, j], color="black", lw=0.4, alpha=0.7)

    # ax.plot(times, x_shock_sim, color="red", lw=2.0, label="shock Simulation")
    ax.plot(times, x_shock_menahem, color="blue", lw=1.8, ls="--", label="shock Analytic")
    ax.plot(times, x_piston_sim, color="black", lw=1.6, label="piston Simulation")
    ax.plot(times, x_piston_menahem, color="green", lw=1.8, ls="--", label="piston Analytic")

    ax.set_xlabel(r"$t$ [sec]")
    ax.set_ylabel(r"$x$ [cm]")
    ax.set_title(rf"$\omega={omega:g},\ \tau={tau:g},\ \gamma={gamma:g},\ N_{{cells}}={n_cells}$")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(times.min(), times.max())

    out_dir = get_menahem_reproduction_figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_tag = str(tau).replace(".", "p").replace("-", "m")
    import time
    time_signature = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"xt_menahem_vs_sim_tau_{tau_tag}_time_{time_signature}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.show()
    plt.close(fig)
    return out_path


def grid_comparison_rad_hydro(
    *,
    tau: float,
    n_cells: int,
    t_end_sec: float,
    store_every: int,
    shock_smoothing_window: int,
) -> Path:
    # Full rad-hydro simulation case (temperature drive + radiation coupling).
    case_base, config_base = get_preset(PRESET_MALKA_HEIZLER)
    case = replace(
        case_base,
        tau=float(tau),
        t_sec_end=float(t_end_sec),
        title=f"Rad-hydro grid comparison (tau={tau:g})",
    )
    config = replace(
        config_base,
        N=int(n_cells),
        store_every=max(1, int(store_every)),
        show_plot=False,
        show_slider=False,
    )
    _, _, _, history = simulate_rad_hydro(case, config)
    if case.rho0 is None:
        raise ValueError("Selected rad_hydro preset must define rho0.")
    if case.r is None:
        raise ValueError("Selected rad_hydro preset must define EOS r (gamma-1).")

    times = np.asarray(history.t, dtype=float)
    x_sim = np.asarray(history.x, dtype=float)
    matched_sim_coordinates = np.zeros_like(x_sim, dtype=float)
    matched_sim_coordinates[:, 1:] = 0.5 * (x_sim[:, 1:] + x_sim[:, :-1])

    # Menahem full model (AblationSolver) sampled on same times and mass grid.
    ablation_solver = AblationSolver(**_ablation_kwargs_from_case(case))
    mass_grid = _build_mass_grid_uniform(case, n_cells)
    times_model = np.linspace(0.0, times.max(), num=100, dtype=float)
    results = [ablation_solver.solve(mass=mass_grid, time=max(float(t), 1e-18)) for t in times_model]
    position_times = np.array([r["position"] for r in results]).T
    shock_position = np.array([r["shock_position"] for r in results], dtype=float)
    piston_position = np.array([r["piston_position"] for r in results], dtype=float)
    heat_position = np.array([r["heat_position"] for r in results], dtype=float)
    boundary_position = np.array([r["boundary_position"] for r in results], dtype=float)

    # Simulation shock from density profile (for full rad-hydro this is robust enough).
    x_shock_sim = np.full(times.size - 1, np.nan, dtype=float)
    for k in range(1, times.size):
        rhok = np.asarray(history.rho[k], dtype=float)
        mk = np.asarray(history.m[k], dtype=float)
        rhok_smooth = _rolling_mean(rhok, shock_smoothing_window)
        ishock, _ = find_shock_front(
            rhok_smooth,
            mk,
            rho_unshocked=float(case.rho0),
            gamma=float(case.r) + 1.0,
            Hugoniot_threshold=0.5,
        )
        if ishock >= 1:
            x_shock_sim[k - 1] = float(x_sim[k, ishock])

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    chosen_cell_indices = np.linspace(0, x_sim.shape[1] - 2, num=min(30, x_sim.shape[1] - 1), dtype=int)
    for j in chosen_cell_indices:
        ax.plot(times, matched_sim_coordinates[:, j + 1], color="royalblue", lw=0.45, alpha=0.55)
    for j in chosen_cell_indices:
        ax.plot(times_model, position_times[j + 2], color="black", lw=0.45, alpha=0.6)

    ax.plot(times[1:], x_shock_sim, lw=2.0, c="red", label="shock Simulation")
    ax.plot(times[1:], shock_position, lw=2.0, ls="--", c="blue", label="shock Analytic")
    ax.plot(times[1:], piston_position, lw=1.6, ls="--", c="green", label="piston Analytic")
    ax.plot(times[1:], heat_position, lw=1.6, ls="--", c="fuchsia", label="heat Analytic")
    ax.plot(times[1:], boundary_position, lw=1.6, ls="--", c="black", label="boundary Analytic")

    ax.set_xlabel(r"$t$ [sec]")
    ax.set_ylabel(r"$x$ [cm]")
    ax.set_title(rf"Full rad-hydro comparison, $\tau={tau:g}$, $N_{{cells}}={n_cells}$")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(times.min(), times.max())

    out_dir = get_menahem_reproduction_figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_tag = str(tau).replace(".", "p").replace("-", "m")
    import time
    time_signature = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"xt_menahem_vs_full_rad_hydro_tau_{tau_tag}_time_{time_signature}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    # Script configuration (edit here instead of command-line arguments).
    comparison_mode = "rad_hydro"  # "hydro" or "rad_hydro"
    # taus_raw = "0.0"
    # n_cells = 100
    # omega = 0.0
    # gamma = 1.25
    # rho0 = 1.0
    # p0_barye = 1.0
    # length_cm = 1.0
    # t_end_sec = None
    # store_every = 1
    # shock_smoothing_window = 5
    # piston_rel_tol = 0.05

    taus = [0.123]
    n_cells = 100
    omega = 0.
    gamma = 1.25
    rho0 = 19.32
    p0_barye = 1.0
    t_end_sec = 1e-9
    store_every = 1
    shock_smoothing_window = 5
    piston_rel_tol = 0.05
    print(f"Running taus={taus}")
    output_paths: list[Path] = []
    for tau in taus:
        if comparison_mode == "hydro":
            out = grid_comparison_hydro(
                tau=tau,
                n_cells=n_cells,
                omega=omega,
                gamma=gamma,
                rho0=rho0,
                p0_barye=p0_barye,
                length_cm=None,
                t_end_sec=t_end_sec,
                store_every=store_every,
                shock_smoothing_window=shock_smoothing_window,
                piston_rel_tol=piston_rel_tol,
            )
        elif comparison_mode == "rad_hydro":
            out = grid_comparison_rad_hydro(
                tau=tau,
                n_cells=n_cells,
                t_end_sec=t_end_sec,
                store_every=store_every,
                shock_smoothing_window=shock_smoothing_window,
            )
        else:
            raise ValueError("comparison_mode must be either 'hydro' or 'rad_hydro'")
        output_paths.append(out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()

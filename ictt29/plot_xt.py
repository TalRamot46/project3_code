# ictt29/plot_xt.py
"""
Space-time (x-t) trajectory and front plotting.

Runs the 1D Rad-Hydro simulation (or loads from cache) and plots
cell boundaries x(t) with diagnosed fronts (Simulation vs Analytic solver
vs Analytic fits).

Front-detection and fit-trajectory utilities are shared with
``full_fitting_eulerian.py`` via ``sim_front_utils``.
"""

import sys
import pickle
from pathlib import Path
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.integrate

# Monkeypatch scipy.integrate.simps for modern SciPy versions
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

# Monkeypatch numpy.trapz for NumPy 2.x
if not hasattr(np, "trapz"):
    if hasattr(scipy.integrate, "trapezoid"):
        np.trapz = scipy.integrate.trapezoid
    else:
        np.trapz = scipy.integrate.trapz


# ---------------------------------------------------------------------------
# Package / solver path setup
# ---------------------------------------------------------------------------
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

_MENAHEM_DIR = Path(__file__).resolve().parents[1] / "menahem_new"
if str(_MENAHEM_DIR) not in sys.path:
    sys.path.insert(0, str(_MENAHEM_DIR))

_ICTT29_DIR = Path(__file__).resolve().parent
if str(_ICTT29_DIR) not in sys.path:
    sys.path.insert(0, str(_ICTT29_DIR))

from project3_code.rad_hydro_sim.problems.presets_utils import get_preset
from project3_code.rad_hydro_sim.problems.presets_config import (
    PRESET_FIG_8_CONSTANT_TEMPERATURE,
)
from project3_code.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project3_code.rad_hydro_sim.verification.menahem_comparison import (
    _ablation_kwargs_from_case,
    _build_mass_grid,
)
from project3_code.rad_hydro_sim.simulation.radiation_step import KELVIN_PER_HEV

from ablation_solver_og import AblationSolver

# Fitting pipelines — needed to overlay analytic-fit front trajectories
from sub_fitting import perform_subsonic_fitting  # type: ignore
from shock_fitting import perform_shock_fitting    # type: ignore

# ---------------------------------------------------------------------------
# Shared front-detection / fit-trajectory utilities
# ---------------------------------------------------------------------------
from sim_front_utils import (
    _rolling_mean,
    detect_sim_ablation_boundary,
    detect_sim_ablation_front,
    detect_sim_shock_front_trajectory,
    compute_fit_front_trajectories,
)


# =============================================================================
# Plotting function
# =============================================================================

def plot_xt_trajectories(
    history,
    case,
    xt_path: str,
    case_title: str,
    ablation_solver=None,
    sub_params=None,
    shock_params=None,
):
    """Plot cell boundaries x(t) and diagnosed fronts.

    Compares three sources:
    * **Simulation** — density-based front detection on history arrays.
    * **Analytic solver** (Menahem AblationSolver) — exact positions.
    * **Analytic fits** — power-law fits, shown only when *sub_params* and
      *shock_params* are provided (obtained from ``full_fitting_eulerian``).

    Parameters
    ----------
    history:
        RadHydroHistory object with fields ``.t``, ``.x``, ``.rho``, ``.m``.
    case:
        RadHydroCase with ``.rho0``, ``.r``, ``.x_max``.
    xt_path:
        Output PNG path.
    case_title:
        Human-readable title for the plot.
    ablation_solver:
        Pre-built AblationSolver (built if None).
    sub_params, shock_params:
        Optional fit parameters from the fitting pipeline.  When both are
        provided the fit front trajectories are overlaid on the plot.
    """
    print(f"Generating space-time (xt) plot for {case_title}...")
    times_full = np.asarray(history.t, dtype=float)
    x_sim_full = np.asarray(history.x, dtype=float)
    # x_sim_full shape: (K, Ncells)  — cell centres, NOT node positions
    n_cells = x_sim_full.shape[1]   # correct: Ncells, not Ncells-1

    # ------------------------------------------------------------------
    # 1)  AblationSolver (Menahem) analytic solution
    # ------------------------------------------------------------------
    if ablation_solver is None:
        ablation_solver = AblationSolver(**_ablation_kwargs_from_case(case))
    mass_grid = _build_mass_grid(case, num_cells=n_cells)

    # Downsample to ~200 uniformly-spaced time frames for the solver calls
    # (the solver is expensive; the simulation history may have thousands).
    stride = max(1, len(times_full) // 200)
    mask = range(1, len(times_full), stride)
    mask_list = list(mask)          # original frame indices

    times_model = times_full[mask_list]
    x_sim       = x_sim_full[mask_list]   # (N_sampled, Ncells)

    results = []
    for t in times_model:
        sol = ablation_solver.solve(mass=mass_grid, time=max(float(t), 1e-18))
        results.append(sol)

    position_times    = np.array([r["position"]          for r in results]).T
    shock_position    = np.array([r["shock_position"]    for r in results], dtype=float)
    piston_position   = np.array([r["piston_position"]   for r in results], dtype=float)
    heat_position     = np.array([r["heat_position"]     for r in results], dtype=float)
    boundary_position = np.array([r["boundary_position"] for r in results], dtype=float)

    # Absolute ablation-boundary: heat_position is the lab-frame position of
    # the heat-wave interface; boundary_position is relative to it.
    abs_boundary_position = boundary_position + heat_position

    # ------------------------------------------------------------------
    # 2)  Simulation fronts (using shared utilities — correct indexing)
    # ------------------------------------------------------------------
    # Pre-extract the downsampled density / mass arrays so detection uses
    # the same frames as times_model / x_sim.
    rho_sampled = np.array([history.rho[i] for i in mask_list], dtype=float)
    m_sampled   = np.array([history.m[i]   for i in mask_list], dtype=float)

    # Shock front (with early-time log-log extrapolation)
    x_shock_sim = detect_sim_shock_front_trajectory(
        rho_sampled,
        m_sampled,
        x_sim,
        rho_unshocked=float(case.rho0),
        gamma=float(case.r) + 1.0,
        smooth_window=5,
        extrap_t_ns=0.002,
        extrap_times=times_model,
    )
    # x_shock_sim[0] is always nan (t=0); align with times_model[1:]
    x_shock_sim_plot = x_shock_sim[1:]   # shape (N_sampled-1,)

    # Ablation boundary: left edge of the leftmost cell
    x_boundary_sim = detect_sim_ablation_boundary(x_sim)   # (N_sampled,) cm

    # Ablation front: leftmost cell above the Hugoniot threshold
    x_ablation_front_sim = detect_sim_ablation_front(
        rho_sampled,
        m_sampled,
        x_sim,
        rho_unshocked=float(case.rho0),
        gamma=float(case.r) + 1.0,
        smooth_window=5,
    )   # (N_sampled,) cm

    # ------------------------------------------------------------------
    # 3)  Analytic-fit front trajectories (optional)
    # ------------------------------------------------------------------
    fit_fronts = None
    if sub_params is not None and shock_params is not None:
        try:
            fit_fronts = compute_fit_front_trajectories(
                times_model, ablation_solver, sub_params, shock_params
            )
        except Exception as exc:
            print(f"  [plot_xt] Could not compute fit fronts: {exc}")

    # ------------------------------------------------------------------
    # 4)  Build figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # --- background Lagrangian cell trajectories ---
    NUM_PRESENTED_CELLS = 100
    chosen_cell_indices = np.round(
        np.linspace(0, n_cells - 1, NUM_PRESENTED_CELLS)
    ).astype(int)

    legend_added = False
    for j in chosen_cell_indices:
        lbl_sim = "Simulation Cells" if not legend_added else None
        lbl_men = "Analytic Cells"   if not legend_added else None
        ax.plot(times_model, x_sim[:, j],
                color="black", lw=0.4, alpha=0.8, label=lbl_sim)
        # position_times has shape (len(mass_grid), N_sampled); the first two
        # rows correspond to the two sentinel mass entries prepended in
        # _build_mass_grid, so mass index j maps to position_times[j+2].
        if j + 2 < position_times.shape[0]:
            ax.plot(times_model, position_times[j + 2],
                    color="blue", lw=0.4, alpha=0.7, label=lbl_men)
        legend_added = True

    # --- Simulation fronts (solid, bold) ---
    ax.plot(times_model[1:], x_shock_sim_plot,
            lw=2.5, c="red",     label="Shock front (simulation)")
    ax.plot(times_model, x_boundary_sim,
            lw=2.0, c="black",   label="Ablation boundary (simulation)")
    ax.plot(times_model, x_ablation_front_sim,
            lw=2.0, c="fuchsia", label="Ablation front (simulation)", alpha=0.85)

    # --- Analytic solver fronts (dashed) ---
    ax.plot(times_model, shock_position,
            lw=2.0, ls="--", c="darkred", label="Shock (solver)")
    ax.plot(times_model, piston_position,
            lw=2.0, ls="--", c="green",   label="Piston (solver)")
    ax.plot(times_model, heat_position,
            lw=2.0, ls="--", c="purple",  label="Heat wave (solver)")
    ax.plot(times_model, abs_boundary_position,
            lw=2.0, ls="--", c="grey",    label="Boundary (solver)")

    # --- Fit fronts (dotted), only when params are available ---
    if fit_fronts is not None:
        ax.plot(times_model, fit_fronts["shock"],
                lw=2.0, ls=":", c="orange",  label="Shock (fit)")
        ax.plot(times_model, fit_fronts["piston"],
                lw=2.0, ls=":", c="cyan",    label="Piston (fit)")
        ax.plot(times_model, fit_fronts["ablation_front"],
                lw=2.0, ls=":", c="violet",  label="Ablation front (fit)")
        ax.plot(times_model, fit_fronts["boundary"],
                lw=2.0, ls=":", c="dimgray", label="Boundary (fit)")

    ax.set_xlabel("time [sec]",     fontsize=12)
    ax.set_ylabel("position [cm]",  fontsize=12)
    ax.set_title(
        f"Space-Time (xt) Trajectories and Fronts\n{case_title}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax.set_xlim([0.0, times_model[-1]])
    ax.set_ylim([-0.1 * x_sim.max(), x_sim.max()])

    fig.tight_layout()
    fig.savefig(xt_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {xt_path}")


# =============================================================================
# Main Orchestration
# =============================================================================

def run_preset_workflow(
    preset_name: str,
    case_label: str,
    case_title: str,
    sub_params=None,
    shock_params=None,
):
    """Run full verification comparison pipeline for a given preset.

    Loads (or runs) the simulation and the AblationSolver, runs the subsonic
    and shock fitting pipelines to obtain *sub_params* / *shock_params*, then
    generates the x-t trajectory plot with all three front sources overlaid:
    simulation, analytic solver, and analytic fits.

    Callers can supply pre-computed *sub_params* / *shock_params* to skip the
    fitting step (e.g. when called from ``full_fitting_eulerian``).
    """
    print("=" * 80)
    print(f"PROCESSING PRESET: {preset_name} -> {case_label}")
    print("=" * 80)

    from data_loader import get_sim_history, get_ablation_solver
    case, history = get_sim_history(preset_name, case_label)
    ablation_solver = get_ablation_solver(case, case_label)

    # Run fitting pipelines if params were not supplied by the caller.
    if sub_params is None:
        print("--- Running Subsonic Ablation Fitting ---")
        sub_params = perform_subsonic_fitting(ablation_solver.heat_solver)
    if shock_params is None:
        print("--- Running Piston Shock Fitting ---")
        shock_params = perform_shock_fitting(ablation_solver.shock_solver)

    # Generate x-t plot
    out_dir = Path("results/ictt") / case_label / "eulerian_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    xt_path = str(out_dir / f"{case_label}_xt.png")
    plot_xt_trajectories(
        history, case, xt_path, case_title,
        ablation_solver=ablation_solver,
        sub_params=sub_params,
        shock_params=shock_params,
    )

    print(f"Preset {preset_name} x-t plot generated successfully.")


def main():
    from project3_code.rad_hydro_sim.problems.presets_config import (
        PRESET_FIG_8_CONSTANT_TEMPERATURE,
        PRESET_FIG_9_CONSTANT_FLUX,
        PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
    )
    run_preset_workflow(
        PRESET_FIG_8_CONSTANT_TEMPERATURE,
        "const_T",
        "Constant Boundary Temperature (tau=0)",
    )
    run_preset_workflow(
        PRESET_FIG_9_CONSTANT_FLUX,
        "const_S",
        "Fig 9 Constant Flux Drive",
    )
    run_preset_workflow(
        PRESET_FIG_10_CONSTANT_ABLATION_PRESSURE,
        "const_P_shock",
        "Fig 10 Constant Ablation Pressure Drive",
    )
    print("\nAll x-t trajectory plots completed successfully!")


if __name__ == "__main__":
    main()

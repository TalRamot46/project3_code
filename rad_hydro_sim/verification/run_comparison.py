# verification/run_comparison.py
"""
Unified runner for rad_hydro_sim verification comparisons.

Compares:
  1. Radiation-only: run_rad_hydro (radiation_only_constant_temperature_drive)
     vs 1D Diffusion self similar in gold (constant temperature drive)
     vs Supersonic solver (radiation self-similar, same physics).
  2. Hydro-only: run_rad_hydro (hydro_only_power_law_pressure_drive)
     vs hydro_sim run_hydro (matching driven shock case).
  3. Full rad_hydro: run_rad_hydro (constant temperature drive) vs Shussman piecewise
     reference (subsonic solver to shock front; shock solver driven by front pressure).

Usage:
  # In main(), set MODE and run:
  python -m project_3.rad_hydro_sim.verification.run_comparison

  Or from repo root:
  python project_3/rad_hydro_sim/verification/run_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project_3 is on path (when run as script): add parent of repo root so "project_3" package resolves
_REPO_PARENT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from project_3.rad_hydro_sim.verification.verification_config import (
    VerificationMode,
    RADIATION_ONLY_PRESET,
    HYDRO_ONLY_PRESET,
    FULL_RAD_HYDRO_PRESET,
    make_verification_output_paths,
)
from project_3.rad_hydro_sim.verification.radiation_data import (
    RadiationData,
    rad_hydro_history_to_radiation_data,
    diffusion_output_to_radiation_data,
    supersonic_output_to_radiation_data,
)
from project_3.rad_hydro_sim.verification.compare_radiation_plots import (
    plot_radiation_comparison_single_time,
    plot_radiation_comparison_slider,
)


# =============================================================================
# Radiation-only: run rad_hydro + 1D Diffusion + Supersonic solver, then compare T and E_rad
# =============================================================================

def run_supersonic_solver_reference(
    case,
    n_times: int = 30,
    iternum: int = 30,
    xsi0: float = 1.0,
) -> Optional[RadiationData]:
    """
    Run the supersonic (radiation self-similar) solver with parameters matching the RadHydroCase.

    Builds a MaterialSuper from the case (alpha, beta=gamma, lambda_, mu, rho0, f, g, sigma),
    uses case.tau (0 for constant temperature drive), and maps dimensionless time to
    physical time via t_end. Returns RadiationData in the same format as diffusion reference.
    """
    try:
        from project_3.shussman_solvers.supersonic_solver import (
            MaterialSuper,
            STEFAN_BOLTZMANN,
            compute_profiles_for_report,
        )
    except ImportError as e:
        print(f"  Could not import supersonic solver: {e}, skipping.")
        return None

    # Material matching the preset (same opacity/EOS as rad_hydro)
    mat = MaterialSuper(
        alpha=float(case.alpha),
        beta=float(case.gamma),
        lambda_=float(case.lambda_),
        mu=float(case.mu),
        rho0=float(case.rho0),
        f=float(case.f),
        g=float(case.g),
        sigma=STEFAN_BOLTZMANN,
        r=float(case.r),
        name="Au_rad_hydro",
    )
    tau = float(case.tau)  # 0 for constant temperature drive
    t_end_sec = float(case.t_end)
    T0_hev = float(case.T0)

    # Dimensionless times 0.05..0.95 map to physical times 0.05*t_end .. 0.95*t_end
    times_dim = np.linspace(0.05, 0.95, n_times)
    profiles = compute_profiles_for_report(
        mat, tau,
        times=times_dim,
        T0=T0_hev,
        iternum=iternum,
        xsi0=xsi0,
    )
    return supersonic_output_to_radiation_data(
        profiles,
        t_end_sec,
        T0_hev,
        label="Supersonic solver (reference)",
        color="green",
        linestyle=":",
    )


def run_radiation_only_comparison(
    skip_rad_hydro: bool = False,
    skip_diffusion: bool = False,
    skip_supersonic: bool = False,
    show_plot: bool = True,
    save_png: bool = True,
) -> None:
    """Run rad_hydro (radiation_only_constant_temperature_drive), 1D Diffusion, and Supersonic solver; compare T, E_rad."""
    from project_3.rad_hydro_sim.problems.presets_utils import get_preset
    from project_3.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project_3.rad_hydro_sim.verification.run_diffusion_1d import run_diffusion_1d

    case, config = get_preset(RADIATION_ONLY_PRESET)
    case_title = case.title or "radiation_only_constant_temperature_drive"
    png_path, gif_path = make_verification_output_paths(f"radiation_only_{case_title}")

    sim_data = None
    if not skip_rad_hydro:
        print("Running rad_hydro (radiation_only_constant_temperature_drive)...")
        x_cells, state, meta, history = simulate_rad_hydro(
            rad_hydro_case=case,
            simulation_config=config,
        )
        sim_data = rad_hydro_history_to_radiation_data(history)
        print(f"  Stored {len(sim_data.times)} time steps.")

    ref_data = None
    if not skip_diffusion:
        print("Running 1D Diffusion self-similar reference...")
        times_sec, z, T_list, E_rad_list = run_diffusion_1d(
            x_max=float(case.x_max),
            t_end=float(case.t_end),
            T_bath_hev=float(case.T0) if case.T0 is not None else 0.86,
            rho0=float(case.rho0) if case.rho0 is not None else 1.0,
            n_times=min(40, max(10, len(sim_data.times) if sim_data else 20)),
            Nz=config.N,
        )
        ref_data = diffusion_output_to_radiation_data(times_sec, z, T_list, E_rad_list)
        print(f"  Stored {len(ref_data.times)} time steps.")

    super_data = None
    if not skip_supersonic:
        print("Running Supersonic solver (radiation self-similar) reference...")
        super_data = run_supersonic_solver_reference(
            case,
            n_times=min(30, max(10, len(sim_data.times) if sim_data else 20)),
            iternum=30,
            xsi0=1.0,
        )
        if super_data is not None:
            print(f"  Stored {len(super_data.times)} time steps.")

    if sim_data is None or ref_data is None:
        print("Need both rad_hydro and diffusion data for comparison.")
        return

    extra_ref_data = [d for d in [super_data] if d is not None]
    title = "Radiation-only: Rad-Hydro vs 1D Diffusion" + (" + Supersonic solver" if extra_ref_data else "")

    print("\nPlotting radiation comparison (T, E_rad vs x)...")
    if show_plot:
        plot_radiation_comparison_slider(
            sim_data, ref_data,
            show=True,
            title=title,
            extra_ref_data=extra_ref_data if extra_ref_data else None,
        )
    if save_png:
        time_mid = 0.5 * (case.t_end * 0.1 + case.t_end * 0.9)
        plot_radiation_comparison_single_time(
            sim_data, ref_data,
            time=time_mid,
            savepath=str(png_path),
            show=False,
            title=title,
            extra_ref_data=extra_ref_data if extra_ref_data else None,
        )
        print(f"Saved PNG: {png_path}")
    print("Radiation-only comparison done.")


# =============================================================================
# Hydro-only: run rad_hydro + hydro_sim (matching case), then compare rho, p, u, e
# =============================================================================

def run_hydro_only_comparison(
    skip_rad_hydro: bool = False,
    skip_hydro_sim: bool = False,
    show_plot: bool = True,
    save_png: bool = True,
) -> None:
    """Run rad_hydro and hydro_sim with matching power-law pressure drive; compare rho, p, u, e."""
    from project_3.rad_hydro_sim.problems.presets_utils import get_preset
    from project_3.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project_3.hydro_sim.problems.driven_shock_problem import DrivenShockCase
    from project_3.hydro_sim.core.geometry import planar
    from project_3.hydro_sim.simulations.lagrangian_sim import (
        simulate_lagrangian,
        SimulationType,
    )
    from project_3.rad_hydro_sim.verification.hydro_data import (
        load_rad_hydro_history,
        load_hydro_history,
    )
    from project_3.hydro_sim.verification.compare_shock_plots import (
        plot_comparison_single_time,
        plot_comparison_slider,
    )

    # Rad-hydro preset (hydro_only_power_law_pressure_drive)
    case_rh, config = get_preset(HYDRO_ONLY_PRESET)
    case_title = case_rh.title or "hydro_only_power_law_pressure_drive"
    png_path, gif_path = make_verification_output_paths(f"hydro_only_{case_title}")

    # Matching driven shock case for hydro_sim (same physics: P0=1, tau=1, gamma=1.25, etc.)
    driven_case = DrivenShockCase(
        gamma=float(case_rh.r + 1),
        x_min=float(case_rh.x_min),
        x_max=float(case_rh.x_max),
        t_end=float(case_rh.t_end),
        rho0=float(case_rh.rho0) if case_rh.rho0 is not None else 1.0,
        p0=float(case_rh.p0) if case_rh.p0 is not None else 1e-6,
        u0=float(case_rh.u0) if case_rh.u0 is not None else 0.0,
        P0=1.0,
        tau=1.0,
        geom=planar(),
        title="Power-law pressure drive (τ=1)",
    )

    sim_data = None
    if not skip_rad_hydro:
        print("Running rad_hydro (hydro_only_power_law_pressure_drive)...")
        x_cells, state, meta, history_rh = simulate_rad_hydro(
            rad_hydro_case=case_rh,
            simulation_config=config,
        )
        sim_data = load_rad_hydro_history(history_rh, label="Rad-Hydro (hydro only)")
        print(f"  Stored {len(sim_data.times)} time steps.")

    ref_data = None
    if not skip_hydro_sim:
        print("Running hydro_sim (matching driven shock)...")
        # Use coarser history for faster verification (e.g. ~100 frames)
        store_every = max(getattr(config, "store_every", 10), config.N // 10)
        x_cells, state, meta, history_h = simulate_lagrangian(
            driven_case,
            sim_type=SimulationType.DRIVEN_SHOCK,
            Ncells=config.N,
            gamma=driven_case.gamma,
            CFL=config.CFL,
            sigma_visc=config.sigma_visc,
            store_every=max(1, store_every),
            geom=driven_case.geom,
        )
        ref_data = load_hydro_history(history_h)
        ref_data.label = "Hydro (run_hydro)"
        ref_data.color = "red"
        ref_data.linestyle = "--"
        print(f"  Stored {len(ref_data.times)} time steps.")

    if sim_data is None or ref_data is None:
        print("Need both rad_hydro and hydro_sim data for comparison.")
        return

    print("\nPlotting hydro comparison (rho, P, u, e vs x)...")
    if show_plot:
        plot_comparison_slider(
            sim_data, ref_data,
            xaxis="x",
            show=True,
            title="Hydro-only: Rad-Hydro vs run_hydro",
        )
    if save_png:
        time_mid = 0.5 * case_rh.t_end
        plot_comparison_single_time(
            sim_data, ref_data,
            time=time_mid,
            xaxis="x",
            savepath=str(png_path),
            show=False,
            title="Hydro-only: Rad-Hydro vs run_hydro",
        )
        print(f"Saved PNG: {png_path}")
    print("Hydro-only comparison done.")


# =============================================================================
# Full rad_hydro: constant T drive vs Shussman (subsonic + shock)
# =============================================================================

def run_full_rad_hydro_comparison(
    skip_rad_hydro: bool = False,
    skip_shussman: bool = False,
    show_plot: bool = True,
    save_png: bool = True,
) -> None:
    """
    Run rad_hydro with constant temperature drive and compare to piecewise Shussman
    reference: subsonic solver (profiles until shock front) + shock solver (driven by
    pressure at front from subsonic). Shock front is diagnosed from rad_hydro solution.
    """
    from project_3.rad_hydro_sim.problems.presets_utils import get_preset
    from project_3.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
    from project_3.rad_hydro_sim.verification.hydro_data import (
        load_rad_hydro_history,
    )
    from project_3.rad_hydro_sim.verification.shussman_comparison import (
        run_shussman_piecewise_reference,
    )
    from project_3.hydro_sim.verification.compare_shock_plots import (
        plot_comparison_single_time,
        plot_comparison_slider,
    )

    case, config = get_preset(FULL_RAD_HYDRO_PRESET)
    case_title = case.title or "rad_hydro_constant_temperature_drive"
    png_path, gif_path = make_verification_output_paths(f"full_rad_hydro_{case_title}")

    sim_data = None
    if not skip_rad_hydro:
        print("Running rad_hydro (rad_hydro_constant_temperature_drive)...")
        x_cells, state, meta, history_rh = simulate_rad_hydro(
            rad_hydro_case=case,
            simulation_config=config,
        )
        sim_data = load_rad_hydro_history(history_rh, label="Rad-Hydro (full)")
        print(f"  Stored {len(sim_data.times)} time steps.")
    if sim_data is None:
        print("Need rad_hydro data for full rad_hydro comparison.")
        return

    ref_data = None
    if not skip_shussman:
        print("Building Shussman piecewise reference (subsonic + shock)...")
        times_sec = np.asarray(sim_data.times, dtype=float)
        ref_data = run_shussman_piecewise_reference(
            case,
            times_sec,
            sim_data.x,
            sim_data.rho,
            subsonic_iternum=1500,
        )
        if ref_data is not None:
            print(f"  Reference has {len(ref_data.times)} time steps.")
    if ref_data is None:
        print("Could not build Shussman reference; skipping comparison.")
        return

    print("\nPlotting full rad_hydro vs Shussman (rho, P, u, e vs x)...")
    if show_plot:
        plot_comparison_slider(
            sim_data,
            ref_data,
            xaxis="x",
            show=True,
            title="Full rad_hydro vs Shussman (subsonic + shock)",
        )
    if save_png:
        time_mid = 0.5 * (float(case.t_end))
        plot_comparison_single_time(
            sim_data,
            ref_data,
            time=time_mid,
            xaxis="x",
            savepath=str(png_path),
            show=False,
            title="Full rad_hydro vs Shussman (subsonic + shock)",
        )
        print(f"Saved PNG: {png_path}")
    print("Full rad_hydro comparison done.")


# =============================================================================
# Main
# =============================================================================

def main():
    # Select mode: RADIATION_ONLY, HYDRO_ONLY, or FULL_RAD_HYDRO
    # MODE = VerificationMode.RADIATION_ONLY
    # MODE = VerificationMode.HYDRO_ONLY
    MODE = VerificationMode.FULL_RAD_HYDRO

    print("=" * 60)
    print("Rad-Hydro Verification Comparison")
    print("=" * 60)
    print(f"Mode: {MODE.value}")
    print()

    if MODE == VerificationMode.RADIATION_ONLY:
        run_radiation_only_comparison(
            skip_rad_hydro=False,
            skip_diffusion=False,
            skip_supersonic=False,
            show_plot=True,
            save_png=True,
        )
    elif MODE == VerificationMode.HYDRO_ONLY:
        run_hydro_only_comparison(
            skip_rad_hydro=False,
            skip_hydro_sim=False,
            show_plot=True,
            save_png=True,
        )
    elif MODE == VerificationMode.FULL_RAD_HYDRO:
        run_full_rad_hydro_comparison(
            skip_rad_hydro=False,
            skip_shussman=False,
            show_plot=True,
            save_png=True,
        )
    else:
        raise ValueError(f"Unknown mode: {MODE}")

    print("\nDone.")


if __name__ == "__main__":
    main()

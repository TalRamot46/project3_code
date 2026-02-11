# run_comparison.py
"""
Unified runner for shock comparison simulations.

Compares hydro_sim (Lagrangian simulation) with Shussman shock solver (project_3.shussman_solvers.shock_solver).
All configuration is done through ComparisonCase and ComparisonConfig dataclasses.

Usage:
    # In main(), change the preset name to run different cases:
    case, config = get_preset("gold_tau_0")
    run_comparison(case, config)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project_3 to path for imports
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent.parent.parent  # project_3
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Configuration and presets (local to hydro_sim.verification)
from project_3.hydro_sim.verification.comparison_config import ComparisonCase, ComparisonConfig, PlotMode
from project_3.hydro_sim.verification.presets import get_preset, list_presets, PRESETS

# Plotting
from project_3.hydro_sim.verification.compare_shock_plots import (
    SimulationData,
    load_shussman_data,
    load_hydro_history,
    plot_comparison_single_time,
    plot_comparison_slider,
    plot_comparison_overlay,
    save_comparison_gif,
)


# ============================================================================
# Simulation Runners
# ============================================================================

def run_hydro_simulation(case: ComparisonCase, config: ComparisonConfig) -> SimulationData:
    """
    Run the hydro_sim simulation and return data in comparison format.
    
    Parameters:
        case: Physical parameters (ComparisonCase)
        config: Run configuration (ComparisonConfig)
        
    Returns:
        SimulationData ready for comparison plotting
    """
    from project_3.hydro_sim.simulations.lagrangian_sim import simulate_lagrangian, SimulationType
    
    driven_case = case.to_driven_shock_case()
    
    print("Running hydro simulation...")
    print(f"  Domain: [{case.x_min:.2e}, {case.x_max:.2e}] cm")
    print(f"  t_end: {case.t_end:.2e} s")
    print(f"  N cells: {config.N}")
    print(f"  P0: {case.P0}, τ: {case.tau}")
    
    x_cells, state, meta, history = simulate_lagrangian(
        driven_case,
        sim_type=SimulationType.DRIVEN_SHOCK,
        Ncells=config.N,
        gamma=case.gamma,
        CFL=config.CFL,
        sigma_visc=config.sigma_visc,
        store_every=max(1, config.N // 100),  # ~100 frames
        geom=driven_case.geom
    )
    
    return load_hydro_history(history)


def run_shussman_solver(case: ComparisonCase, save_path: str | None = None) -> SimulationData:
    """
    Run the shussman_shock_solver and return data in comparison format.
    
    Parameters:
        case: Physical parameters (ComparisonCase)
        save_path: Optional path to save NPZ file
        
    Returns:
        SimulationData ready for comparison plotting
    """
    from project_3.shussman_solvers.shock_solver.run_shock_solver import compute_shock_profiles
    
    params = case.get_shussman_params()
    
    print("Running self-similar solver...")
    print(f"  Material: {params['material'].name}")
    print(f"  P0: {params['P0']}")
    print(f"  Times: {len(params['times'])} snapshots")
    
    if save_path is None:
        save_path = str(Path(__file__).parent / "shock_profiles.npz")
    
    compute_shock_profiles(
        mat=params['material'],
        P0=params['P0'],
        Pw=params['Pw'],
        times=params['times'],
        save_npz=save_path,
    )
    
    return load_shussman_data(save_path)


# ============================================================================
# Unified Comparison Runner
# ============================================================================

def run_comparison(case: ComparisonCase, config: ComparisonConfig) -> None:
    """
    Run a full shock comparison simulation.
    
    This is the main entry point for running any comparison.
    
    Parameters:
        case: Physical parameters (ComparisonCase)
        config: Run configuration (ComparisonConfig)
    """
    print("=" * 60)
    print("Shock Comparison: Hydro Simulation vs Self-Similar Solution")
    print("=" * 60)
    print(f"Case: {case.title}")
    print(f"  P0: {case.P0}, τ: {case.tau}")
    print(f"  ρ0: {case.rho0} g/cm³")
    print(f"  t_end: {case.t_end:.2e} s")
    print(f"  Domain: [{case.x_min:.2e}, {case.x_max:.2e}] cm")
    print(f"  N cells: {config.N}")
    print()
    
    # Get output paths
    png_path, gif_path = case.output_paths
    
    # Run hydro simulation
    if not config.skip_sim:
        sim_data = run_hydro_simulation(case, config)
    else:
        print("Skipping hydro simulation (skip_sim=True)")
        sim_data = None
    
    # Run shussman solver
    if not config.skip_solver:
        ref_data = run_shussman_solver(case)
    elif config.npz_path:
        print(f"Loading existing NPZ: {config.npz_path}")
        ref_data = load_shussman_data(config.npz_path)
    else:
        print("Skipping solver (skip_solver=True)")
        ref_data = None
    
    # Check we have data
    if sim_data is None or ref_data is None:
        print("Error: Need both simulation and reference data for comparison.")
        return
    
    print()
    print(f"Simulation data: {len(sim_data.times)} time steps")
    print(f"Reference data: {len(ref_data.times)} time steps")
    print()

    return sim_data, ref_data, case, config, png_path, gif_path
    


def _plot_results(
    sim_data: SimulationData,
    ref_data: SimulationData,
    case: ComparisonCase,
    config: ComparisonConfig,
    png_path: Path,
    gif_path: Path,
) -> None:
    """
    Handle all plotting based on configuration.
    """
    if config.mode == PlotMode.SLIDER:
        plot_comparison_slider(
            sim_data, ref_data,
            xaxis=config.xaxis,
            show=config.show_plot,
            title=case.title,
        )
    
    elif config.mode == PlotMode.SINGLE:
        time = config.time_for_single if config.time_for_single else case.t_end
        savepath = str(png_path) if config.save_png else None
        plot_comparison_single_time(
            sim_data, ref_data,
            time=time,
            xaxis=config.xaxis,
            savepath=savepath,
            show=config.show_plot,
            title=case.title,
        )
    
    elif config.mode == PlotMode.OVERLAY:
        savepath = str(png_path) if config.save_png else None
        plot_comparison_overlay(
            sim_data, ref_data,
            times=list(case.times),
            xaxis=config.xaxis,
            savepath=savepath,
            show=config.show_plot,
            title=case.title,
        )
    
    elif config.mode == PlotMode.GIF:
        save_comparison_gif(
            sim_data, ref_data,
            gif_path=str(gif_path),
            xaxis=config.xaxis,
            fps=10,
            stride=max(1, len(sim_data.times) // 50),
            title=case.title,
        )
        print(f"Saved GIF to: {gif_path}")
    
    # Additional saves if requested
    if config.save_png and config.mode != PlotMode.SINGLE:
        # Save a single-time plot as well
        plot_comparison_single_time(
            sim_data, ref_data,
            time=case.t_end,
            xaxis=config.xaxis,
            savepath=str(png_path),
            show=False,
            title=case.title,
        )
        print(f"Saved PNG to: {png_path}")
    
    if config.save_gif and config.mode != PlotMode.GIF:
        save_comparison_gif(
            sim_data, ref_data,
            gif_path=str(gif_path),
            xaxis=config.xaxis,
            fps=10,
            stride=max(1, len(sim_data.times) // 50),
            title=case.title,
        )
        print(f"Saved GIF to: {gif_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run a comparison with a predefined preset."""
    
    # ===== SELECT YOUR PRESET HERE =====
    PRESET = "gold_tau_neg"
    # ===================================
    
    # List available presets for reference
    # list_presets()
    
    # Get case and config
    case, config = get_preset(PRESET)
    
    # Run comparison
    sim_data, ref_data, case, config, png_path, gif_path = run_comparison(case, config)

    # Plot based on mode
    _plot_results(sim_data, ref_data, case, config, png_path, gif_path)
    
    print("Done!")

if __name__ == "__main__":
    main()

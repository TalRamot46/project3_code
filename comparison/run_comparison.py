# run_comparison.py
"""
Run comparison between hydro_sim (Lagrangian simulation) and 
shussman_shock_solver (self-similar solution).

Usage:
    python run_comparison.py --mode slider
    python run_comparison.py --mode overlay --save figures/comparison.png
    python run_comparison.py --mode gif --output figures/comparison.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

# Add parent directories to path for imports
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
_hydro_sim = _project_root / "hydro_sim"
_shussman = _project_root / "shussman_shock_solver"

for p in [str(_project_root), str(_hydro_sim), str(_shussman)]:
    if p not in sys.path:
        sys.path.insert(0, p)

project_root = _project_root

from comparison.shock_config import ShockComparisonConfig, gold_constant_drive
from comparison.compare_shock_plots import (
    SimulationData,
    load_shussman_data,
    load_hydro_history,
    plot_comparison_single_time,
    plot_comparison_slider,
    plot_comparison_overlay,
    save_comparison_gif,
)


def run_hydro_simulation(config: ShockComparisonConfig):
    """Run the hydro_sim simulation with the given config."""
    from simulations.driven_shock_sim import simulate_driven_shock
    
    case = config.to_driven_shock_case()
    
    print(f"Running hydro simulation...")
    print(f"  Domain: [{config.x_min}, {config.x_max}] cm")
    print(f"  t_end: {config.t_end:.2e} s")
    print(f"  N cells: {config.Ncells}")
    print(f"  P0: {config.P0}, tau: {config.tau}")
    
    x_cells, state, meta, history = simulate_driven_shock(
        case,
        Ncells=config.Ncells,
        CFL=config.CFL,
        sigma_visc=config.sigma_visc,
        store_every=max(1, config.Ncells // 100),  # ~100 frames
    )
    
    return history


def run_shussman_solver(config: ShockComparisonConfig, save_path: str | None = None):
    """Run the shussman_shock_solver and return the NPZ path."""
    from run_shock_solver import compute_shock_profiles
    
    params = config.get_shussman_params()
    
    print(f"Running self-similar solver...")
    print(f"  Material: {params['material'].name}")
    print(f"  P0: {params['P0']}")
    print(f"  Times: {len(params['times'])} snapshots")
    
    if save_path is None:
        save_path = str(project_root / "comparison" / "shock_profiles.npz")
    
    data = compute_shock_profiles(
        mat=params['material'],
        P0=params['P0'],
        Pw=params['Pw'],
        times=params['times'],
        save_npz=save_path,
    )
    
    return save_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare hydro_sim with shussman_shock_solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode", type=str, default="slider",
        choices=["slider", "single", "overlay", "gif"],
        help="Plotting mode"
    )
    parser.add_argument(
        "--xaxis", type=str, default="m",
        choices=["m", "x"],
        help="X-axis variable: m (mass) or x (position)"
    )
    parser.add_argument(
        "--time", type=float, default=None,
        help="Time for single-time plot (default: last time)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure path"
    )
    parser.add_argument(
        "--output", type=str, default="comparison.gif",
        help="Output GIF path (for gif mode)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not show interactive plots"
    )
    parser.add_argument(
        "--skip-sim", action="store_true",
        help="Skip running simulation, load existing data"
    )
    parser.add_argument(
        "--skip-solver", action="store_true",
        help="Skip running solver, load existing NPZ"
    )
    parser.add_argument(
        "--npz", type=str, default=None,
        help="Path to existing shussman NPZ file"
    )
    
    # Config parameters
    parser.add_argument("--P0", type=float, default=10.0, help="Pressure amplitude")
    parser.add_argument("--tau", type=float, default=0.0, help="Power-law exponent")
    parser.add_argument("--t_end", type=float, default=100e-9, help="End time (s)")
    parser.add_argument("--N", type=int, default=501, help="Number of cells")
    parser.add_argument("--rho0", type=float, default=19.32, help="Initial density")
    
    return parser


def get_default_args():
    """Return default arguments for direct execution."""
    class Args:
        mode = "single"
        xaxis = "m"
        time = None
        save = None
        output = "comparison.gif"
        no_show = False
        skip_sim = False
        skip_solver = False
        npz = None
        P0 = 10.0
        tau = 1.0
        t_end = 100e-9
        N = 500
        rho0 = 19.32
    return Args()


def main():
    USE_PARSER = False
    
    if USE_PARSER:
        parser = create_parser()
        args = parser.parse_args()
    else:
        args = get_default_args()
    
    # Create configuration
    from shussman_shock_solver.materials_shock import au_supersonic_variant_1
    
    config = ShockComparisonConfig(
        # material and pressure drive
        material=au_supersonic_variant_1(),
        P0=args.P0,
        tau=args.tau,

        # initial conditions
        rho0=args.rho0,

        # time
        t_end=args.t_end,

        # simulation parameters
        Ncells=args.N,

        # plot
        title=f"Shock Comparison (P0={args.P0}, τ={args.tau})",
    )
    
    print("=" * 60)
    print("Shock Comparison: Hydro Simulation vs Self-Similar Solution")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  P0: {config.P0}, tau: {config.tau}")
    print(f"  rho0: {config.rho0} g/cm³")
    print(f"  t_end: {config.t_end:.2e} s")
    print(f"  Domain: [{config.x_min:.2e}, {config.x_max:.2e}] cm")
    print(f"  N cells: {config.Ncells}")
    print()
    
    # Run hydro simulation
    if not args.skip_sim:
        history = run_hydro_simulation(config)
        sim_data = load_hydro_history(history)
    else:
        print("Skipping hydro simulation (--skip-sim)")
        sim_data = None
    
    # Run shussman solver
    if not args.skip_solver:
        npz_path = run_shussman_solver(config)
        ref_data = load_shussman_data(npz_path)
    elif args.npz:
        print(f"Loading existing NPZ: {args.npz}")
        ref_data = load_shussman_data(args.npz)
    else:
        print("Skipping solver (--skip-solver)")
        ref_data = None
    
    # Check we have data
    if sim_data is None or ref_data is None:
        print("Error: Need both simulation and reference data for comparison.")
        return
    
    print()
    print(f"Simulation data: {len(sim_data.times)} time steps")
    print(f"Reference data: {len(ref_data.times)} time steps")
    print()
    
    # Plot based on mode
    if args.mode == "slider":
        plot_comparison_slider(
            sim_data, ref_data,
            xaxis=args.xaxis,
            show=not args.no_show,
            title=config.title,
        )
    
    elif args.mode == "single":
        time = args.time if args.time else config.t_end
        plot_comparison_single_time(
            sim_data, ref_data,
            time=time,
            xaxis=args.xaxis,
            savepath=args.save,
            show=not args.no_show,
            title=config.title,
        )
    
    elif args.mode == "overlay":
        plot_comparison_overlay(
            sim_data, ref_data,
            times=list(config.times),
            xaxis=args.xaxis,
            savepath=args.save,
            show=not args.no_show,
            title=config.title,
        )
    
    elif args.mode == "gif":
        gif_path = str(project_root / "comparison" / "figures" / args.output)
        save_comparison_gif(
            sim_data, ref_data,
            gif_path=gif_path,
            xaxis=args.xaxis,
            fps=10,
            stride=max(1, len(sim_data.times) // 50),
            title=config.title,
        )
    
    print("Done!")


if __name__ == "__main__":
    main()

# run_hydro.py
"""
Unified runner for hydrodynamic simulations.
Supports Riemann, Driven Shock, and Sedov explosion problems.

Usage:
    python run_hydro.py riemann --test 1 --N 1000
    python run_hydro.py shock --N 1000 --P0 1.0 --tau 1.5
    python run_hydro.py sedov --N 500 --E0 1.0
"""
import numpy as np
import sys
from pathlib import Path
import argparse
from enum import Enum

# Problem definitions
from problems.riemann_problem import RiemannCase, RIEMANN_TEST_CASES
from problems.driven_shock_problem import DrivenShockCase
from problems.sedov_problem import SedovExplosionCase, SEDOV_TEST_CASES

# Simulations
from simulations.riemann_sim import simulate_riemann
from simulations.driven_shock_sim import simulate_driven_shock

# Unified plotting
from plotting.hydro_plots import (
    plot_riemann_results,
    plot_shock_results,
    plot_sedov_results,
    plot_history_slider,
    save_history_gif,
)


class ProblemType(str, Enum):
    RIEMANN = "riemann"
    SHOCK = "shock"
    SEDOV = "sedov"


# ============================================================================
# Argument Parsing
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subparsers for each problem type."""
    parser = argparse.ArgumentParser(
        description="Unified hydrodynamic simulation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_hydro.py riemann --test 1 --N 1000
    python run_hydro.py shock --N 1000 --P0 1.0 --tau 1.5 --t_end 0.5
    python run_hydro.py sedov --case standard_spherical --N 500
        """
    )
    
    subparsers = parser.add_subparsers(dest="problem", required=True, help="Problem type")
    
    # Common arguments helper
    def add_common_args(p):
        p.add_argument("--N", type=int, default=1000, help="Number of cells")
        p.add_argument("--CFL", type=float, default=0.5, help="CFL number")
        p.add_argument("--sigma", type=float, default=1.0, help="Artificial viscosity coefficient")
        p.add_argument("--save", type=str, default=None, help="Save figure path (png)")
        p.add_argument("--no-show", action="store_true", help="Do not show plots")
        p.add_argument("--gif", type=str, default=None, help="Save animation GIF path")
        p.add_argument("--slider", action="store_true", help="Show interactive slider plot")
    
    # ----- Riemann subparser -----
    p_riemann = subparsers.add_parser("riemann", help="Riemann shock tube problem")
    p_riemann.add_argument("--test", type=int, default=1, choices=[1, 2, 3, 4],
                           help="Test case ID (1-4)")
    p_riemann.add_argument("--gamma", type=float, default=1.4, help="Adiabatic index")
    add_common_args(p_riemann)
    
    # ----- Driven Shock subparser -----
    p_shock = subparsers.add_parser("shock", help="Driven shock problem")
    p_shock.add_argument("--gamma", type=float, default=1.4, help="Adiabatic index")
    p_shock.add_argument("--t_end", type=float, default=1.0, help="Final simulation time")
    p_shock.add_argument("--rho0", type=float, default=1.0, help="Initial density")
    p_shock.add_argument("--p0", type=float, default=1e-6, help="Initial pressure")
    p_shock.add_argument("--u0", type=float, default=0.0, help="Initial velocity")
    p_shock.add_argument("--x_min", type=float, default=0.0, help="Left boundary")
    p_shock.add_argument("--x_max", type=float, default=1.0, help="Right boundary")
    p_shock.add_argument("--P0", type=float, default=1.0, help="Pressure amplitude for power-law drive")
    p_shock.add_argument("--tau", type=float, default=0.0, help="Power-law exponent: p ~ t^tau")
    add_common_args(p_shock)
    
    # ----- Sedov subparser -----
    p_sedov = subparsers.add_parser("sedov", help="Sedov-Taylor point explosion")
    p_sedov.add_argument("--case", type=str, default="standard_spherical",
                         choices=list(SEDOV_TEST_CASES.keys()),
                         help="Predefined Sedov case name")
    p_sedov.add_argument("--gamma", type=float, default=1.4, help="Adiabatic index")
    p_sedov.add_argument("--E0", type=float, default=None, help="Override explosion energy")
    p_sedov.add_argument("--rho0", type=float, default=None, help="Override ambient density")
    p_sedov.add_argument("--t_end", type=float, default=None, help="Override end time")
    add_common_args(p_sedov)
    
    return parser


def get_default_args(problem: ProblemType):
    """Return default arguments when not using the argument parser."""
    class Args:
        pass
    
    args = Args()
    args.N = 1000
    args.CFL = 1/3
    args.sigma = 1.0
    args.save = None
    args.no_show = False
    args.gif = 'project_3/shussman_shock_solver/figures/driven_shock/shock.gif'
    args.slider = True
    
    if problem == ProblemType.RIEMANN:
        args.problem = "riemann"
        args.test = 1
        args.gamma = 1.4
        
    elif problem == ProblemType.SHOCK:
        args.problem = "shock"
        args.gamma = 1.25
        args.t_end = 4e-9
        args.rho0 = 19.32
        args.p0 = 1e-3
        args.u0 = 0.0
        args.x_min = 0.0
        args.x_max = 1 / 19.32 * 1e-7
        args.P0 = 30.0
        args.tau = 0
        args.N = 1001
        args.CFL = 0.2
        
    elif problem == ProblemType.SEDOV:
        args.problem = "sedov"
        args.case = "standard_spherical"
        args.gamma = 1.4
        args.E0 = None
        args.rho0 = None
        args.t_end = None
        
    return args


# ============================================================================
# Problem Setup
# ============================================================================

def setup_riemann(args) -> tuple:
    """Set up Riemann problem case."""
    case = RIEMANN_TEST_CASES[args.test]
    sigma = args.sigma if args.sigma is not None else case.sigma_visc
    return case, sigma


def setup_shock(args) -> DrivenShockCase:
    """Set up Driven Shock problem case."""
    case = DrivenShockCase(
        title="Power-law driven shock",
        x_min=args.x_min,
        x_max=args.x_max,
        t_end=args.t_end,
        rho0=args.rho0,
        p0=args.p0,
        u0=args.u0,
        gamma=args.gamma,
        tau=args.tau,
        P0=args.P0,
    )
    return case


def setup_sedov(args) -> SedovExplosionCase:
    """Set up Sedov explosion problem case."""
    base_case = SEDOV_TEST_CASES[args.case]
    
    # Allow overrides for specific parameters
    from dataclasses import replace
    overrides = {}
    if args.E0 is not None:
        overrides["E0"] = args.E0
    if args.rho0 is not None:
        overrides["rho0"] = args.rho0
    if args.t_end is not None:
        overrides["t_end"] = args.t_end
        
    if overrides:
        case = replace(base_case, **overrides)
    else:
        case = base_case
        
    return case


# ============================================================================
# Main Runners
# ============================================================================

def run_riemann(args):
    """Run Riemann problem simulation and plotting."""
    case, sigma = setup_riemann(args)
    
    # Run simulation
    x_cells, num, ex, meta = simulate_riemann(
        args.test, 
        Ncells=args.N, 
        gamma=args.gamma, 
        CFL=args.CFL, 
        sigma_visc=sigma
    )
    
    # Plot results
    plot_riemann_results(
        x_cells=x_cells,
        numerical=num,
        exact=ex,
        meta=meta,
        savepath=args.save,
        show=not args.no_show,
    )
    
    return x_cells, num, ex, meta


def run_shock(args):
    """Run Driven Shock problem simulation and plotting."""
    case = setup_shock(args)
    
    # Run simulation
    x_cells, state, meta, hist = simulate_driven_shock(
        case,
        Ncells=args.N,
        CFL=args.CFL,
        sigma_visc=args.sigma,
    )
    
    # Plot final profiles
    plot_shock_results(
        x_cells=x_cells,
        state=state,
        case=case,
        savepath=args.save,
        show=not args.no_show,
    )
    
    # Interactive slider
    if args.slider:
        plot_history_slider(hist, case, show=True)
    
    # Save GIF
    if args.gif:
        save_history_gif(hist, case, gif_path=args.gif, fps=20, stride=2)
    
    return x_cells, state, meta, hist


def run_sedov(args):
    """Run Sedov explosion problem simulation and plotting."""
    case = setup_sedov(args)
    
    # Note: Sedov simulation not yet implemented - placeholder
    # Once implemented, this would call simulate_sedov similar to others
    print(f"Sedov case configured: {case.title}")
    print(f"  E0={case.E0}, rho0={case.rho0}, t_end={case.t_end}")
    print("  [Simulation not yet implemented - placeholder]")
    
    # For now, just initialize and show initial conditions
    from core.geometry import spherical
    from problems.sedov_problem import init_sedov_explosion
    
    x_nodes = np.linspace(case.x_min + 0.01, case.x_max, args.N + 1)
    geom = spherical()
    state, m = init_sedov_explosion(x_nodes, geom, args.gamma, case)
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    
    # Plot initial condition
    plot_sedov_results(
        x_cells=x_cells,
        state=state,
        case=case,
        savepath=args.save,
        show=not args.no_show,
    )
    
    return x_cells, state, case


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    USE_PARSER = False
    
    if USE_PARSER:
        parser = create_parser()
        args = parser.parse_args()
    else:
        # Default to Riemann for testing
        args = get_default_args(ProblemType.SHOCK)
    
    # Dispatch to appropriate runner
    if args.problem == "riemann":
        run_riemann(args)
    elif args.problem == "shock":
        run_shock(args)
    elif args.problem == "sedov":
        run_sedov(args)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")


if __name__ == "__main__":
    main()

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
from simulations.driven_shock_sim import simulate_driven_shock, simulate_lagrangian, simulate_sedov

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
    CONTINUOUS_SHOCK = "continuous_shock"  # alias for shock with tau=0

# ============================================================================
# Argument Parsing
# ============================================================================


def get_default_args(problem: ProblemType):
    """Return default arguments when not using the argument parser."""
    class Args:
        pass
    
    args = Args()
    args.N = 316
    args.CFL = 1/3
    args.sigma = 1.0
    args.save = None
    args.no_show = False
    args.store_every = 100  
    args.gif = 'project_3/hydro_sim/figures/shock1.gif'
    args.slider = True
    
    if problem == ProblemType.RIEMANN:
        args.problem = "riemann"
        args.test = 1
        args.gamma = 1.4
        
    elif problem == ProblemType.SHOCK:
        args.problem = "shock"
        args.gamma = 1.25
        args.t_end = 100e-9
        args.rho0 = 19.32
        args.p0 = 1e-3
        args.u0 = 0.0
        args.x_min = 0.0
        args.x_max = 3e-6 / args.rho0
        args.P0 = 10.0
        args.tau = 0.0
        args.N = 1001
        args.CFL = 0.2
    
    elif problem == ProblemType.CONTINUOUS_SHOCK:
        args.problem = "shock"
        args.gamma = 1.25
        args.t_end = 5e-3
        args.rho0 = 19.32
        args.p0 = 1e-3
        args.u0 = 0.0
        args.x_min = 0.0
        args.x_max = 3e-3 / args.rho0
        args.P0 = 10.0
        args.tau = 1.0
        args.N = 2001
        args.CFL = 0.2

    elif problem == ProblemType.SEDOV:
        args.problem = "sedov"
        args.case = "standard_planar"
        args.gamma = 1.4
        
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
    case = SEDOV_TEST_CASES[args.case]
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
    
    # Determine geometry from case name
    from core.geometry import spherical, cylindrical, planar
    
    case_name = args.case.lower()
    if "cylindrical" in case_name:
        geom = cylindrical()
    elif "planar" in case_name:
        geom = planar()
    else:
        geom = spherical()
    
    print(f"Running Sedov simulation: {case.title}")
    print(f"  E0={case.E0}, rho0={case.rho0}, t_end={case.t_end}")
    print(f"  Geometry: alpha={geom.alpha}, Ncells={args.N}")
    
    # Run simulation
    x_cells, state, meta, hist = simulate_lagrangian(
        case,
        "sedov",
        Ncells=args.N,
        gamma=args.gamma,
        CFL=args.CFL,
        sigma_visc=args.sigma,
        store_every=args.store_every,
        geom=geom,
    )
    # Plot final profiles
    plot_sedov_results(
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


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    args = get_default_args(ProblemType.SEDOV)
    
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

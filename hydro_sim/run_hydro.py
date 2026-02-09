# run_hydro.py
"""
Unified runner for hydrodynamic simulations.

Supports Riemann, Driven Shock, and Sedov explosion problems.
All configuration is done through ProblemCase and SimulationConfig dataclasses.

Usage:
    # In main(), change the preset name to run different cases:
    case, config = get_preset("sedov_spherical")
    run_simulation(case, config)
"""
import sys
import numpy as np
from pathlib import Path

# Ensure the current directory is in sys.path for imports to work
_hydro_sim_dir = Path(__file__).parent.absolute()
if str(_hydro_sim_dir) not in sys.path:
    sys.path.insert(0, str(_hydro_sim_dir))

# Problem definitions and configuration
from problems.simulation_config import SimulationConfig
from problems.presets import get_preset, get_preset_names, list_presets, PRESETS
from problems.riemann_problem import RiemannCase
from problems.driven_shock_problem import DrivenShockCase
from problems.sedov_problem import SedovExplosionCase
from problems.Hydro_case import HydroCase

# Simulations
from simulations.lagrangian_sim import simulate_lagrangian, SimulationType

# Unified plotting
from plotting.hydro_plots import (
    plot_riemann_results,
    plot_shock_results,
    plot_sedov_results,
    plot_history_slider,
    save_history_gif,
)

# For Riemann exact solution
from simulations.riemann_exact import sample_solution


# ============================================================================
# Simulation Type Detection
# ============================================================================

def _get_sim_type(case: HydroCase) -> SimulationType:
    """Determine simulation type from case class."""
    if isinstance(case, RiemannCase):
        return SimulationType.RIEMANN
    elif isinstance(case, DrivenShockCase):
        return SimulationType.DRIVEN_SHOCK
    elif isinstance(case, SedovExplosionCase):
        return SimulationType.SEDOV
    else:
        raise TypeError(f"Unknown case type: {type(case)}")


# ============================================================================
# Unified Simulation Runner
# ============================================================================

def run_simulation(
    case: HydroCase,
    config: SimulationConfig,
) -> tuple:
    """
    Run a hydrodynamic simulation.
    
    This is the main entry point for running any simulation type.
    The simulation type is automatically detected from the case class.
    """
    sim_type = _get_sim_type(case)
    
    print(f"Running {sim_type.value} simulation: {case.title}")
    print(f"  γ={case.gamma}, x∈[{case.x_min}, {case.x_max}], t_end={case.t_end}")
    print(f"  N={config.N}, CFL={config.CFL}, σ={config.sigma_visc}")
    
    # Run simulation
    x_cells, state, meta, history = simulate_lagrangian(
        case,
        sim_type,
        Ncells=config.N,
        gamma=case.gamma,
        CFL=config.CFL,
        sigma_visc=config.sigma_visc,
        store_every=config.store_every,
        geom=case.geom,
    )
    
    # Add config to meta
    meta["config"] = config
    
    return x_cells, state, meta, history

# ============================================================================
# Plotting Dispatch
# ============================================================================

def plot_results(
    x_cells: np.ndarray,
    state,
    case: HydroCase,
    config: SimulationConfig,
    history=None,
) -> None:
    """
    Plot simulation results based on problem type.
    
    Handles:
    - Final state plots
    - Interactive slider (if config.show_slider)
    - GIF animation (if config.gif_path)
    """
    sim_type = _get_sim_type(case)
    
    # Plot final state
    if sim_type == SimulationType.RIEMANN:
        # Compute exact solution for comparison
        rho_ex, u_ex, p_ex, e_ex = sample_solution(
            x_cells, case.t_end,
            case.left, case.right, case.gamma
        )
        u_num = 0.5 * (state.u[:-1] + state.u[1:])
        
        plot_riemann_results(
            x_cells=x_cells,
            numerical=dict(rho=state.rho, p=state.p, u=u_num, e=state.e),
            exact=dict(rho=rho_ex, p=p_ex, u=u_ex, e=e_ex),
            meta=dict(
                test_id=case.title or "Riemann",
                title_extra=case.title, 
                t_end=case.t_end,
                x_min=case.x_min,
                x_max=case.x_max,
                Ncells=config.N,
                gamma=case.gamma,
            ),
            savepath=config.save_path,
            show=config.show_plot,
        )
        
    elif sim_type == SimulationType.DRIVEN_SHOCK:
        plot_shock_results(
            x_cells=x_cells,
            state=state,
            case=case,
            savepath=config.save_path,
            show=config.show_plot,
        )
        
    elif sim_type == SimulationType.SEDOV:
        plot_sedov_results(
            x_cells=x_cells,
            state=state,
            case=case,
            savepath=config.save_path,
            show=config.show_plot,
        )
    
    # Interactive slider
    if config.show_slider and history is not None:
        plot_history_slider(history, case, show=True)
    
    # Save GIF
    if config.gif_path and history is not None:
        save_history_gif(history, case, gif_path=config.gif_path, fps=20, stride=2)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run a simulation with a predefined preset."""
    
    # ===== SELECT YOUR PRESET HERE =====
    # run over all presets for testing:
    for preset_name in get_preset_names():
        print(f"\n=== Running preset: {preset_name} ===")
        
        # Get case and config
        case, config = get_preset(preset_name)
        
        # Auto-generate output paths for PNG and GIF based on case title
        config = config.with_output_paths(case.title)
        
        # Run simulation
        x_cells, state, meta, history = run_simulation(case, config)
        
        # Plot results (will automatically save PNG and GIF)
        plot_results(x_cells, state, case, config, history)
        

if __name__ == "__main__":
    main()

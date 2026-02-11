import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Add parent directory to path so project_3 module can be found
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_3.rad_hydro_sim.problems.presets_utils import (
    get_preset_names,
    get_preset,
)
from project_3.hydro_sim.plotting.hydro_plots import save_history_gif
from project_3.rad_hydro_sim.simulation.iterator import simulate_rad_hydro
from project_3.rad_hydro_sim.plotting.slider import plot_history_slider
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.hydro_sim.problems.simulation_config import SIMULATION_CONFIGS, SimulationConfig

def run_simulation(
    case: RadHydroCase,
    config: SimulationConfig,
) -> tuple:
    """
    Run a hydrodynamic simulation.
    
    This is the main entry point for running any simulation type.
    The simulation type is automatically detected from the case class.
    """    
    print(f"Running simulation: {case.title}")
    print(f"  r={case.r}, x∈[{case.x_min}, {case.x_max}], t_end={case.t_end}")
    
    # Run simulation
    x_cells, state, meta, history = simulate_rad_hydro(
        rad_hydro_case=case,
        simulation_config=config,
    )
    
    # Add config to meta
    meta["config"] = config
    
    return x_cells, state, meta, history

def plot_results(
    x_cells: np.ndarray,
    state,
    case: RadHydroCase,
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
    # Auto-generate output paths for PNG and GIF based on case title
    config = config.with_output_paths(case.title)
    
    # Plot final state
    # if config.show_plot:
    #     plot_final_state(x_cells, state, case, config)
        
    # Interactive slider
    if history is not None and config.show_slider:
        plot_history_slider(history, case, show=True)
    
    # Save GIF
    if config.gif_path and history is not None:
        save_history_gif(history, case, gif_path=config.gif_path, fps=20, stride=2)



def main():
    """Run a simulation with a predefined preset."""
    
    # ===== SELECT YOUR PRESET HERE =====
    preset_name = "rad_hydro_constant_temperature_drive"  
    
    # Get case and config from preset
    case, config = get_preset(preset_name)

    print(f"\n=== Running preset: {preset_name} ===")
                
    # Run simulation
    x_cells, state, meta, history = run_simulation(case, config)
    
    # Plot results (will automatically save PNG and GIF)
    plot_results(x_cells, state, case, config, history)
        

if __name__ == "__main__":
    main()

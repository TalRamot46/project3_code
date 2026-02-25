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
from project_3.rad_hydro_sim.output_paths import make_rad_hydro_output_paths
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
    preset_name: str | None = None,
) -> None:
    """
    Plot simulation results based on problem type.
    
    Handles:
    - Final state plots
    - Interactive slider (if config.show_slider)
    - GIF animation (if config.gif_path)
    Outputs go to results/rad_hydro_sim/figures/png and .../gif.
    """
    # Use rad_hydro_sim figures dir; fallback to preset name if case.title is empty
    case_name = case.title or preset_name or "rad_hydro_run"
    png_path, gif_path = make_rad_hydro_output_paths(case_name)
    config = SimulationConfig(
        N=config.N,
        CFL=config.CFL,
        sigma_visc=config.sigma_visc,
        store_every=config.store_every,
        save_path=str(png_path),
        gif_path=str(gif_path),
        show_plot=config.show_plot,
        show_slider=config.show_slider,
    )
        
    # Interactive slider (optionally save static PNG from current frame)
    if history is not None and config.show_slider:
        plot_history_slider(
            history, case,
            savepath=config.save_path,
            show=True,
        )
    
    # Save GIF
    if config.gif_path and history is not None:
        save_history_gif(history, case, gif_path=config.gif_path, fps=20, stride=2)


def main():
    """Run a simulation with a predefined preset.
    Outputs: results/rad_hydro_sim/figures/png/<name>.png, .../gif/<name>.gif
    """
    # ===== SELECT YOUR PRESET HERE =====
    # Preset = physical case name (SIMPLE_TEST_CASES key). Use constants from presets_config.
    # Run list_presets() from presets_utils for a grouped list.
    from project_3.rad_hydro_sim.problems.presets_config import (
        PRESET_CONSTANT_PRESSURE,
        PRESET_FIG_10,
        PRESET_CONSTANT_T_RADIATION
    )
    preset_name = PRESET_CONSTANT_T_RADIATION  
    
    # Get case and config from preset
    case, config = get_preset(preset_name)

    print(f"\n=== Running preset: {preset_name} ===")
                
    # Run simulation
    x_cells, state, meta, history = run_simulation(case, config)
    
    # Plot results (PNG/GIF under results/rad_hydro_sim/figures/)
    plot_results(x_cells, state, case, config, history, preset_name=preset_name)
        

if __name__ == "__main__":
    main()

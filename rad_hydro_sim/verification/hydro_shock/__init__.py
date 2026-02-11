# comparison/__init__.py
"""
Comparison module for driven shock simulations.
Compares results from hydro_sim (Lagrangian simulation) with
shussman_shock_solver (self-similar solution).
"""

# Configuration classes
from .comparison_config import (
    ComparisonCase,
    ComparisonConfig,
    PlotMode,
    make_output_paths,
)

# Presets
from .presets import (
    get_preset,
    list_presets,
    get_preset_names,
    PRESETS,
    COMPARISON_CASES,
    COMPARISON_CONFIGS,
)

# Plotting utilities
from .compare_shock_plots import (
    SimulationData,
    load_shussman_data,
    load_hydro_history,
    plot_comparison_single_time,
    plot_comparison_slider,
    plot_comparison_overlay,
    save_comparison_gif,
)

# Main runner
from .run_comparison import run_comparison

__all__ = [
    # Config
    "ComparisonCase",
    "ComparisonConfig",
    "PlotMode",
    "make_output_paths",
    # Presets
    "get_preset",
    "list_presets",
    "get_preset_names",
    "PRESETS",
    "COMPARISON_CASES",
    "COMPARISON_CONFIGS",
    # Plotting
    "SimulationData",
    "load_shussman_data",
    "load_hydro_history",
    "plot_comparison_single_time",
    "plot_comparison_slider",
    "plot_comparison_overlay",
    "save_comparison_gif",
    # Runner
    "run_comparison",
]

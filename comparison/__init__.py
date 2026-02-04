# comparison/__init__.py
"""
Comparison module for driven shock simulations.
Compares results from hydro_sim (Lagrangian simulation) with
shussman_shock_solver (self-similar solution).
"""

from .shock_config import ShockComparisonConfig, gold_constant_drive, gold_power_law_drive
from .compare_shock_plots import (
    SimulationData,
    load_shussman_data,
    load_hydro_history,
    plot_comparison_single_time,
    plot_comparison_slider,
    plot_comparison_overlay,
    save_comparison_gif,
)

__all__ = [
    "ShockComparisonConfig",
    "gold_constant_drive",
    "gold_power_law_drive",
    "SimulationData",
    "load_shussman_data",
    "load_hydro_history",
    "plot_comparison_single_time",
    "plot_comparison_slider",
    "plot_comparison_overlay",
    "save_comparison_gif",
]

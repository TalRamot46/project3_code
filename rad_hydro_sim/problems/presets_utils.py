# problems/presets.py
"""
Predefined simulation presets.

Each preset combines a physical problem case with appropriate numerical settings.
This provides ready-to-run configurations for common test problems.

Usage:
    from problems.presets import get_preset, list_presets
    
    case, config = get_preset("sedov_spherical")
"""
from typing import Tuple, Dict

from project_3.hydro_sim.problems.simulation_config import (
    SimulationConfig,
    SIMULATION_CONFIGS,
    get_config,
)
from project_3.hydro_sim.problems.Hydro_case import HydroCase
from project_3.hydro_sim.problems.riemann_problem import RIEMANN_TEST_CASES
from project_3.hydro_sim.problems.driven_shock_problem import DRIVEN_SHOCK_TEST_CASES
from project_3.hydro_sim.problems.sedov_problem import SEDOV_TEST_CASES
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from presets_config import PRESETS
# ============================================================================
# Preset Access Functions
# ============================================================================

def get_preset(name: str) -> Tuple[RadHydroCase, SimulationConfig]:
    """
    Get a predefined test case and simulation configuration.
    
    Parameters:
        name: Preset name (see PRESETS dict or call list_presets())
        
    Returns:
        case: ProblemCase instance with physical parameters
        config: SimulationConfig with numerical parameters
        
    Raises:
        ValueError: If preset name is not found
            """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> None:
    """Print all available presets grouped by problem type."""
    print("Available presets:")
    print("=" * 70)
    
    # Group by prefix
    group1 = [(k, v) for k, v in PRESETS.items() if k.startswith("riemann")]
    
    def print_group(title, items):
        print(f"\n{title}:")
        print("-" * 70)
        for name, (case, config) in sorted(items):
            print(f"  {name:25s} - {case.title:30s} (N={config.N})")
    
    print_group("Group1", group1)
    print()


def get_preset_names() -> list:
    """Return list of all available preset names."""
    return sorted(PRESETS.keys())

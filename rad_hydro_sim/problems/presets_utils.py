# problems/presets.py
"""
Predefined simulation presets.

Each preset combines a physical problem case with appropriate numerical settings.
This provides ready-to-run configurations for common test problems.

Usage:
    from problems.presets import get_preset, list_presets
    
    case, config = get_preset("sedov_spherical")
"""
from typing import Tuple

from project_3.hydro_sim.problems.simulation_config import SimulationConfig
from project_3.rad_hydro_sim.problems.RadHydroCase import RadHydroCase
from project_3.rad_hydro_sim.problems.presets_config import PRESETS, FULL_RAD_HYDRO_PRESET_NAMES
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
    """Print all available presets grouped by scenario (from case.scenario)."""
    print("Available rad_hydro presets (physical case names):")
    print("=" * 70)

    def print_group(title: str, items: list) -> None:
        if not items:
            return
        print(f"\n  {title}:")
        for name, (case, config) in sorted(items):
            title_str = case.title or "(no title)"
            print(f"    {name:42s}  {title_str:40s} N={config.N}")

    hydro = [(k, v) for k, v in PRESETS.items() if getattr(v[0], "scenario", "") == "hydro_only"]
    rad_only = [(k, v) for k, v in PRESETS.items() if getattr(v[0], "scenario", "") == "radiation_only"]
    full_rh = [(k, v) for k, v in PRESETS.items() if k in FULL_RAD_HYDRO_PRESET_NAMES]
    print_group("Hydro-only (driven shock)", hydro)
    print_group("Radiation-only (vs 1D Diffusion)", rad_only)
    print_group("Full rad-hydro", full_rh)
    print()


def get_preset_names() -> list:
    """Return list of all available preset names."""
    return sorted(PRESETS.keys())

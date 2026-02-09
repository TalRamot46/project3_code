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

from .simulation_config import SimulationConfig, SIMULATION_CONFIGS, get_config
from .Hydro_case import HydroCase
from .riemann_problem import RIEMANN_TEST_CASES
from .driven_shock_problem import DRIVEN_SHOCK_TEST_CASES
from .sedov_problem import SEDOV_TEST_CASES


# ============================================================================
# Preset Configurations
# ============================================================================

# All available presets: maps preset name -> (case, config)
PRESETS: Dict[str, Tuple[HydroCase, SimulationConfig]] = {
    # -------------------------------------------------------------------------
    # Riemann shock tube presets
    # -------------------------------------------------------------------------
    "riemann_sod": (
        RIEMANN_TEST_CASES["sod"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "riemann_strong": (
        RIEMANN_TEST_CASES["strong_shock"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "riemann_reverse": (
        RIEMANN_TEST_CASES["reverse_shock"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "riemann_colliding": (
        RIEMANN_TEST_CASES["colliding"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    
    # -------------------------------------------------------------------------
    # Driven shock presets
    # -------------------------------------------------------------------------
    "shock_constant": (
        DRIVEN_SHOCK_TEST_CASES["constant_drive"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "shock_linear": (
        DRIVEN_SHOCK_TEST_CASES["linear_drive"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "shock_gold": (
        DRIVEN_SHOCK_TEST_CASES["gold_wall"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "shock_gold_continuous": (
        DRIVEN_SHOCK_TEST_CASES["gold_wall_continuous"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    
    # -------------------------------------------------------------------------
    # Sedov-Taylor explosion presets
    # -------------------------------------------------------------------------
    "sedov_spherical": (
        SEDOV_TEST_CASES["spherical"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "sedov_cylindrical": (
        SEDOV_TEST_CASES["cylindrical"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "sedov_planar": (
        SEDOV_TEST_CASES["planar"],
        SIMULATION_CONFIGS["gif_only"],
    ),
    "sedov_strong": (
        SEDOV_TEST_CASES["strong_spherical"],
        SIMULATION_CONFIGS["gif_only"],
    ),
}


# ============================================================================
# Preset Access Functions
# ============================================================================

def get_preset(name: str) -> Tuple[HydroCase, SimulationConfig]:
    """
    Get a predefined test case and simulation configuration.
    
    Parameters:
        name: Preset name (see PRESETS dict or call list_presets())
        
    Returns:
        case: ProblemCase instance with physical parameters
        config: SimulationConfig with numerical parameters
        
    Raises:
        ValueError: If preset name is not found
        
    Example:
        case, config = get_preset("sedov_spherical")
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
    riemann = [(k, v) for k, v in PRESETS.items() if k.startswith("riemann")]
    shock = [(k, v) for k, v in PRESETS.items() if k.startswith("shock")]
    sedov = [(k, v) for k, v in PRESETS.items() if k.startswith("sedov")]
    
    def print_group(title, items):
        print(f"\n{title}:")
        print("-" * 70)
        for name, (case, config) in sorted(items):
            print(f"  {name:25s} - {case.title:30s} (N={config.N})")
    
    print_group("Riemann Shock Tube", riemann)
    print_group("Driven Shock", shock)
    print_group("Sedov-Taylor Explosion", sedov)
    print()


def get_preset_names() -> list:
    """Return list of all available preset names."""
    return sorted(PRESETS.keys())

# comparison/presets.py
"""
Predefined comparison presets.

Each preset combines a physical ComparisonCase with appropriate run configuration.
This provides ready-to-run configurations for common comparison scenarios.

Usage:
    from comparison.presets import get_preset, list_presets
    
    case, config = get_preset("gold_tau_0")
"""
from typing import Tuple, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from project_3.hydro_sim.verification.comparison_config import (
    ComparisonCase,
    ComparisonConfig,
    PlotMode,
)
from project_3.shussman_solvers.shock_solver.materials_shock import au_supersonic_variant_1


# ============================================================================
# Predefined Comparison Cases
# ============================================================================

COMPARISON_CASES: Dict[str, ComparisonCase] = {
    # -------------------------------------------------------------------------
    # Gold with constant pressure drive (τ=0)
    # -------------------------------------------------------------------------
    "gold_tau_0": ComparisonCase(
        gamma=1.25,
        rho0=19.32,
        P0_phys_Barye=1e12,
        tau=0.0,
        Pw=(2.0, 0.0, 0.0),
        t_end=1e-9,
        x_max=5e-3 / 19.32,
        n_snapshots=10000,
        material=au_supersonic_variant_1(),
        title="Gold τ=0 (constant drive)",
    ),
    
    # -------------------------------------------------------------------------
    # Gold with linear pressure drive (τ=1)
    # -------------------------------------------------------------------------
    "gold_tau_1": ComparisonCase(
        gamma=1.25,
        rho0=19.32,
        P0_phys_Barye=10.0,
        tau=1.0,
        Pw=(2.0, 0.0, 1.0),  # Pw2 = tau for self-similar
        t_end=5e-3,
        x_max=3e-3 / 19.32,
        n_snapshots=10000,
        material=au_supersonic_variant_1(),
        title="Gold τ=1 (linear drive)",
    ),
    
    # -------------------------------------------------------------------------
    # Gold with decaying pressure (τ=-0.447)
    # -------------------------------------------------------------------------
    "gold_tau_neg": ComparisonCase(
        gamma=1.25,
        rho0=19.32,
        P0_phys_Barye=2.71e12,
        tau=-0.45,
        Pw=(0.0, 0.0, -0.45),
        t_end=1e-9,
        x_max=15e-3 / 19.32,
        n_snapshots=10000,
        material=au_supersonic_variant_1(),
        title="Gold τ=-0.447 (decaying drive)",
    ),
    
    # -------------------------------------------------------------------------
    # High-resolution gold constant drive
    # -------------------------------------------------------------------------
    "gold_tau_0_hires": ComparisonCase(
        gamma=1.25,
        rho0=19.32,
        P0_phys_Barye=10.0,
        tau=0.0,
        Pw=(2.0, 0.0, 0.0),
        t_end=100e-9,
        x_max=3e-6 / 19.32,
        n_snapshots=20000,
        material=au_supersonic_variant_1(),
        title="Gold τ=0 high-res",
    ),
}


# ============================================================================
# Predefined Configurations
# ============================================================================

COMPARISON_CONFIGS: Dict[str, ComparisonConfig] = {
    "default": ComparisonConfig(
        N=500,
        CFL=0.2,
        sigma_visc=1.0,
        mode=PlotMode.SLIDER,
        show_plot=True,
    ),
    "gif_only": ComparisonConfig(
        N=500,
        CFL=0.2,
        sigma_visc=1.0,
        mode=PlotMode.GIF,
        show_plot=False,
        save_gif=True,
    ),
    "all_outputs": ComparisonConfig(
        N=500,
        CFL=0.2,
        sigma_visc=1.0,
        mode=PlotMode.SLIDER,
        show_plot=True,
        save_png=True,
        save_gif=True,
    ),
}


# ============================================================================
# Combined Presets
# ============================================================================

PRESETS: Dict[str, Tuple[ComparisonCase, ComparisonConfig]] = {
    # Standard cases with default config
    "gold_tau_0": (
        COMPARISON_CASES["gold_tau_0"],
        COMPARISON_CONFIGS["all_outputs"],
    ),
    "gold_tau_1": (
        COMPARISON_CASES["gold_tau_1"],
        COMPARISON_CONFIGS["all_outputs"],
    ),
    "gold_tau_neg": (
        COMPARISON_CASES["gold_tau_neg"],
        COMPARISON_CONFIGS["all_outputs"],
    ),
        
    # GIF output versions
    "gold_tau_0_gif": (
        COMPARISON_CASES["gold_tau_0"],
        COMPARISON_CONFIGS["gif_only"],
    ),
    "gold_tau_neg_gif": (
        COMPARISON_CASES["gold_tau_neg"],
        COMPARISON_CONFIGS["gif_only"],
    ),
}


# ============================================================================
# Preset Access Functions
# ============================================================================

def get_preset(name: str) -> Tuple[ComparisonCase, ComparisonConfig]:
    """
    Get a predefined comparison case and configuration.
    
    Parameters:
        name: Preset name (see PRESETS dict or call list_presets())
        
    Returns:
        case: ComparisonCase instance with physical parameters
        config: ComparisonConfig with run-time options
        
    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> None:
    """Print all available comparison presets."""
    print("Available comparison presets:")
    print("=" * 70)
    
    for name, (case, config) in sorted(PRESETS.items()):
        mode_str = config.mode.value
        print(f"  {name:25s} - {case.title:30s} (N={config.N}, mode={mode_str})")
    print()


def get_preset_names() -> list:
    """Return list of all available preset names."""
    return sorted(PRESETS.keys())

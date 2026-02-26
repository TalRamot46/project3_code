# problems/simulation_config.py
"""
Simulation configuration - parameters for the numerical solver.

This module separates numerical/solver parameters from physical problem parameters.
Physical parameters belong in ProblemCase subclasses (e.g., RiemannCase, SedovExplosionCase).

Structure mirrors comparison/comparison_config.py:
  - SimulationConfig: Numerical solver parameters (N, CFL, etc.)
  - SIMULATION_CONFIGS: Predefined numerical configurations
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from pathlib import Path
from enum import Enum


class ProblemType(str, Enum):
    """Supported physical problem types."""
    RIEMANN = "riemann"
    DRIVEN_SHOCK = "driven_shock"
    SEDOV = "sedov"


# ============================================================================
# Output Path Generation (all outputs under results/)
# ============================================================================

def get_results_dir() -> Path:
    """Base directory for simulation outputs (figures, data). All runs write under results/."""
    root = Path(__file__).resolve().parent.parent.parent
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_figures_dir() -> Path:
    """Get the base figures directory for hydro_sim outputs (under results/)."""
    base = get_results_dir() / "hydro_sim"
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_output_paths(case_name: str) -> Tuple[Path, Path]:
    """
    Generate output paths for PNG and GIF files based on case name.
    
    Parameters:
        case_name: Name or title of the simulation case
        
    Returns:
        (png_path, gif_path) tuple with full paths
    """
    base_dir = get_figures_dir()
    png_dir = base_dir / "png"
    gif_dir = base_dir / "gif"
    
    # Ensure directories exist
    png_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean case name for filename
    safe_name = (case_name
                 .replace(" ", "_")
                 .replace("=", "")
                 .replace("(", "")
                 .replace(")", "")
                 .replace(",", "")
                 .replace(":", ""))
    
    return png_dir / f"{safe_name}.png", gif_dir / f"{safe_name}.gif"


# ============================================================================
# Simulation Configuration
# ============================================================================

@dataclass
class SimulationConfig:
    """Configuration parameters for hydrodynamic simulations."""
    # Numerical parameters
    N: int = 1000
    CFL: float = 1/3
    sigma_visc: float = 1.0
    store_every: int = 100
    
    # Output parameters
    save_path: Optional[str] = None
    gif_path: Optional[str] = None
    show_plot: bool = True
    show_slider: bool = False
    # Fraction of t_end at which to take the PNG snapshot (0.0..1.0)
    png_time_frac: float = 0.5
    
    def with_output_paths(self, case_name: str) -> "SimulationConfig":
        """
        Return a new config with auto-generated output paths based on case name.
        
        Parameters:
            case_name: Name/title of the case for generating filenames
            
        Returns:
            New SimulationConfig with save_path and gif_path set
        """
        png_path, gif_path = make_output_paths(case_name)
        return SimulationConfig(
            N=self.N,
            CFL=self.CFL,
            sigma_visc=self.sigma_visc,
            store_every=self.store_every,
            save_path=str(png_path),
            gif_path=str(gif_path),
            show_plot=self.show_plot,
            show_slider=self.show_slider,
            png_time_frac=self.png_time_frac,
        )
    
    def _get_parameters(self) -> Tuple[int, float, float, int, Optional[str], Optional[str], bool, bool]:
        """Return parameters as a tuple (for easy unpacking)."""
        return (
            self.N,
            self.CFL,
            self.sigma_visc,
            self.store_every,
            self.save_path,
            self.gif_path,
            self.show_plot,
            self.show_slider,
        )


# ============================================================================
# Predefined Simulation Configurations
# ============================================================================
# Lightweight: all_outputs = slider + PNG at png_time_frac * t_end.
# Override N, store_every, png_time_frac manually when running a test.
# gif_only = hydro_sim presets that only need GIF output.
# ============================================================================

SIMULATION_CONFIGS: Dict[str, SimulationConfig] = {
    # All outputs: slider + PNG at png_time_frac * t_end (default 0.5)
    "all_outputs": SimulationConfig(
        N=200,
        CFL=1/3,
        sigma_visc=1.0,
        store_every=10,
        show_plot=True,
        show_slider=True,
        png_time_frac=0.5,
    ),
    # GIF output only (used by hydro_sim presets)
    "gif_only": SimulationConfig(
        N=500,
        CFL=1/3,
        sigma_visc=1.0,
        store_every=50,
        show_plot=False,
    ),
}


def get_config(name: str) -> SimulationConfig:
    """
    Get a predefined simulation configuration by name.
    
    Parameters:
        name: Configuration name (see SIMULATION_CONFIGS)
        
    Returns:
        SimulationConfig instance
        
    Raises:
        ValueError: If name not found
    """
    if name not in SIMULATION_CONFIGS:
        available = ", ".join(sorted(SIMULATION_CONFIGS.keys()))
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return SIMULATION_CONFIGS[name]
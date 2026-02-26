# comparison/comparison_config.py
"""
Configuration classes for shock comparison simulations.

This module provides:
  - ComparisonCase: Extends DrivenShockCase with material info for self-similar solver
  - ComparisonConfig: Run-time options (mode, skip_sim, output paths, etc.)

The design mirrors hydro_sim's separation of physical (Case) and numerical (Config) parameters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime
import numpy as np
import sys

# Ensure project_3 is on path (hydro_shock is at project_3/rad_hydro_sim/verification/hydro_shock)
_project3 = Path(__file__).resolve().parent.parent.parent.parent
if str(_project3) not in sys.path:
    sys.path.insert(0, str(_project3))

from project_3.hydro_sim.problems.driven_shock_problem import DrivenShockCase
from project_3.hydro_sim.core.geometry import planar
from project_3.shussman_solvers.shock_solver.materials_shock import Material, au_supersonic_variant_1


class PlotMode(str, Enum):
    """Available plotting modes for comparison."""
    SLIDER = "slider"
    SINGLE = "single"
    OVERLAY = "overlay"
    GIF = "gif"


# ============================================================================
# Output Path Generation (under results/)
# ============================================================================

def get_output_dir() -> Path:
    """Get the base output directory for hydro_sim shock verification figures."""
    from project_3.hydro_sim.problems.simulation_config import get_results_dir
    base = get_results_dir() / "hydro_sim_verification"
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_output_paths(case_name: str) -> Tuple[Path, Path]:
    """
    Generate output paths for PNG and GIF files based on case name.
    
    Returns:
        (png_path, gif_path) tuple
    """
    base_dir = get_output_dir()
    png_dir = base_dir / "png"
    gif_dir = base_dir / "gif"
    
    # Ensure directories exist
    png_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean case name for filename and append timestamp so runs don't overwrite
    safe_name = case_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}"
    
    return png_dir / f"{filename}.png", gif_dir / f"{filename}.gif"


# ============================================================================
# Physical Case: ComparisonCase
# ============================================================================

@dataclass(frozen=True)
class ComparisonCase:
    """
    Physical parameters for shock comparison.
    
    Extends DrivenShockCase parameters with material info for self-similar solver.
    This is the "physics" part of the configuration.
    
    Attributes:
        # Inherited from DrivenShockCase concept:
        gamma: Adiabatic index
        x_min, x_max: Domain boundaries
        t_end: Final simulation time
        rho0: Initial density
        p0: Background pressure (for hydro_sim)
        u0: Initial velocity
        P0: Driving pressure amplitude
        tau: Power-law exponent for p(t) = P0 * t^tau
        
        # Additional for comparison:
        material: Material properties for self-similar solver
        Pw: Wall pressure parameters (Pw0, Pw1, Pw2) for self-similar solver
        n_snapshots: Number of time snapshots for comparison
        title: Case description
    """
    # Physical parameters (same as DrivenShockCase)
    gamma: float = 5/3
    x_min: float = 0.0
    x_max: float = None  # Auto-computed if None
    t_end: float = 100e-9
    rho0: float = 19.32  # Gold density
    p0: float = 1e-3     # Small background pressure
    u0: float = 0.0
    P0_phys_Barye: float = 10.0
    tau: float = 0.0
    
    # Self-similar solver parameters
    material: Material = field(default_factory=au_supersonic_variant_1)
    Pw: Tuple[float, float, float] = (2.0, 0.0, 0.0)
    
    # Common parameters
    n_snapshots: int = 10000
    title: str = "Shock Comparison"
    
    def __post_init__(self):
        # Auto-compute x_max based on density if not provided
        if self.x_max is None:
            object.__setattr__(self, 'x_max', 3e-6 / self.rho0)
    
    @property
    def times(self) -> np.ndarray:
        """Generate snapshot times for comparison."""
        return np.linspace(0.1 * self.t_end, self.t_end, self.n_snapshots)
    
    def to_driven_shock_case(self) -> DrivenShockCase:
        """Convert to DrivenShockCase for hydro_sim."""
        return DrivenShockCase(
            title=self.title,
            gamma=self.gamma,
            x_min=self.x_min,
            x_max=self.x_max,
            t_end=self.t_end,
            geom=planar(),
            rho0=self.rho0,
            p0=self.p0,
            u0=self.u0,
            tau=self.tau,
            P0=self.P0_phys_Barye,
        )
    
    def get_shussman_params(self) -> dict:
        """Get parameters for shussman_shock_solver."""
        return {
            'material': self.material,
            'P0_phys_Barye': self.P0_phys_Barye,
            'Pw': list(self.Pw) if self.Pw is not None else None,
            'times': self.times,
            'tau': self.tau if self.tau is not None else None,
        }
    
    @property
    def output_paths(self) -> Tuple[Path, Path]:
        """Get (png_path, gif_path) for this case."""
        return make_output_paths(self.title)


# ============================================================================
# Run Configuration: ComparisonConfig
# ============================================================================

@dataclass
class ComparisonConfig:
    """
    Run-time configuration for comparison simulations.
    
    This is the "numerical/runtime" part of the configuration,
    analogous to SimulationConfig in hydro_sim.
    
    Attributes:
        # Numerical parameters
        N: Number of cells for hydro simulation
        CFL: CFL number
        sigma_visc: Artificial viscosity coefficient
        store_every: How often to store frames
        
        # Run options
        mode: Plotting mode (slider, single, overlay, gif)
        xaxis: X-axis variable ("m" for mass, "x" for position)
        skip_sim: Skip running hydro simulation
        skip_solver: Skip running self-similar solver
        npz_path: Path to existing NPZ file (if skip_solver=True)
        
        # Output options
        show_plot: Whether to show interactive plots
        save_png: Whether to save PNG
        save_gif: Whether to save GIF
        time_for_single: Time for single-frame plot (None = t_end)
    """
    # Numerical parameters
    N: int = 500
    CFL: float = 0.2
    sigma_visc: float = 1.0
    store_every: int = 100
    
    # Run options
    mode: PlotMode = PlotMode.SLIDER
    xaxis: str = "m"
    skip_sim: bool = False
    skip_solver: bool = False
    npz_path: Optional[str] = None
    
    # Output options
    show_plot: bool = True
    save_png: bool = False
    save_gif: bool = False
    time_for_single: Optional[float] = None

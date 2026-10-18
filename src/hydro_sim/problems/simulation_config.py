# problems/simulation_config.py
"""
Simulation configuration - parameters for the numerical solver.

This module separates numerical/solver parameters from physical problem parameters.
Physical parameters belong in ProblemCase subclasses (e.g., RiemannCase, SedovExplosionCase).

Structure mirrors comparison/comparison_config.py:
  - SimulationConfig: Numerical solver parameters (N, CFL, etc.)
  - SIMULATION_CONFIGS: Predefined numerical configurations
"""
from dataclasses import dataclass, field, replace
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

    # --- IMC plan, Phase B (md/phase_b_diffusion_changes.md). Every default
    # reproduces the original behaviour; rad_hydro_sim reads these. ---
    # Material update after the gray diffusion solve: "u_r" sets
    # aT^4 = (A E + aT*^4)/(1+A) (original); "conservative" adds exactly the
    # energy the radiation lost, F&C (1.18).
    material_update: str = "u_r"
    # Store the material energy as u_m = f T^beta rho^(1-mu) - a T^4, reading
    # Rosen's e as matter + equilibrium radiation (manuscript Eq. um_def).
    subtract_radiation_energy: bool = False
    # Carry E_rad per unit mass through the hydro step (manuscript d(vE_r)/dt):
    # E_rad <- E_rad V_old / V_new.  Original: E_rad unchanged.
    rad_energy_per_mass: bool = False
    # Right face of the radiation solve: "ghost" (original: E = 0 in a ghost
    # cell one face-spacing away) or "marshak_vacuum" (outgoing flux c E / 2).
    right_rad_bc: str = "ghost"
    # Hydro energy update: "standard" (original predictor) or "compatible"
    # (cell work equals the nodes' kinetic-energy change: total energy is
    # conserved to round-off; planar only).
    hydro_energy: str = "standard"
    # Record the total-energy budget every step in meta["energy_budget"].
    energy_diagnostics: bool = False
    # If set, dt ignores the E_rad / T_rad relative-change control and ramps
    # by 1.1 per step up to this cap (s); CFL still applies with hydro.  Gives
    # identical dt sequences to the diffusion and IMC solvers.
    dt_ramp_max: Optional[float] = None
    # Radiation solver: "diffusion" (original) or "imc" (src/imc, radiation_only).
    radiation_solver: str = "diffusion"
    imc_n_emission: int = 20000      # target emission packets per step
    imc_n_bath: int = 20000          # bath packets per step
    imc_max_census: int = 200000     # comb trigger
    imc_seed: int = 1

    def with_output_paths(self, case_name: str) -> "SimulationConfig":
        """
        Return a new config with auto-generated output paths based on case name.

        Parameters:
            case_name: Name/title of the case for generating filenames

        Returns:
            New SimulationConfig with save_path and gif_path set
        """
        png_path, gif_path = make_output_paths(case_name)
        # replace() keeps every other field, including the Phase B options.
        return replace(self, save_path=str(png_path), gif_path=str(gif_path))
    
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
        N=100,
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
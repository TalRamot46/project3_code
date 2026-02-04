# shock_config.py
"""
Shared configuration for driven shock simulations and self-similar solutions.
Both shussman_shock_solver and hydro_sim use these parameters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import from shussman_shock_solver
sys.path.insert(0, str(Path(__file__).parent.parent))

from shussman_shock_solver.materials_shock import Material, au_supersonic_variant_1


@dataclass(frozen=True)
class ShockComparisonConfig:
    """
    Unified configuration for comparing hydro simulation with self-similar solver.
    
    Attributes:
        material: Material properties for self-similar solver
        P0: Driving pressure amplitude (dyne/cm^2)
        tau: Power-law exponent for time-dependent pressure p(t) = P0 * t^tau
        Pw: Optional wall pressure parameters for self-similar [Pw0, Pw1, Pw2]
        rho0: Initial density (g/cm^3)
        p0_background: Background pressure ahead of shock (for hydro_sim)
        gamma: Adiabatic index (ideal gas EOS for hydro_sim)
        x_min: Left domain boundary
        x_max: Right domain boundary
        t_end: Final simulation time
        times: Array of snapshot times for comparison
        Ncells: Number of cells for hydro simulation
        CFL: CFL number for hydro simulation
        sigma_visc: Artificial viscosity coefficient
    """
    # Material and pressure drive
    material: Material
    P0: float = 10.0
    tau: float = 0.0
    Pw: tuple = (2.0, 0.0, 0.0)  # Wall pressure parameters for self-similar
    
    # Initial conditions
    rho0: float = 19.32  # Gold density g/cm^3
    p0_background: float = 1e-3  # Small background pressure for hydro_sim
    gamma: float = 1.25
    
    # Domain
    x_min: float = 0.0
    x_max: float = None  # Set based on rho0 if None
    
    # Time
    t_end: float = 100e-9  # 100 ns
    times: tuple = None  # Snapshot times, auto-generated if None
    n_snapshots: int = 10  # Number of snapshot times
    
    # Simulation parameters
    Ncells: int = 1001
    CFL: float = 0.2
    sigma_visc: float = 1.0
    
    # Plotting options
    title: str = "Driven Shock Comparison"
    
    def __post_init__(self):
        # Set x_max based on domain size if not provided
        if self.x_max is None:
            # Use a domain that's 3 microns divided by density
            object.__setattr__(self, 'x_max', 3e-6 / self.rho0)
        
        # Generate snapshot times if not provided
        if self.times is None:
            # Logarithmically spaced times from 1% of t_end to t_end
            times = np.linspace(0.1 * self.t_end, self.t_end, self.n_snapshots)
            object.__setattr__(self, 'times', tuple(times))
    
    def to_driven_shock_case(self):
        """Convert to DrivenShockCase for hydro_sim."""
        from hydro_sim.problems.driven_shock_problem import DrivenShockCase
        return DrivenShockCase(
            title=self.title,
            x_min=self.x_min,
            x_max=self.x_max,
            t_end=self.t_end,
            rho0=self.rho0,
            p0=self.p0_background,
            u0=0.0,
            gamma=self.gamma,
            tau=self.tau,
            P0=self.P0,
        )
    
    def get_shussman_params(self):
        """Get parameters for shussman_shock_solver."""
        return {
            'material': self.material,
            'P0': self.P0,
            'Pw': list(self.Pw),
            'times': np.array(self.times),
        }


# ============================================================================
# Predefined Configurations
# ============================================================================

def gold_constant_drive() -> ShockComparisonConfig:
    """Gold with constant pressure drive (tau=0)."""
    return ShockComparisonConfig(
        material=au_supersonic_variant_1(),
        P0=10.0,
        tau=0.0,
        rho0=19.32,
        t_end=100e-9,
        title="Gold - Constant Drive (τ=0)",
    )


def gold_power_law_drive(tau: float = 0.5) -> ShockComparisonConfig:
    """Gold with power-law pressure drive."""
    return ShockComparisonConfig(
        material=au_supersonic_variant_1(),
        P0=10.0,
        tau=tau,
        rho0=19.32,
        t_end=100e-9,
        title=f"Gold - Power Law Drive (τ={tau})",
    )


# Dictionary of available configurations
SHOCK_CONFIGS = {
    "gold_constant": gold_constant_drive,
    "gold_power_law": gold_power_law_drive,
}

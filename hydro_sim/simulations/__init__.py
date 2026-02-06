# simulations/__init__.py
"""
Simulation runners for hydrodynamic problems.
"""
from .driven_shock_sim import (
    simulate_lagrangian,
    SimulationHistory,
    ShockHistory,  # Alias for backward compatibility
    SimulationType,
)
from .riemann_exact import sample_solution, solve_star_region

__all__ = [
    # Unified simulation
    "simulate_lagrangian",
    "SimulationHistory",
    "SimulationType",
    # Backward compatibility
    "ShockHistory",
    # Exact solutions
    "sample_solution",
    "solve_star_region",
]
